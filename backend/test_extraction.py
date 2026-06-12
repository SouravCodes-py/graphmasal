from ingestion.loader_factory import LoaderFactory
import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Load PPT
file_path = "data/Module4_cs part1.pptx"
loader = LoaderFactory.get_loader(file_path)
text = loader.extract_text(file_path)
print(f"Total text length: {len(text)} characters")

# Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def groq_call(prompt: str) -> str:
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content

# Step 1 — Concept extraction in chunks
CHUNK_SIZE = 8000
chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
print(f"Total chunks: {len(chunks)}")

all_concepts = []
seen_names = set()

for i, chunk in enumerate(chunks):
    print(f"Extracting chunk {i+1}/{len(chunks)}...")
    concept_prompt = f"""
Extract educational concepts suitable as nodes in a knowledge graph.
Rules:
- Ignore chapter titles and section headings.
- Ignore duplicate concepts.
- Return atomic concepts only.
- Be exhaustive — extract every distinct concept, sub-concept, and technique mentioned.
- Prefer more concepts over fewer, do not merge related concepts into one.
- Do NOT include real-world product names, brand names, or illustrative examples.
- Return JSON only, no preamble, no markdown fences.

Format:
{{
  "concepts": [
    {{
      "name": "concept name",
      "definition": "one sentence definition",
      "prerequisites": ["concept name"]
    }}
  ]
}}

Text:
{chunk}
"""
    result = json.loads(groq_call(concept_prompt))
    for c in result["concepts"]:
        if c["name"] not in seen_names:
            seen_names.add(c["name"])
            all_concepts.append(c)

concepts_data = {"concepts": all_concepts}
print(f"Raw extraction: {len(concepts_data['concepts'])} concepts")

# Fix dangling prerequisites — drop any prereq not in the extracted concept names
valid_names = {c["name"] for c in concepts_data["concepts"]}
for c in concepts_data["concepts"]:
    c["prerequisites"] = [p for p in c["prerequisites"] if p in valid_names]

# Step 2 — Filter
concept_names = [c["name"] for c in concepts_data["concepts"]]

filter_prompt = f"""
You are cleaning a knowledge graph for an educational course.

From this list of concepts, remove any that are:
- Real-world product names (e.g. washing machines, Wi-Fi routers)
- Brand names or specific devices
- Illustrative examples that are not themselves learnable concepts
- Overly specific implementation details that aren't taught as standalone topics

Keep only concepts a student would actually need to understand and be tested on.

Input:
{json.dumps(concept_names)}

Return JSON only, no preamble, no markdown fences.
Format:
{{
  "concepts": ["concept name", "concept name"]
}}
"""

print("\nStep 2 — Filtering concepts (Groq)...")
filtered = json.loads(groq_call(filter_prompt))
clean_concept_names = set(filtered["concepts"])

concepts_data["concepts"] = [
    c for c in concepts_data["concepts"]
    if c["name"] in clean_concept_names
]
print(f"After filtering: {len(concepts_data['concepts'])} concepts")
print(json.dumps(concepts_data, indent=2))

# Step 3 — Relationship extraction
relationship_prompt = f"""
You are building a knowledge graph from educational material.

These are the ONLY valid concept names — source and target must
both come from this list exactly:
{json.dumps(list(clean_concept_names))}

Task: identify prerequisite relationships.
A relationship means: source should be understood before target.
Only return strongly supported relationships. Confidence >= 0.7 only.

Return JSON only, no preamble, no markdown fences.

Format:
{{
  "relationships": [
    {{
      "source": "concept name",
      "target": "concept name",
      "confidence": 0.95
    }}
  ]
}}
"""

print("\nStep 3 — Extracting relationships (Groq)...")
relationships_data = json.loads(groq_call(relationship_prompt))

relationships_data["relationships"] = [
    r for r in relationships_data["relationships"]
    if r["source"] in clean_concept_names
    and r["target"] in clean_concept_names
]

print(f"Extracted {len(relationships_data['relationships'])} relationships")
print(json.dumps(relationships_data, indent=2))

from graph.neo4j_service import Neo4jService

neo4j = Neo4jService()
summary = neo4j.populate_graph(concepts_data, relationships_data, subject="Computer Architecture")
print(f"\nNeo4j write complete: {summary}")
neo4j.close()