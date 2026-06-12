import os
from typing import List, Dict, Any
from openai import OpenAI
from pydantic import ValidationError

from .models import Concept, ConceptList, Relationship, RelationshipList, KnowledgeGraph
from .prompts import CONCEPT_EXTRACTION_SYSTEM_PROMPT, RELATIONSHIP_EXTRACTION_SYSTEM_PROMPT
from .utils import setup_logger

logger = setup_logger(__name__)

class KnowledgeExtractionAgent:
    """
    Agent responsible for extracting concepts and prerequisite relationships
    from educational text using a two-step LLM pipeline.
    """
    
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        # Initialize OpenAI client. It reads OPENAI_API_KEY from environment variables automatically.
        self.client = OpenAI()
        logger.info(f"Initialized KnowledgeExtractionAgent with model {self.model_name}")

    def extract_concepts(self, text: str) -> List[Concept]:
        """
        Step 1: Extract concepts from the raw text.
        """
        logger.info("Starting concept extraction...")
        response = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[
                {"role": "system", "content": CONCEPT_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract educational concepts from the following text:\n\n{text}"}
            ],
            response_format=ConceptList,
        )
        
        parsed_response = response.choices[0].message.parsed
        concepts = parsed_response.concepts if parsed_response else []
        logger.info(f"Extracted {len(concepts)} concepts.")
        return concepts

    def extract_relationships(self, text: str, concepts: List[Concept]) -> List[Relationship]:
        """
        Step 2: Infer prerequisite relationships between the extracted concepts based on the text.
        """
        logger.info("Starting relationship extraction...")
        
        # Format the concepts list for the prompt context
        concepts_context = "\n".join([f"- ID: {c.id}, Name: {c.name}, Description: {c.description}" for c in concepts])
        
        user_prompt = (
            f"Given the following original text:\n\n{text}\n\n"
            f"And the following extracted concepts:\n\n{concepts_context}\n\n"
            "Identify prerequisite relationships between these concepts based strictly on the text."
        )

        response = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[
                {"role": "system", "content": RELATIONSHIP_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format=RelationshipList,
        )
        
        parsed_response = response.choices[0].message.parsed
        relationships = parsed_response.relationships if parsed_response else []
        logger.info(f"Extracted {len(relationships)} relationships.")
        return relationships

    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Main pipeline method:
        1. Extract concepts
        2. Extract relationships
        3. Return as a single structured JSON dictionary.
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for extraction.")
            return KnowledgeGraph(concepts=[], relationships=[]).model_dump()
            
        try:
            # Step 1: Extract Concepts
            concepts = self.extract_concepts(text)
            
            # Step 2: Extract Relationships (only if there are multiple concepts)
            relationships = []
            if len(concepts) > 1:
                relationships = self.extract_relationships(text, concepts)
            elif len(concepts) == 1:
                logger.info("Only 1 concept extracted; skipping relationship extraction.")
                
            # Final output aggregation
            kg = KnowledgeGraph(concepts=concepts, relationships=relationships)
            return kg.model_dump()
            
        except Exception as e:
            logger.error(f"Error during knowledge extraction: {e}")
            raise
