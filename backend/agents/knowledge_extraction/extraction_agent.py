"""
GraphMASAL · Knowledge Extraction Agent
========================================
Extracts concepts and prerequisite relationships from educational text
using a two-step LLM pipeline powered entirely by Google Gemini.

Pipeline:
    1. Extract concepts (Gemini 2.5 Flash → structured JSON)
    2. Embed concept names (gemini-embedding-001 · batched)
    3. Extract relationships (Gemini 2.5 Flash → structured JSON)

Environment variables required:
    GOOGLE_API_KEY
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

import google.generativeai as genai

from .models import Concept, ConceptList, KnowledgeGraph, Relationship, RelationshipList
from .prompts import CONCEPT_EXTRACTION_SYSTEM_PROMPT, RELATIONSHIP_EXTRACTION_SYSTEM_PROMPT
from .utils import setup_logger

logger = setup_logger(__name__)

EXTRACTION_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"


def _clean_json(raw: str) -> str:
    """Strip markdown fences if Gemini wraps output in ```json ... ```."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


class KnowledgeExtractionAgent:
    """
    Extracts a knowledge graph (concepts + prerequisite relationships)
    from raw educational text using Gemini 2.5 Flash.
    """

    def __init__(self, model_name: str = EXTRACTION_MODEL):
        self.model_name = model_name
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY is not set")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        logger.info("Initialized KnowledgeExtractionAgent with model %s", model_name)

    # ------------------------------------------------------------------
    # Step 1 — Concept extraction
    # ------------------------------------------------------------------

    def extract_concepts(self, text: str) -> List[Concept]:
        """Call Gemini to extract concepts from raw text. Returns a list of Concept objects."""
        logger.info("Starting concept extraction...")

        user_prompt = (
            f"Extract educational concepts from the following text:\n\n{text}"
        )

        full_prompt = f"{CONCEPT_EXTRACTION_SYSTEM_PROMPT}\n\n{user_prompt}"

        response = self.model.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=4096,
            ),
        )

        raw = _clean_json(response.text)

        try:
            data = json.loads(raw)
            parsed = ConceptList(**data)
            concepts = parsed.concepts
        except Exception as e:
            logger.error("Failed to parse concept extraction response: %s\nRaw: %s", e, raw[:500])
            concepts = []

        logger.info("Extracted %d concepts.", len(concepts))
        return concepts

    # ------------------------------------------------------------------
    # Step 2 — Embedding (batched)
    # ------------------------------------------------------------------

    def embed_concepts(self, concepts: List[Concept]) -> List[Concept]:
        """Embed all concept names in a single batched call using gemini-embedding-001."""
        if not concepts:
            return concepts

        names = [c.name for c in concepts]
        logger.info("Embedding %d concept names with %s...", len(names), EMBEDDING_MODEL)

        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=names,
            task_type="SEMANTIC_SIMILARITY",
        )

        for concept, embedding_vector in zip(concepts, result["embedding"]):
            concept.embedding = embedding_vector

        dim = len(result["embedding"][0])
        logger.info("Embedded %d concepts (dim=%d).", len(concepts), dim)
        return concepts

    # ------------------------------------------------------------------
    # Step 3 — Relationship extraction
    # ------------------------------------------------------------------

    def extract_relationships(self, text: str, concepts: List[Concept]) -> List[Relationship]:
        """Call Gemini to infer prerequisite relationships between extracted concepts."""
        logger.info("Starting relationship extraction...")

        concepts_context = "\n".join(
            f"- ID: {c.id}, Name: {c.name}, Description: {c.description}"
            for c in concepts
        )

        user_prompt = (
            f"Given the following original text:\n\n{text}\n\n"
            f"And the following extracted concepts:\n\n{concepts_context}\n\n"
            "Identify prerequisite relationships between these concepts based strictly on the text."
        )

        full_prompt = f"{RELATIONSHIP_EXTRACTION_SYSTEM_PROMPT}\n\n{user_prompt}"

        response = self.model.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=4096,
            ),
        )

        raw = _clean_json(response.text)

        try:
            data = json.loads(raw)
            parsed = RelationshipList(**data)
            relationships = parsed.relationships
        except Exception as e:
            logger.error("Failed to parse relationship extraction response: %s\nRaw: %s", e, raw[:500])
            relationships = []

        logger.info("Extracted %d relationships.", len(relationships))
        return relationships

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Full extraction pipeline:
            1. Extract concepts
            2. Embed concept names
            3. Extract relationships
            4. Return as structured dict

        Parameters
        ----------
        text : str
            Raw educational text extracted from a PDF/DOCX/PPTX etc.

        Returns
        -------
        dict
            KnowledgeGraph serialised as a dict:
            ``{ "concepts": [...], "relationships": [...] }``
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for extraction.")
            return KnowledgeGraph(concepts=[], relationships=[]).model_dump()

        try:
            # Step 1 — extract concepts
            concepts = self.extract_concepts(text)

            # Step 2 — embed concept names
            concepts = self.embed_concepts(concepts)

            # Step 3 — extract relationships
            relationships = []
            if len(concepts) > 1:
                relationships = self.extract_relationships(text, concepts)
            elif len(concepts) == 1:
                logger.info("Only 1 concept extracted — skipping relationship extraction.")

            kg = KnowledgeGraph(concepts=concepts, relationships=relationships)
            return kg.model_dump()

        except Exception as e:
            logger.error("Error during knowledge extraction: %s", e)
            raise