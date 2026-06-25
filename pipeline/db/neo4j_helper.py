"""
Neo4j connection helper for pipeline agents.

Thin wrapper around the synchronous Neo4j driver, matching the same
connection pattern Person A uses in backend/graph/neo4j_service.py.
Provides reusable query methods for Diagnostic and Student Modeling agents.

LangGraph runs sync node functions in a thread pool automatically,
so sync driver calls here are safe from FastAPI's async context.
"""

import os
import logging
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase, exceptions

load_dotenv()

logger = logging.getLogger(__name__)

_driver = None


def get_driver():
    """
    Lazy singleton Neo4j driver initialisation.
    Reads NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD from environment.
    """
    global _driver
    if _driver is None:
        uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        user = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        try:
            _driver = GraphDatabase.driver(uri, auth=(user, password))
            _driver.verify_connectivity()
            logger.info("Pipeline Neo4j driver initialised and connected.")
        except exceptions.ServiceUnavailable as e:
            logger.error(f"Cannot connect to Neo4j at {uri}: {e}")
            raise
    return _driver


def run_query(cypher: str, params: dict | None = None) -> list[dict[str, Any]]:
    """
    Execute a read Cypher query and return all results as a list of dicts.

    Example:
        results = run_query(
            "MATCH (c:Concept) WHERE c.mastery_score < $threshold RETURN c.name",
            {"threshold": 0.5}
        )
    """
    driver = get_driver()
    with driver.session() as session:
        result = session.run(cypher, params or {})
        return [record.data() for record in result]


def run_write(cypher: str, params: dict | None = None) -> dict[str, Any] | None:
    """
    Execute a write Cypher query and return the first record (or None).

    Example:
        run_write(
            "MATCH (c:Concept {name: $name}) SET c.mastery_score = $score RETURN c",
            {"name": "Chain Rule", "score": 0.51}
        )
    """
    driver = get_driver()
    with driver.session() as session:
        result = session.run(cypher, params or {})
        record = result.single()
        return record.data() if record else None


def close():
    """Close the driver connection. Called on app shutdown."""
    global _driver
    if _driver:
        _driver.close()
        _driver = None
        logger.info("Pipeline Neo4j driver closed.")
