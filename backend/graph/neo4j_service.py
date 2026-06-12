import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from neo4j import GraphDatabase, exceptions

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jService:
    """Service class for interacting with the Neo4j database."""

    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        self.uri = uri or os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        self.user = user or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")

        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j database.")
        except exceptions.ServiceUnavailable as e:
            logger.error(f"Failed to connect to Neo4j database: {e}")
            raise

    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed.")

    def create_concept(self, concept_id: str, name: str, description: str, mastery_score: float = 0.0) -> Optional[Dict[str, Any]]:
        query = """
        MERGE (c:Concept {id: $id})
        SET c.name = $name,
            c.description = $description,
            c.mastery_score = $mastery_score
        RETURN c
        """
        parameters = {
            "id": concept_id,
            "name": name,
            "description": description,
            "mastery_score": mastery_score
        }
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                record = result.single()
                if record:
                    return dict(record["c"])
                return None
        except Exception as e:
            logger.error(f"Error creating concept {concept_id}: {e}")
            raise

    def create_prerequisite(self, prerequisite_id: str, target_id: str, weight: float = 1.0) -> bool:
        query = """
        MATCH (p:Concept {id: $prerequisite_id})
        MATCH (t:Concept {id: $target_id})
        MERGE (p)-[r:PREREQUISITE_OF]->(t)
        SET r.weight = $weight
        RETURN r
        """
        parameters = {
            "prerequisite_id": prerequisite_id,
            "target_id": target_id,
            "weight": weight
        }
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                if result.single():
                    return True
                else:
                    logger.warning(f"Could not create prerequisite. Ensure concepts '{prerequisite_id}' and '{target_id}' exist.")
                    return False
        except Exception as e:
            logger.error(f"Error creating prerequisite relationship: {e}")
            raise

    def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        query = """
        MATCH (c:Concept {id: $id})
        RETURN c
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, id=concept_id)
                record = result.single()
                if record:
                    return dict(record["c"])
                return None
        except Exception as e:
            logger.error(f"Error retrieving concept {concept_id}: {e}")
            raise

    def get_prerequisites(self, concept_id: str) -> List[Dict[str, Any]]:
        query = """
        MATCH (p:Concept)-[r:PREREQUISITE_OF]->(c:Concept {id: $id})
        RETURN p, r.weight AS weight
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, id=concept_id)
                prerequisites = []
                for record in result:
                    prereq = dict(record["p"])
                    prereq["relationship_weight"] = record["weight"]
                    prerequisites.append(prereq)
                return prerequisites
        except Exception as e:
            logger.error(f"Error retrieving prerequisites for {concept_id}: {e}")
            raise

    def update_mastery_score(self, concept_id: str, new_score: float) -> Optional[Dict[str, Any]]:
        new_score = max(0.0, min(1.0, new_score))
        query = """
        MATCH (c:Concept {id: $id})
        SET c.mastery_score = $score
        RETURN c
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, id=concept_id, score=new_score)
                record = result.single()
                if record:
                    return dict(record["c"])
                else:
                    logger.warning(f"Concept '{concept_id}' not found for mastery update.")
                    return None
        except Exception as e:
            logger.error(f"Error updating mastery score for {concept_id}: {e}")
            raise

    # ── Graph population ────────────────────────────────────────────────────

    def write_concept_node(self, name: str, definition: str, subject: str = "") -> bool:
        """
        Merge a Concept node by name. Safe to call multiple times — will not
        create duplicates. Used by the Knowledge Extraction Agent.
        """
        query = """
        MERGE (c:Concept {name: $name})
        SET c.definition = $definition,
            c.subject = $subject,
            c.created_at = datetime()
        RETURN c
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, name=name, definition=definition, subject=subject)
                return result.single() is not None
        except Exception as e:
            logger.error(f"Error writing concept node '{name}': {e}")
            raise

    def write_prerequisite_edge(self, source: str, target: str, weight: float = 1.0) -> bool:
        """
        Merge a PREREQUISITE_OF edge between two concepts identified by name.
        source must be understood before target.
        """
        query = """
        MATCH (a:Concept {name: $source})
        MATCH (b:Concept {name: $target})
        MERGE (a)-[r:PREREQUISITE_OF]->(b)
        SET r.weight = $weight
        RETURN r
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, source=source, target=target, weight=weight)
                if result.single():
                    return True
                else:
                    logger.warning(f"Could not create edge '{source}' -> '{target}'. One or both nodes may not exist.")
                    return False
        except Exception as e:
            logger.error(f"Error writing prerequisite edge '{source}' -> '{target}': {e}")
            raise

    def populate_graph(self, concepts_data: dict, relationships_data: dict, subject: str = "") -> Dict[str, int]:
        """
        Write all concepts and relationships from extraction output to Neo4j.
        Returns a summary of how many nodes and edges were written.
        """
        nodes_written = 0
        edges_written = 0
        edges_failed = 0

        logger.info(f"Writing {len(concepts_data['concepts'])} concept nodes...")
        for concept in concepts_data["concepts"]:
            try:
                self.write_concept_node(
                    name=concept["name"],
                    definition=concept.get("definition", ""),
                    subject=subject
                )
                nodes_written += 1
            except Exception as e:
                logger.error(f"Failed to write node '{concept['name']}': {e}")

        logger.info(f"Writing {len(relationships_data['relationships'])} prerequisite edges...")
        for rel in relationships_data["relationships"]:
            try:
                success = self.write_prerequisite_edge(
                    source=rel["source"],
                    target=rel["target"],
                    weight=rel.get("confidence", 1.0)
                )
                if success:
                    edges_written += 1
                else:
                    edges_failed += 1
            except Exception as e:
                logger.error(f"Failed to write edge '{rel['source']}' -> '{rel['target']}': {e}")
                edges_failed += 1

        summary = {
            "nodes_written": nodes_written,
            "edges_written": edges_written,
            "edges_failed": edges_failed
        }
        logger.info(f"Graph population complete: {summary}")
        return summary


# Example Usage
if __name__ == "__main__":
    print("--- Neo4j Service Example ---")

    try:
        service = Neo4jService()

        # 1. Create Concepts
        print("\n1. Creating Concepts...")
        concept1 = service.create_concept(
            concept_id="c001",
            name="Functions",
            description="Mathematical functions",
            mastery_score=0.8
        )
        print(f"Created: {concept1}")

        concept2 = service.create_concept(
            concept_id="c002",
            name="Composite Functions",
            description="Functions applied to functions",
            mastery_score=0.5
        )
        print(f"Created: {concept2}")

        concept3 = service.create_concept(
            concept_id="c003",
            name="Chain Rule",
            description="Derivative of composite functions",
            mastery_score=0.2
        )
        print(f"Created: {concept3}")

        # 2. Create Prerequisites
        print("\n2. Creating Prerequisites...")
        service.create_prerequisite("c001", "c002", weight=0.9)
        service.create_prerequisite("c002", "c003", weight=1.0)
        print("Prerequisites established.")

        # 3. Get Concept
        print("\n3. Retrieving Concept...")
        fetched_concept = service.get_concept("c003")
        print(f"Fetched c003: {fetched_concept}")

        # 4. Get Prerequisites
        print("\n4. Retrieving Prerequisites for Chain Rule (c003)...")
        prereqs = service.get_prerequisites("c003")
        for p in prereqs:
            print(f"- {p['name']} (Weight: {p['relationship_weight']})")

        # 5. Update Mastery Score
        print("\n5. Updating Mastery Score...")
        updated_concept = service.update_mastery_score("c003", 0.45)
        print(f"Updated c003 mastery score: {updated_concept['mastery_score']}")

    except Exception as e:
        print(f"An error occurred during example execution: {e}")
    finally:
        if 'service' in locals():
            service.close()