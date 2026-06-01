import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from neo4j import GraphDatabase, exceptions

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jService:
    """Service class for interacting with the Neo4j database."""

    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize the Neo4j driver. Uses environment variables if arguments are not provided.
        """
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
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed.")

    def create_concept(self, concept_id: str, name: str, description: str, mastery_score: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        Create a new Concept node.
        """
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
        """
        Create a PREREQUISITE_OF relationship between two Concept nodes.
        (prerequisite_id) -[:PREREQUISITE_OF {weight: weight}]-> (target_id)
        """
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
        """
        Retrieve a Concept node by its ID.
        """
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
        """
        Get all concepts that are prerequisites of the given concept.
        (p)-[:PREREQUISITE_OF]->(c:Concept {id: concept_id})
        """
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
        """
        Update the mastery_score of a Concept node.
        """
        # Ensure score is between 0.0 and 1.0
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


# Example Usage
if __name__ == "__main__":
    # Ensure you have your .env file set up with NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD
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
        service.create_prerequisite("c001", "c002", weight=0.9) # Functions -> Composite Functions
        service.create_prerequisite("c002", "c003", weight=1.0) # Composite Functions -> Chain Rule
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
