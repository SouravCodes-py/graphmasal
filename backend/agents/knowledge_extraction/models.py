from pydantic import BaseModel, Field
from typing import List

class Concept(BaseModel):
    id: str = Field(description="Unique identifier for the concept, e.g., 'c001', 'c002'")
    name: str = Field(description="Name of the educational concept")
    description: str = Field(description="Concise explanation of the concept")

class ConceptList(BaseModel):
    concepts: List[Concept]

class Relationship(BaseModel):
    source: str = Field(description="ID of the prerequisite concept")
    target: str = Field(description="ID of the target concept that depends on the source")
    relationship_type: str = Field(default="PREREQUISITE_OF", description="The type of relationship")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")

class RelationshipList(BaseModel):
    relationships: List[Relationship]

class KnowledgeGraph(BaseModel):
    concepts: List[Concept]
    relationships: List[Relationship]
