# GraphMASAL Schema

## Concept Node

Represents a learning concept in the knowledge graph.

Properties:

| Field         | Type   | Description                 |
| ------------- | ------ | --------------------------- |
| id            | string | Unique concept identifier   |
| name          | string | Concept name                |
| description   | string | Short explanation           |
| mastery_score | float  | Student mastery (0.0 - 1.0) |
| embedding     | vector | Semantic embedding          |

Example:

```json
{
  "id": "c001",
  "name": "Chain Rule",
  "description": "Derivative of composite functions",
  "mastery_score": 0.34
}
```

---

## Relationship Schema

### PREREQUISITE_OF

Represents dependency between concepts.

Example:

Functions
→ Composite Functions
→ Chain Rule

Properties:

| Field  | Type  | Description         |
| ------ | ----- | ------------------- |
| weight | float | Dependency strength |

Example:

```json
{
  "type": "PREREQUISITE_OF",
  "weight": 0.9
}
```

---

## Student State

Shared between LangGraph agents.

```python
class StudentState(TypedDict):
    student_id: str
    message: str
    current_concept: str | None
    root_cause: str | None
    mastery_score: float
    study_plan: list[str]
    memory_context: list[str]
```

---

## Diagnostic Agent Output

```json
{
  "root_cause": "Chain Rule",
  "mastery_score": 0.34
}
```

---

## Planning Agent Output

```json
{
  "study_plan": [
    "Functions",
    "Composite Functions",
    "Chain Rule"
  ]
}
```

---

## Quiz Agent Output

```json
{
  "concept": "Chain Rule",
  "questions": [
    {
      "type": "recall",
      "question": "What is the Chain Rule?"
    }
  ]
}
```

---

## Databases

### Neo4j

Stores:

* Concepts
* Prerequisite relationships
* Mastery scores
* Embeddings

### PostgreSQL / Supabase

Stores:

* Quiz history
* Session history
* Episodic memory
* Generated quizzes

---

## Naming Convention

Use snake_case.

Examples:

* mastery_score
* current_concept
* study_plan
* root_cause
