from state import StudentState

def fetch_memory(state: StudentState) -> dict:
    # STUB — replace with pgvector cosine search on Day 4-5
    return {
        "memory_context": ["Student struggled with chain rule last session."]
    }

def write_memory(state: StudentState) -> dict:
    # STUB — replace with LLM summarise + embed + pgvector write on Day 4-5
    return {}
