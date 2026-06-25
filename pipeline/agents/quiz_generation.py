from state import StudentState

def run_quiz_generation(state: StudentState) -> dict:
    # STUB — replace with real LLM question generation on Day 6-7
    return {
        "quiz_questions": [
            {"type": "recall", "question_text": "What is the chain rule?", "difficulty": "easy"},
            {"type": "application", "question_text": "If f(x) = sin(x²), what is f′(x)?", "difficulty": "medium"},
            {"type": "conceptual", "question_text": "Why does the chain rule require multiplying the outer and inner derivatives?", "difficulty": "hard"},
        ]
    }
