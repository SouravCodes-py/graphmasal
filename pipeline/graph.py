from langgraph.graph import StateGraph, END
from state import StudentState
from agents.diagnostic import run_diagnostic
from agents.student_modeling import run_student_modeling
from agents.planning import run_planning
from agents.memory import fetch_memory, write_memory
from agents.tutor import run_tutor
from agents.quiz_generation import run_quiz_generation


def should_rediagnose(state: StudentState) -> str:
    """
    Routing function after Tutor runs.
    If student introduced a new concept outside the learning path,
    re-run Diagnostic. Otherwise end the turn.
    """
    if state.get("rediagnose"):
        return "diagnostic"
    return "end"


def should_trigger_quiz(state: StudentState) -> str:
    """
    Routing function after Tutor runs.
    If Tutor flagged that the current concept needs assessment, run Quiz Gen.
    Otherwise route to rediagnose check.
    """
    if state.get("quiz_triggered"):
        return "quiz_generation"
    return "check_rediagnose"


def build_graph() -> StateGraph:
    graph = StateGraph(StudentState)

    # --- Register nodes ---
    graph.add_node("student_modeling", run_student_modeling)
    graph.add_node("diagnostic", run_diagnostic)
    graph.add_node("planning", run_planning)
    graph.add_node("memory_fetch", fetch_memory)
    graph.add_node("tutor", run_tutor)
    graph.add_node("quiz_generation", run_quiz_generation)
    graph.add_node("memory_write", write_memory)

    # --- Entry point ---
    graph.set_entry_point("student_modeling")

    # --- Linear edges ---
    graph.add_edge("student_modeling", "diagnostic")
    graph.add_edge("diagnostic", "planning")
    graph.add_edge("planning", "memory_fetch")
    graph.add_edge("memory_fetch", "tutor")

    # --- Conditional edges after Tutor ---
    graph.add_conditional_edges(
        "tutor",
        should_trigger_quiz,
        {
            "quiz_generation": "quiz_generation",
            "check_rediagnose": "memory_write",
        }
    )

    graph.add_edge("quiz_generation", "student_modeling")  # score feedback loop

    # --- Conditional: re-diagnose or end ---
    graph.add_conditional_edges(
        "memory_write",
        should_rediagnose,
        {
            "diagnostic": "diagnostic",
            "end": END,
        }
    )

    return graph.compile()


compiled_graph = build_graph()
