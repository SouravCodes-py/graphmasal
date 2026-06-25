from typing import TypedDict, Optional
from langgraph.graph import MessagesState


class StudentState(TypedDict):
    # --- Identity ---
    student_id: str
    session_id: str

    # --- Input ---
    student_message: str
    tone_preference: str                  # 'explain_simply' | 'just_facts' | 'encouraging'

    # --- Knowledge graph (set by KEA) ---
    knowledge_graph_ready: bool

    # --- Diagnostic output ---
    failed_concept: Optional[str]
    root_cause: Optional[str]
    root_mastery: Optional[float]
    diagnosis_depth: Optional[int]

    # --- Planning output ---
    target_concept: Optional[str]
    learning_path: Optional[list[str]]    # ordered list of concept names
    path_rationale: Optional[str]
    estimated_concepts: Optional[int]

    # --- Mastery snapshot ---
    mastery_scores: Optional[dict]        # { concept_name: float }
    mastery_threshold: float              # default 0.5

    # --- Memory ---
    memory_context: Optional[list[str]]   # top-3 past session summaries, fetched on demand

    # --- Quiz ---
    current_concept: Optional[str]        # concept currently being taught/assessed
    quiz_triggered: bool                  # whether quiz should follow this tutor response
    quiz_questions: Optional[list[dict]]  # [{ type, question_text, difficulty }]

    # --- Tutor output ---
    tutor_response: Optional[str]

    # --- Routing flags ---
    new_concept_flagged: bool             # student introduced concept outside learning_path
    rediagnose: bool                      # orchestrator should re-run Diagnostic
