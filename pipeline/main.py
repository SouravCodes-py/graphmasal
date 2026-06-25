from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from graph import compiled_graph
from state import StudentState
import uuid

app = FastAPI(title="GraphMASAL API")


# --- Request / Response models ---

class ChatRequest(BaseModel):
    student_id: str
    message: str
    tone: str = "encouraging"
    session_id: str = None

class ChatResponse(BaseModel):
    tutor_response: str
    study_path: list[str] | None = None
    quiz: list[dict] | None = None
    memory_context: list[str] | None = None

class StateResponse(BaseModel):
    student_id: str
    mastery: list[dict]     # [{ concept, score }]
    at_risk: list[str]

class QuizAnswerRequest(BaseModel):
    session_id: str
    concept: str
    question_id: str
    selected_index: int

class QuizAnswerResponse(BaseModel):
    correct: bool
    new_mastery: float
    follow_up: str | None = None


# --- Endpoints ---

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main entry point. Runs the full LangGraph pipeline and returns
    the tutor response plus any structured data (study path, quiz).
    
    Day 1: runs stub agents, returns hardcoded data.
    Day 4-5: streams tutor response via SSE.
    """
    initial_state: StudentState = {
        "student_id": req.student_id,
        "session_id": req.session_id or str(uuid.uuid4()),
        "student_message": req.message,
        "tone_preference": req.tone,
        "knowledge_graph_ready": True,   # assume True until KEA is integrated
        "mastery_threshold": 0.5,
        "quiz_triggered": False,
        "new_concept_flagged": False,
        "rediagnose": False,
        # all other fields start as None
        "failed_concept": None,
        "root_cause": None,
        "root_mastery": None,
        "diagnosis_depth": None,
        "target_concept": None,
        "learning_path": None,
        "path_rationale": None,
        "estimated_concepts": None,
        "mastery_scores": None,
        "memory_context": None,
        "current_concept": None,
        "quiz_questions": None,
        "tutor_response": None,
    }

    result = await compiled_graph.ainvoke(initial_state)

    return ChatResponse(
        tutor_response=result["tutor_response"],
        study_path=result.get("learning_path"),
        quiz=result.get("quiz_questions") if result.get("quiz_triggered") else None,
        memory_context=result.get("memory_context"),
    )


@app.get("/state", response_model=StateResponse)
async def get_state(student_id: str):
    """
    Returns current mastery scores and at-risk topics for the sidebar.
    Called on page load and after every quiz submission.
    
    Day 1: returns hardcoded data.
    Day 2-3: reads from Neo4j.
    """
    return StateResponse(
        student_id=student_id,
        mastery=[
            {"concept": "Chain Rule", "score": 0.34},
            {"concept": "Composite Functions", "score": 0.58},
            {"concept": "Derivatives", "score": 0.81},
            {"concept": "u-substitution", "score": 0.22},
        ],
        at_risk=["Integration by parts", "Taylor series"],
    )


@app.post("/quiz/answer", response_model=QuizAnswerResponse)
async def submit_quiz_answer(req: QuizAnswerRequest):
    """
    Receives a quiz answer. Routes score to Student Modeling Agent.
    Re-fetches updated mastery from Neo4j and returns it.
    
    Day 1: returns hardcoded correct/incorrect.
    Day 6-7: real scoring and mastery update.
    """
    return QuizAnswerResponse(
        correct=True,
        new_mastery=0.51,
        follow_up=None,
    )


@app.post("/upload")
async def upload_material(file: UploadFile = File(...)):
    """
    Accepts a PDF or text file. Triggers Knowledge Extraction Agent (Person A).
    
    Day 1: accepts file, returns stub response.
    Day 2-3: real extraction pipeline (Person A owns this).
    """
    return {
        "status": "received",
        "filename": file.filename,
        "message": "Extraction pipeline stub — KEA not yet connected."
    }


@app.get("/graph/concepts")
async def get_concepts():
    """
    Debug endpoint. Returns all concept nodes in Neo4j.
    Day 1: returns hardcoded list.
    """
    return {
        "concepts": ["Chain Rule", "Composite Functions", "Derivatives", "u-substitution", "Integration"]
    }
