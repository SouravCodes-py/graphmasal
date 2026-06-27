"""
GraphMASAL — Memory Agent  (memory_agent.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Semantic long-term memory for student tutoring sessions.

Two public functions:
    • write(student_id, session_events) — summarise via Groq (Llama 3.3 70B),
      embed via Google text-embedding-004, persist to Supabase (pgvector).
    • fetch(student_id, query) — embed query, cosine similarity search via
      Supabase RPC, return top-3 summaries.

Environment variables required:
    GROQ_API_KEY, GOOGLE_API_KEY, SUPABASE_URL, SUPABASE_KEY (or SUPABASE_ANON_KEY)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from groq import Groq
from supabase import create_client, Client as SupabaseClient

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("graphmasal.memory_agent")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)-8s %(name)s — %(message)s")
    )
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIMS = 3072
GROQ_MODEL = "llama-3.3-70b-versatile"
SUPABASE_TABLE = "semantic_memories"
RPC_FUNCTION = "match_semantic_memories"
TOP_K = 3

# ---------------------------------------------------------------------------
# Lazy-initialised clients
# ---------------------------------------------------------------------------
_groq_client: Optional[Groq] = None
_supabase_client: Optional[SupabaseClient] = None


def _get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is not set")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def _get_supabase() -> SupabaseClient:
    global _supabase_client
    if _supabase_client is None:
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
        if not url or not key:
            raise EnvironmentError(
                "SUPABASE_URL and/or SUPABASE_KEY (or SUPABASE_ANON_KEY) is not set"
            )
        url = url.replace("/rest/v1/", "").replace("/rest/v1", "").rstrip("/")
        _supabase_client = create_client(url, key)
    return _supabase_client


def _get_google_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise EnvironmentError("GOOGLE_API_KEY is not set")
    return key


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _summarise_session(session_events: List[Dict[str, Any]]) -> str:
    """Call Groq (Llama 3.3 70B) to produce a 2-3 sentence session summary."""

    events_text = json.dumps(session_events, indent=2)

    system_prompt = (
        "You are a concise educational-analytics assistant. "
        "Given a list of concept-performance events from a student's tutoring "
        "session, write a 2-3 sentence summary that captures:\n"
        "  1. Which concepts the student struggled with and why.\n"
        "  2. Which concepts clicked or showed improvement.\n"
        "Keep it factual and useful for a future tutor reviewing this student's history."
    )

    user_prompt = (
        f"Here are the session events:\n\n{events_text}\n\n"
        "Summarise this session in 2-3 sentences."
    )

    response = _get_groq().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=256,
    )

    summary: str = response.choices[0].message.content.strip()
    logger.debug("Session summary generated (%d chars).", len(summary))
    return summary


def _embed(text: str) -> List[float]:
    """Return a 768-dim embedding via Google text-embedding-004."""
    genai.configure(api_key=_get_google_key())
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text
    )
    embedding = result["embedding"]
    logger.debug("Embedding generated (dim=%d).", len(embedding))
    return embedding


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write(
    student_id: str,
    session_events: List[Dict[str, Any]],
) -> Optional[str]:
    """Summarise a tutoring session and store it as a semantic memory.

    Parameters
    ----------
    student_id : str
        Unique identifier for the student.
    session_events : list[dict]
        Each dict should have ``concept``, ``score``, and ``result`` keys.

    Returns
    -------
    str or None
        The generated summary text on success, ``None`` on failure.
    """
    try:
        logger.info("write() called for student_id=%s with %d events.",
                    student_id, len(session_events))

        # 1. Summarise via Groq
        summary = _summarise_session(session_events)
        logger.info("Summary: %s", summary)

        # 2. Embed via Google
        embedding = _embed(summary)

        # 3. Persist to Supabase
        row = {
            "student_id": student_id,
            "summary": summary,
            "embedding": embedding,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        result = _get_supabase().table(SUPABASE_TABLE).insert(row).execute()
        logger.info(
            "Inserted memory row for student_id=%s (rows inserted=%d).",
            student_id,
            len(result.data) if result.data else 0,
        )
        return summary

    except Exception:
        logger.exception("write() failed for student_id=%s", student_id)
        return None


def fetch(
    student_id: str,
    query: str,
) -> List[str]:
    """Retrieve top-3 most relevant past session summaries for a student.

    Parameters
    ----------
    student_id : str
        Unique identifier for the student.
    query : str
        Natural-language query (e.g. ``"struggles with differentiation"``).

    Returns
    -------
    list[str]
        Up to 3 summary strings ordered by cosine similarity.
        Returns empty list on error or no results.
    """
    try:
        logger.info("fetch() called for student_id=%s, query='%s'",
                    student_id, query[:80])

        # 1. Embed the query
        query_embedding = _embed(query)

        # 2. Supabase RPC — pgvector cosine similarity
        result = (
            _get_supabase()
            .rpc(
                RPC_FUNCTION,
                {
                    "p_student_id": student_id,
                    "p_embedding": query_embedding,
                    "p_match_count": TOP_K,
                },
            )
            .execute()
        )

        rows = result.data or []
        summaries: List[str] = [row["summary"] for row in rows]

        logger.info("fetch() returning %d summaries for student_id=%s.",
                    len(summaries), student_id)
        return summaries

    except Exception:
        logger.exception("fetch() failed for student_id=%s", student_id)
        return []


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    TEST_STUDENT = "test-student-001"

    dummy_events = [
        {"concept": "Chain Rule", "score": 0.4, "result": "struggled"},
        {"concept": "Product Rule", "score": 0.85, "result": "mastered"},
        {"concept": "Quotient Rule", "score": 0.55, "result": "needs review"},
    ]

    print("\n=== write() ===")
    summary = write(TEST_STUDENT, dummy_events)
    if summary:
        print(f"  ✓ Stored summary: {summary}")
    else:
        print("  ✗ write() returned None — check logs above.")

    print("\n=== fetch() ===")
    results = fetch(TEST_STUDENT, "What did the student struggle with in calculus?")
    if results:
        for i, s in enumerate(results, 1):
            print(f"  [{i}] {s}")
    else:
        print("  (no results)")

    print("\nDone.")