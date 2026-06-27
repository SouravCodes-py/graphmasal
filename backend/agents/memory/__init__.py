"""
GraphMASAL — Memory Agent
~~~~~~~~~~~~~~~~~~~~~~~~~
Provides long-term semantic memory for student tutoring sessions.

Public API:
    write(student_id, session_events) -> str | None
    fetch(student_id, query)          -> list[str]
"""

from .memory_agent import write, fetch

__all__ = ["write", "fetch"]
