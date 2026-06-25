"""
Student Modeling Agent — GraphMASAL Track B

Maintains a live, numerical picture of what the student knows.
Every concept has a mastery score between 0 and 1 that updates
after every quiz answer or interaction using a weighted moving average.

Schema note: Person A stores mastery_score directly on the Concept node.
We read/write Concept.mastery_score accordingly.
"""

import logging
from state import StudentState
from db.neo4j_helper import run_query, run_write

logger = logging.getLogger(__name__)

# ── Tunable parameters ──────────────────────────────────────────────────────
WEIGHT_NEW = 0.3          # How much a single interaction affects the score
WEIGHT_OLD = 0.7          # How much history is retained
DEFAULT_MASTERY = 0.2     # Initial score for unseen concepts (assumes some prior exposure)


def run_student_modeling(state: StudentState) -> dict:
    """
    Two-mode operation:

    Mode 1 — UPDATE (interaction_score is set):
        Reads the current mastery for the assessed concept from Neo4j,
        applies weighted moving average, writes the new score back.

    Mode 2 — READ-ONLY (interaction_score is None):
        Just reads all concept mastery scores from Neo4j and populates
        the mastery_scores dict in StudentState.

    Always returns the full mastery_scores snapshot.
    """
    interaction_score = state.get("interaction_score")
    concept_assessed = state.get("current_concept")
    score_delta = None

    # ── Mode 1: Update mastery for a specific concept ────────────────────
    if interaction_score is not None and concept_assessed:
        try:
            # Clamp input score to [0, 1]
            interaction_score = max(0.0, min(1.0, float(interaction_score)))

            # Step 1 — Read current score from Neo4j
            results = run_query(
                "MATCH (c:Concept {name: $name}) RETURN c.mastery_score AS score",
                {"name": concept_assessed},
            )

            if not results:
                logger.warning(
                    f"Concept '{concept_assessed}' not found in graph — skipping update."
                )
            else:
                old_score = results[0]["score"]
                if old_score is None:
                    old_score = DEFAULT_MASTERY

                # Step 2 — Weighted moving average
                new_score = (WEIGHT_NEW * interaction_score) + (WEIGHT_OLD * old_score)
                new_score = max(0.0, min(1.0, new_score))  # clamp to [0, 1]
                score_delta = round(new_score - old_score, 6)

                # Step 3 — Write back to Neo4j
                run_write(
                    """
                    MATCH (c:Concept {name: $name})
                    SET c.mastery_score = $new_score
                    RETURN c.name AS name, c.mastery_score AS score
                    """,
                    {"name": concept_assessed, "new_score": new_score},
                )
                logger.info(
                    f"Updated mastery for '{concept_assessed}': "
                    f"{old_score:.3f} → {new_score:.3f} (delta={score_delta:+.3f})"
                )

        except Exception as e:
            logger.error(f"Student modeling update error: {e}", exc_info=True)
            # Don't block the pipeline — fall through to read all scores

    # ── Always: Read all mastery scores from Neo4j ───────────────────────
    try:
        all_scores = run_query(
            "MATCH (c:Concept) RETURN c.name AS name, c.mastery_score AS score"
        )
        mastery_scores = {
            row["name"]: row["score"] if row["score"] is not None else DEFAULT_MASTERY
            for row in all_scores
        }
    except Exception as e:
        logger.error(f"Failed to read mastery scores from Neo4j: {e}", exc_info=True)
        # Fallback: keep whatever was in state before
        mastery_scores = state.get("mastery_scores") or {}

    result: dict = {
        "mastery_scores": mastery_scores,
        "interaction_score": None,  # clear so it doesn't re-trigger on next pass
    }
    if score_delta is not None:
        result["score_delta"] = score_delta

    return result
