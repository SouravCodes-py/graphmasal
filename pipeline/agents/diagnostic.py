"""
Diagnostic Agent — GraphMASAL Track B

Traces backward through the knowledge graph from a failed concept
to find the root cause: the deepest, weakest prerequisite.

Schema note: Person A stores mastery_score directly on the Concept node,
not on a HAS_MASTERY edge. We query Concept.mastery_score accordingly.
"""

import logging
from state import StudentState
from db.neo4j_helper import run_query

logger = logging.getLogger(__name__)


def run_diagnostic(state: StudentState) -> dict:
    """
    Walk backwards from failed_concept through PREREQUISITE_OF edges.
    Find the deepest ancestor with mastery_score below mastery_threshold.

    If failed_concept is not set in state, returns empty diagnosis
    (e.g. on the first turn when no failure has occurred yet).
    """
    failed_concept = state.get("failed_concept")
    if not failed_concept:
        logger.info("No failed_concept in state — skipping diagnosis.")
        return {
            "root_cause": None,
            "root_mastery": None,
            "diagnosis_depth": None,
        }

    mastery_threshold = state.get("mastery_threshold", 0.5)

    try:
        # ── Step 1-3: Backward traversal + filter + select root cause ──
        #
        # Cypher walks backward up to 10 hops through PREREQUISITE_OF.
        # Filters to ancestors with mastery_score < threshold.
        # Orders by depth DESC (deepest first), then score ASC (weakest first).
        # LIMIT 1 gives us the single best root cause.
        results = run_query(
            """
            MATCH path = (root:Concept)-[:PREREQUISITE_OF*1..10]->(failed:Concept {name: $failed_concept})
            WHERE root.mastery_score < $mastery_threshold
            RETURN root.name AS root_cause,
                   root.mastery_score AS root_mastery,
                   length(path) AS depth
            ORDER BY depth DESC, root.mastery_score ASC
            LIMIT 1
            """,
            {
                "failed_concept": failed_concept,
                "mastery_threshold": mastery_threshold,
            },
        )

        if results:
            row = results[0]
            logger.info(
                f"Root cause identified: {row['root_cause']} "
                f"(mastery={row['root_mastery']}, depth={row['depth']})"
            )
            return {
                "root_cause": row["root_cause"],
                "root_mastery": row["root_mastery"],
                "diagnosis_depth": row["depth"],
            }

        # ── No weak prerequisites found ──
        # The failed concept itself is the direct issue.
        concept_results = run_query(
            "MATCH (c:Concept {name: $name}) RETURN c.mastery_score AS score",
            {"name": failed_concept},
        )

        if concept_results:
            own_score = concept_results[0]["score"]
            logger.info(
                f"No weak prerequisites for '{failed_concept}' — "
                f"returning it as root cause (mastery={own_score})."
            )
            return {
                "root_cause": failed_concept,
                "root_mastery": own_score,
                "diagnosis_depth": 0,
            }

        # ── Concept not in graph at all ──
        logger.warning(f"Concept '{failed_concept}' not found in Neo4j graph.")
        return {
            "root_cause": failed_concept,
            "root_mastery": None,
            "diagnosis_depth": 0,
        }

    except Exception as e:
        # Graceful fallback — never crash the pipeline
        logger.error(f"Diagnostic agent error: {e}", exc_info=True)
        return {
            "root_cause": failed_concept,
            "root_mastery": None,
            "diagnosis_depth": 0,
        }
