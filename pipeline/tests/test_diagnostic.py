"""
Unit tests for the Diagnostic Agent.

All tests use mock Neo4j data — no live database required.
We patch db.neo4j_helper.run_query to return controlled results.
"""

from unittest.mock import patch
import pytest

from agents.diagnostic import run_diagnostic


def _make_state(**overrides):
    """Create a minimal StudentState dict for testing."""
    base = {
        "student_id": "test_student",
        "session_id": "test_session",
        "student_message": "I don't understand Integration",
        "tone_preference": "encouraging",
        "knowledge_graph_ready": True,
        "mastery_threshold": 0.5,
        "quiz_triggered": False,
        "new_concept_flagged": False,
        "rediagnose": False,
        "failed_concept": None,
        "root_cause": None,
        "root_mastery": None,
        "diagnosis_depth": None,
        "target_concept": None,
        "learning_path": None,
        "path_rationale": None,
        "estimated_concepts": None,
        "mastery_scores": None,
        "interaction_score": None,
        "score_delta": None,
        "memory_context": None,
        "current_concept": None,
        "quiz_questions": None,
        "tutor_response": None,
    }
    base.update(overrides)
    return base


class TestDiagnosticAgent:
    """Tests for run_diagnostic()."""

    def test_no_failed_concept_skips_diagnosis(self):
        """When failed_concept is not set, diagnosis should be skipped."""
        state = _make_state(failed_concept=None)
        result = run_diagnostic(state)
        assert result["root_cause"] is None
        assert result["root_mastery"] is None
        assert result["diagnosis_depth"] is None

    @patch("agents.diagnostic.run_query")
    def test_deep_weak_prerequisite_found(self, mock_query):
        """
        Student fails Integration, Chain Rule mastery is 0.34 →
        root cause should be Chain Rule (deepest weak prerequisite).
        """
        mock_query.return_value = [
            {"root_cause": "Chain Rule", "root_mastery": 0.34, "depth": 2}
        ]

        state = _make_state(failed_concept="Integration", mastery_threshold=0.5)
        result = run_diagnostic(state)

        assert result["root_cause"] == "Chain Rule"
        assert result["root_mastery"] == 0.34
        assert result["diagnosis_depth"] == 2

    @patch("agents.diagnostic.run_query")
    def test_all_prerequisites_above_threshold(self, mock_query):
        """
        All prerequisites above 0.5 → root cause should be the
        failed concept itself with depth 0.
        """
        # First call: traversal returns no results (all prereqs are strong)
        # Second call: get the concept's own mastery score
        mock_query.side_effect = [
            [],  # no weak ancestors
            [{"score": 0.42}],  # own mastery
        ]

        state = _make_state(failed_concept="Integration", mastery_threshold=0.5)
        result = run_diagnostic(state)

        assert result["root_cause"] == "Integration"
        assert result["root_mastery"] == 0.42
        assert result["diagnosis_depth"] == 0

    @patch("agents.diagnostic.run_query")
    def test_concept_not_in_graph(self, mock_query):
        """
        Concept not in the graph at all → graceful fallback,
        returns the failed concept with depth 0 and no mastery.
        """
        mock_query.side_effect = [
            [],  # no traversal results
            [],  # concept not found
        ]

        state = _make_state(failed_concept="Quantum Mechanics", mastery_threshold=0.5)
        result = run_diagnostic(state)

        assert result["root_cause"] == "Quantum Mechanics"
        assert result["root_mastery"] is None
        assert result["diagnosis_depth"] == 0

    @patch("agents.diagnostic.run_query")
    def test_deepest_weak_node_returned_not_shallowest(self, mock_query):
        """
        3-level deep weak prerequisite → should return the deepest one,
        not the shallowest.
        """
        # The Cypher ORDER BY depth DESC, score ASC, LIMIT 1 already does this,
        # but we verify the agent returns the first (deepest) result.
        mock_query.return_value = [
            {"root_cause": "Algebra", "root_mastery": 0.20, "depth": 3}
        ]

        state = _make_state(failed_concept="Integration", mastery_threshold=0.5)
        result = run_diagnostic(state)

        assert result["root_cause"] == "Algebra"
        assert result["diagnosis_depth"] == 3

    @patch("agents.diagnostic.run_query")
    def test_neo4j_error_graceful_fallback(self, mock_query):
        """
        Neo4j throws an exception → should not crash,
        returns failed concept as fallback.
        """
        mock_query.side_effect = Exception("Neo4j connection refused")

        state = _make_state(failed_concept="Integration")
        result = run_diagnostic(state)

        assert result["root_cause"] == "Integration"
        assert result["root_mastery"] is None
        assert result["diagnosis_depth"] == 0

    @patch("agents.diagnostic.run_query")
    def test_custom_mastery_threshold(self, mock_query):
        """
        Stricter threshold (0.7) should surface more weak concepts.
        """
        mock_query.return_value = [
            {"root_cause": "Functions", "root_mastery": 0.65, "depth": 1}
        ]

        state = _make_state(
            failed_concept="Derivatives", mastery_threshold=0.7
        )
        result = run_diagnostic(state)

        assert result["root_cause"] == "Functions"
        assert result["root_mastery"] == 0.65

        # Verify the threshold was passed to the Cypher query
        call_args = mock_query.call_args
        assert call_args[0][1]["mastery_threshold"] == 0.7
