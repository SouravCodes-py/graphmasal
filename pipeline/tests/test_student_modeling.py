"""
Unit tests for the Student Modeling Agent.

All tests use mock Neo4j data — no live database required.
We patch db.neo4j_helper.run_query and run_write to return controlled results.
"""

from unittest.mock import patch
import pytest

from agents.student_modeling import run_student_modeling, WEIGHT_NEW, WEIGHT_OLD, DEFAULT_MASTERY


def _make_state(**overrides):
    """Create a minimal StudentState dict for testing."""
    base = {
        "student_id": "test_student",
        "session_id": "test_session",
        "student_message": "test",
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


class TestStudentModelingAgent:
    """Tests for run_student_modeling()."""

    @patch("agents.student_modeling.run_query")
    @patch("agents.student_modeling.run_write")
    def test_weighted_update(self, mock_write, mock_query):
        """
        Score 0.9 on concept with current mastery 0.34
        → new score ≈ 0.508 (0.3 * 0.9 + 0.7 * 0.34)
        """
        # Mock read current score
        mock_query.side_effect = [
            [{"score": 0.34}],  # Read specific concept
            [{"name": "Chain Rule", "score": 0.508}] # Read all concepts at the end
        ]

        state = _make_state(interaction_score=0.9, current_concept="Chain Rule")
        result = run_student_modeling(state)

        # Check write was called with correct new score
        call_args = mock_write.call_args
        assert call_args[0][1]["new_score"] == pytest.approx(0.508)
        
        # Check delta returned
        assert result["score_delta"] == pytest.approx(0.508 - 0.34)
        
        # Check interaction_score is cleared
        assert result["interaction_score"] is None

    @patch("agents.student_modeling.run_query")
    @patch("agents.student_modeling.run_write")
    def test_score_clamping(self, mock_write, mock_query):
        """
        Score 0.0 → score decreases but stays >= 0.
        Input interaction_score out of bounds is clamped.
        """
        mock_query.side_effect = [
            [{"score": 0.1}],  # Read specific concept
            [{"name": "Chain Rule", "score": 0.07}] # Read all concepts
        ]

        # Provide a negative interaction score to test clamping
        state = _make_state(interaction_score=-0.5, current_concept="Chain Rule")
        result = run_student_modeling(state)

        # Interaction score should be clamped to 0.0
        # New score: 0.3 * 0.0 + 0.7 * 0.1 = 0.07
        call_args = mock_write.call_args
        assert call_args[0][1]["new_score"] == pytest.approx(0.07)

    @patch("agents.student_modeling.run_query")
    @patch("agents.student_modeling.run_write")
    def test_new_concept_initialization(self, mock_write, mock_query):
        """
        First interaction on a new concept → initialised at 0.2 before update.
        """
        mock_query.side_effect = [
            [{"score": None}],  # Read specific concept returns None
            [{"name": "New Concept", "score": 0.26}] # Read all concepts
        ]

        state = _make_state(interaction_score=0.4, current_concept="New Concept")
        run_student_modeling(state)

        # Old score assumed 0.2
        # New score: 0.3 * 0.4 + 0.7 * 0.2 = 0.12 + 0.14 = 0.26
        call_args = mock_write.call_args
        assert call_args[0][1]["new_score"] == pytest.approx(0.26)

    @patch("agents.student_modeling.run_query")
    @patch("agents.student_modeling.run_write")
    def test_neo4j_error_fallback(self, mock_write, mock_query):
        """
        If Neo4j write fails, we don't crash, we just fall through to reading all scores.
        """
        mock_query.side_effect = [
            [{"score": 0.5}],  # Read specific concept
            [{"name": "Chain Rule", "score": 0.5}] # Read all concepts
        ]
        mock_write.side_effect = Exception("Neo4j write failed")

        state = _make_state(interaction_score=0.9, current_concept="Chain Rule")
        # Should not raise exception
        result = run_student_modeling(state)
        
        # Read all concepts still returns 0.5
        assert result["mastery_scores"]["Chain Rule"] == 0.5

    @patch("agents.student_modeling.run_query")
    def test_read_only_mode(self, mock_query):
        """
        When interaction_score is None, we just read all scores and return them.
        """
        mock_query.return_value = [
            {"name": "Chain Rule", "score": 0.5},
            {"name": "Derivatives", "score": None} # Should default to 0.2
        ]

        state = _make_state(interaction_score=None, current_concept="Chain Rule")
        result = run_student_modeling(state)

        assert "score_delta" not in result
        assert result["mastery_scores"]["Chain Rule"] == 0.5
        assert result["mastery_scores"]["Derivatives"] == DEFAULT_MASTERY
