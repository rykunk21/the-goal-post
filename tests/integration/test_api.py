"""Integration tests for API endpoints.

These tests verify the full pipeline works end-to-end.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch


@pytest.fixture
def client():
    """Create test client for API."""
    from src.api.main import app
    return TestClient(app)


class TestHealthEndpoint:
    """Test M5.1: Endpoint Availability."""

    def test_health_check_returns_200(self, client):
        """Health endpoint should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_check_returns_json(self, client):
        """Health endpoint should return JSON."""
        response = client.get("/health")
        assert response.headers["content-type"] == "application/json"


class TestPredictionEndpoint:
    """Test prediction pipeline."""

    @patch('src.api.routes.prediction.load_team_data')
    @patch('src.api.routes.prediction.run_mcmc')
    def test_predict_returns_win_probabilities(self, mock_mcmc, mock_data, client):
        """Prediction endpoint should return win probabilities."""
        # Mock the model responses
        mock_data.return_value = {
            "home_team": {"mu": [0.1] * 16, "sigma": [0.5] * 16},
            "away_team": {"mu": [0.1] * 16, "sigma": [0.5] * 16}
        }
        mock_mcmc.return_value = {
            "home_win_prob": 0.65,
            "away_win_prob": 0.35,
            "home_expected_score": 75.2,
            "away_expected_score": 68.4,
            "spread": "Home -6.8"
        }

        response = client.post("/predict", json={
            "home_team_id": "150",  # Duke
            "away_team_id": "127",  # Michigan State
            "is_neutral_site": False
        })

        assert response.status_code == 200
        data = response.json()
        assert "home_win_prob" in data
        assert "away_win_prob" in data

    def test_predict_requires_team_ids(self, client):
        """Prediction should fail without required team IDs."""
        response = client.post("/predict", json={})
        assert response.status_code == 422  # Validation error


class TestTrainingEndpoints:
    """Test model training endpoints."""

    @patch('src.api.routes.training.train_vae')
    def test_train_vae_returns_job_id(self, mock_train, client):
        """VAE training endpoint should return job ID."""
        mock_train.return_value = "job_123"

        response = client.post("/train/vae", json={
            "epochs": 100,
            "learning_rate": 0.001
        })

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data

    @patch('src.api.routes.training.train_nn')
    def test_train_nn_returns_job_id(self, mock_train, client):
        """NN training endpoint should return job ID."""
        mock_train.return_value = "job_456"

        response = client.post("/train/nn", json={
            "epochs": 50,
            "batch_size": 32
        })

        assert response.status_code == 200


class TestModelStatus:
    """Test model status endpoint."""

    def test_model_status_returns_current_state(self, client):
        """Model status should return current model state."""
        response = client.get("/model/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "vae_trained" in data
        assert "nn_trained" in data
        assert "training_data_games" in data


class TestSchemaValidation:
    """Test M5.2: Response Schema Validation."""

    def test_predict_response_schema(self, client):
        """Prediction response should match OpenAPI schema."""
        # This test would validate against OpenAPI spec
        # after implementation
        pass

    def test_error_responses_are_consistent(self, client):
        """Error responses should have consistent schema."""
        response = client.post("/predict", json={"home_team_id": ""})
        assert response.status_code == 422
        
        error = response.json()
        assert "detail" in error
