"""Tests for Federated Learning Server and Client (Milestone 16.2)."""

import asyncio
import numpy as np
import pytest
from hub.dp_trainer import DPModel
from hub.federated_server import FederatedAggregationServer, InsufficientParticipantsError
from hub.federated_client import FederatedHubClient


def test_receive_gradients_from_three_hubs_aggregate_succeeds():
    global_model = DPModel(weights=np.zeros(3), bias=0.0)
    server = FederatedAggregationServer(global_model, min_participants=3)

    g1 = [np.array([0.1, 0.2, 0.3]), np.array([0.05])]
    g2 = [np.array([0.2, 0.3, 0.4]), np.array([0.08])]
    g3 = [np.array([0.15, 0.25, 0.35]), np.array([0.06])]

    server.receive_gradients("hub_1", g1, n_samples=100, epsilon=1.2)
    server.receive_gradients("hub_2", g2, n_samples=150, epsilon=1.1)
    server.receive_gradients("hub_3", g3, n_samples=120, epsilon=1.3)

    agg = server.aggregate()
    assert len(agg) == 2
    assert agg[0].shape == (3,)
    assert agg[1].shape == (1,)


def test_aggregate_raises_insufficient_participants_error_with_less_than_min():
    global_model = DPModel(weights=np.zeros(2), bias=0.0)
    server = FederatedAggregationServer(global_model, min_participants=3)

    server.receive_gradients("hub_1", [np.array([0.1, 0.2])], n_samples=50, epsilon=1.0)
    server.receive_gradients("hub_2", [np.array([0.2, 0.3])], n_samples=50, epsilon=1.0)

    with pytest.raises(InsufficientParticipantsError):
        server.aggregate()


def test_fedavg_weighted_average_correct():
    global_model = DPModel(weights=np.zeros(2), bias=0.0)
    server = FederatedAggregationServer(global_model, min_participants=2)

    # Hub A: 100 samples, Hub B: 50 samples (Total: 150)
    # Weights: A has 100/150 = 2/3, B has 50/150 = 1/3
    g_a = [np.array([3.0, 6.0])]
    g_b = [np.array([0.0, 0.0])]

    server.receive_gradients("hub_a", g_a, n_samples=100, epsilon=1.0)
    server.receive_gradients("hub_b", g_b, n_samples=50, epsilon=1.0)

    agg = server.aggregate()
    expected = (2.0 / 3.0) * np.array([3.0, 6.0]) + (1.0 / 3.0) * np.array([0.0, 0.0])
    assert np.allclose(agg[0], expected)


def test_broadcast_global_weights_returns_serializable_dict():
    global_model = DPModel(weights=np.array([1.0, 2.0]), bias=0.5)
    server = FederatedAggregationServer(global_model, min_participants=2)

    server.receive_gradients("hub_1", [np.array([0.2, 0.4]), np.array([0.1])], n_samples=100, epsilon=1.0)
    server.receive_gradients("hub_2", [np.array([0.4, 0.2]), np.array([0.2])], n_samples=100, epsilon=1.0)

    weights_dict = server.broadcast_global_weights()
    assert isinstance(weights_dict, dict)
    assert "round" in weights_dict
    assert "weights" in weights_dict
    assert "bias" in weights_dict
    assert isinstance(weights_dict["weights"], list)
    assert isinstance(weights_dict["bias"], float)


def test_get_round_status_increments_round_counter_after_aggregation():
    global_model = DPModel(weights=np.zeros(2), bias=0.0)
    server = FederatedAggregationServer(global_model, min_participants=2)

    status_1 = server.get_round_status()
    assert status_1["round"] == 1
    assert status_1["participants"] == 0

    server.receive_gradients("h1", [np.array([0.1, 0.1])], 50, 1.0)
    server.receive_gradients("h2", [np.array([0.1, 0.1])], 50, 1.0)
    server.broadcast_global_weights()

    status_2 = server.get_round_status()
    assert status_2["round"] == 2
    assert status_2["participants"] == 0


@pytest.mark.anyio
async def test_federated_hub_client_participate_in_round_posts_and_updates_registry():
    global_model = DPModel(weights=np.zeros(3), bias=0.0)
    server = FederatedAggregationServer(global_model, min_participants=2)

    # Mock ModelRegistry
    class MockRegistry:
        def __init__(self):
            self.saved = []
        def register_model(self, model, metrics, hyperparameters):
            self.saved.append((model, metrics, hyperparameters))

    registry = MockRegistry()
    client1 = FederatedHubClient(server_url="http://hub.internal", hub_id="h1", registry=registry, server=server)
    client2 = FederatedHubClient(server_url="http://hub.internal", hub_id="h2", registry=registry, server=server)

    rng = np.random.RandomState(42)
    X1, y1 = rng.normal(size=(50, 3)), rng.randint(0, 2, size=50)
    X2, y2 = rng.normal(size=(50, 3)), rng.randint(0, 2, size=50)

    res1 = await client1.participate_in_round(X1, y1)
    assert res1["status"] == "SUBMITTED"

    res2 = await client2.participate_in_round(X2, y2)
    assert res2["status"] == "AGGREGATED"
    assert res2["round"] == 1
    assert len(registry.saved) == 1


def test_global_model_weights_different_from_any_single_hub_local_weights():
    global_model = DPModel(weights=np.array([0.5, 0.5]), bias=0.0)
    server = FederatedAggregationServer(global_model, min_participants=2, learning_rate=1.0)

    # Hub 1 gradient: [1.0, 0.0] -> local update direction
    # Hub 2 gradient: [0.0, 1.0] -> local update direction
    server.receive_gradients("h1", [np.array([1.0, 0.0]), np.array([0.0])], 100, 1.0)
    server.receive_gradients("h2", [np.array([0.0, 1.0]), np.array([0.0])], 100, 1.0)

    server.broadcast_global_weights()

    # Aggregated gradient: [0.5, 0.5], new weights: [0.5 - 0.5, 0.5 - 0.5] = [0.0, 0.0]
    assert not np.allclose(global_model.weights, [0.5 - 1.0, 0.5])
    assert not np.allclose(global_model.weights, [0.5, 0.5 - 1.0])
    assert np.allclose(global_model.weights, [0.0, 0.0])
