import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "backend" / "src"))

from sandbox_simulation import SandboxSimulation


@pytest.mark.asyncio
async def test_mock_backend_returns_valid_actions():
    sim = SandboxSimulation(num_agents=2, grid_size=3, backend="mock", seed=42)
    snapshot = sim.reset()
    assert snapshot["backend"] == "mock"
    for _ in range(5):
        result = await sim.step()
        for entry in result["debug"]:
            action = entry["action"]
            assert action["action"] in {"move", "talk", "wait"}
            if action["action"] == "move":
                assert action.get("direction") in {"up", "down", "left", "right"}
            if action["action"] == "talk":
                assert "target" in action and "message" in action
                assert isinstance(action["message"], str) and action["message"].strip()


@pytest.mark.asyncio
async def test_mock_backend_updates_conversation_log():
    sim = SandboxSimulation(num_agents=2, grid_size=3, backend="mock", seed=1)
    sim.reset()
    sim.agents[0].x = 0
    sim.agents[0].y = 0
    sim.agents[1].x = 1
    sim.agents[1].y = 0

    result = await sim.step()
    messages = result["turnMessages"]
    if messages:
        entry = messages[0]
        assert {"from", "to", "message", "turn"}.issubset(entry.keys())
        assert entry["message"].strip()


@pytest.mark.asyncio
async def test_mock_backend_records_history():
    sim = SandboxSimulation(num_agents=3, grid_size=4, backend="mock", seed=7)
    sim.reset()
    for _ in range(5):
        await sim.step()
    history = sim.history()
    assert len(history) == 5
    assert history[-1]["snapshot"]["backend"] == "mock"


@pytest.mark.asyncio
async def test_player_agent_requires_input_and_applies_action():
    sim = SandboxSimulation(num_agents=2, grid_size=3, backend="mock", player_agent=True, seed=3)
    snapshot = sim.reset()
    assert snapshot["playerAgent"] is True

    pending = await sim.step()
    assert pending["requiresPlayer"] is True
    player = pending["player"]
    assert player["agent"].endswith(str(sim.num_agents))

    # choose a wait action (always available)
    wait_action = next(
        (entry for entry in player["legal_actions"] if entry["action"] == "wait"),
        {"action": "wait"},
    )
    result = await sim.apply_player_action(wait_action)
    assert not result.get("requiresPlayer", False)
    assert sim.pending_player is None
    assert sim.history()[-1]["turn"] == sim.turn
