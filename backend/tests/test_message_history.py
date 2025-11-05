"""Test message_history feature."""

import pytest

from sandbox_simulation import SandboxSimulation


@pytest.mark.asyncio
async def test_message_history_field_exists():
    """Test that message_history field is initialized."""
    sim = SandboxSimulation(num_agents=2, grid_size=3, backend="mock", seed=42)
    sim.reset()

    # Verify all agents have message_history field
    for agent in sim.agents:
        assert hasattr(agent, "message_history")
        assert isinstance(agent.message_history, list)


@pytest.mark.asyncio
async def test_message_history_accumulates_on_talk():
    """Test that message_history accumulates when agent receives messages."""
    sim = SandboxSimulation(num_agents=2, grid_size=3, backend="mock", seed=42)
    sim.reset()

    # Position agents adjacent to each other
    sim.agents[0].x = 0
    sim.agents[0].y = 0
    sim.agents[1].x = 1
    sim.agents[1].y = 0

    # Manually trigger a talk action (deterministic test)
    agent0 = sim.agents[0]
    agent1 = sim.agents[1]

    # Simulate agent0 talking to agent1
    action = {"action": "talk", "target": "agent2", "message": "Hello agent2!"}
    debug_entry = {"notes": ""}

    # Simulate turn 1
    sim.turn = 1
    sim._active_turn_messages = []
    sim._apply_action(agent0, action, debug_entry)

    # Verify agent1 received the message in message_history
    assert len(agent1.message_history) == 1
    assert agent1.message_history[0]["from"] == "agent1"
    assert agent1.message_history[0]["message"] == "Hello agent2!"
    assert agent1.message_history[0]["turn"] == 1

    # Agent0 should not have any messages
    assert len(agent0.message_history) == 0

    # Simulate agent1 talking back to agent0
    action2 = {"action": "talk", "target": "agent1", "message": "Hi agent1!"}
    debug_entry2 = {"notes": ""}
    sim.turn = 2
    sim._apply_action(agent1, action2, debug_entry2)

    # Verify agent0 now has one message
    assert len(agent0.message_history) == 1
    assert agent0.message_history[0]["from"] == "agent2"
    assert agent0.message_history[0]["message"] == "Hi agent1!"
    assert agent0.message_history[0]["turn"] == 2

    # Agent1 should still have only one message (not their own)
    assert len(agent1.message_history) == 1


@pytest.mark.asyncio
async def test_message_appears_in_next_step_observation():
    """Test that when an agent is talked to, the message appears in their next step observation."""
    sim = SandboxSimulation(num_agents=2, grid_size=3, backend="mock", seed=42)
    sim.reset()

    # Position agents adjacent
    sim.agents[0].x = 0
    sim.agents[0].y = 0
    sim.agents[1].x = 1
    sim.agents[1].y = 0

    # Manually trigger agent0 talking to agent1 in turn 1
    sim.turn = 1
    sim._active_turn_messages = []
    sim._active_turn_debug = []

    agent0 = sim.agents[0]
    agent1 = sim.agents[1]

    action = {"action": "talk", "target": "agent2", "message": "Hello from agent1!"}
    debug_entry = {"notes": ""}
    sim._apply_action(agent0, action, debug_entry)

    # Verify agent1's inbox has the message
    assert agent1.inbox is not None
    assert agent1.inbox["from"] == "agent1"
    assert agent1.inbox["message"] == "Hello from agent1!"

    # Now simulate the next step where agent1 should see this in their observation
    import json
    from sandbox_simulation import _legal_actions

    sim.turn = 2
    sim._active_turn_messages = []
    sim._active_turn_debug = []

    legal_actions = _legal_actions(agent1, sim.agents, sim.grid_size)

    # Build observation (this is what happens in _continue_turn_from)
    observation = {
        "you": agent1.name,
        "positions": {state.name: state.position for state in sim.agents},
        "grid_size": sim.grid_size,
        "turn": sim.turn,
        "legal_actions": legal_actions,
        "traits": sim.agent_profiles,
    }
    if agent1.inbox:
        observation["message"] = agent1.inbox
    if agent1.message_history:
        observation["message_history"] = agent1.message_history

    # Verify the current message is in observation
    assert "message" in observation
    assert observation["message"]["from"] == "agent1"
    assert observation["message"]["message"] == "Hello from agent1!"

    # Verify the message_history also has it
    assert "message_history" in observation
    assert len(observation["message_history"]) == 1
    assert observation["message_history"][0]["from"] == "agent1"
    assert observation["message_history"][0]["message"] == "Hello from agent1!"
    assert observation["message_history"][0]["turn"] == 1

    # After clearing inbox (which happens after building observation)
    agent1.inbox = None

    # In turn 3, agent1 should no longer have "message" but still have "message_history"
    sim.turn = 3
    observation2 = {
        "you": agent1.name,
        "positions": {state.name: state.position for state in sim.agents},
        "grid_size": sim.grid_size,
        "turn": sim.turn,
        "legal_actions": legal_actions,
        "traits": sim.agent_profiles,
    }
    if agent1.inbox:
        observation2["message"] = agent1.inbox
    if agent1.message_history:
        observation2["message_history"] = agent1.message_history

    # No current message
    assert "message" not in observation2

    # But message_history persists
    assert "message_history" in observation2
    assert len(observation2["message_history"]) == 1


@pytest.mark.asyncio
async def test_message_history_included_in_observation():
    """Test that message_history is included in observation when non-empty."""
    sim = SandboxSimulation(num_agents=2, grid_size=3, backend="mock", seed=42)
    sim.reset()

    # Position agents adjacent
    sim.agents[0].x = 0
    sim.agents[0].y = 0
    sim.agents[1].x = 1
    sim.agents[1].y = 0

    # Manually add a message to agent1's history
    sim.agents[1].message_history.append({
        "from": "agent1",
        "message": "Test message",
        "turn": 1
    })

    # Run a step and check the observation includes message_history
    sim.turn = 1
    sim._active_turn_messages = []
    sim._active_turn_debug = []

    # Build observation for agent1
    from sandbox_simulation import _legal_actions
    import json

    agent = sim.agents[1]
    legal_actions = _legal_actions(agent, sim.agents, sim.grid_size)

    observation = {
        "you": agent.name,
        "positions": {state.name: state.position for state in sim.agents},
        "grid_size": sim.grid_size,
        "turn": sim.turn,
        "legal_actions": legal_actions,
        "traits": sim.agent_profiles,
    }

    if agent.inbox:
        observation["message"] = agent.inbox
    if agent.message_history:
        observation["message_history"] = agent.message_history

    # Verify message_history is in observation
    assert "message_history" in observation
    assert len(observation["message_history"]) == 1
    assert observation["message_history"][0]["from"] == "agent1"
