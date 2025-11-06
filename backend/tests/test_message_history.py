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
async def test_message_history_keeps_only_latest():
    """Test that message_history keeps only the most recent message (not accumulating)."""
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

    # Simulate agent0 talking to agent1 (first message)
    action = {"action": "talk", "target": "agent2", "message": "First message"}
    debug_entry = {"notes": ""}

    # Simulate turn 1
    sim.turn = 1
    sim._active_turn_messages = []
    sim._apply_action(agent0, action, debug_entry)

    # Verify agent1 received the first message
    assert len(agent1.message_history) == 1
    assert agent1.message_history[0]["from"] == "agent1"
    assert agent1.message_history[0]["message"] == "First message"
    assert agent1.message_history[0]["turn"] == 1

    # Agent0 should not have any messages
    assert len(agent0.message_history) == 0

    # Simulate agent0 talking to agent1 again (second message)
    action2 = {"action": "talk", "target": "agent2", "message": "Second message"}
    debug_entry2 = {"notes": ""}
    sim.turn = 3
    sim._apply_action(agent0, action2, debug_entry2)

    # Verify agent1 now has ONLY the latest message (old one is replaced)
    assert len(agent1.message_history) == 1
    assert agent1.message_history[0]["from"] == "agent1"
    assert agent1.message_history[0]["message"] == "Second message"
    assert agent1.message_history[0]["turn"] == 3


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
async def test_message_history_keeps_multiple_messages_same_turn():
    """Test that message_history keeps all messages from the same turn."""
    sim = SandboxSimulation(num_agents=3, grid_size=3, backend="mock", seed=42)
    sim.reset()

    # Position all agents adjacent to agent3
    # agent3 is at (1,1), agent1 at (0,1), agent2 at (1,0)
    sim.agents[0].x = 0  # agent1
    sim.agents[0].y = 1
    sim.agents[1].x = 1  # agent2
    sim.agents[1].y = 0
    sim.agents[2].x = 1  # agent3
    sim.agents[2].y = 1

    agent1 = sim.agents[0]
    agent2 = sim.agents[1]
    agent3 = sim.agents[2]

    # Both agent1 and agent2 talk to agent3 in turn 1
    sim.turn = 1
    sim._active_turn_messages = []

    action1 = {"action": "talk", "target": "agent3", "message": "Message from agent1"}
    debug_entry1 = {"notes": ""}
    sim._apply_action(agent1, action1, debug_entry1)

    # agent3 should have agent1's message
    assert len(agent3.message_history) == 1
    assert agent3.message_history[0]["from"] == "agent1"

    action2 = {"action": "talk", "target": "agent3", "message": "Message from agent2"}
    debug_entry2 = {"notes": ""}
    sim._apply_action(agent2, action2, debug_entry2)

    # agent3 should now have BOTH messages (same turn)
    assert len(agent3.message_history) == 2
    assert agent3.message_history[0]["from"] == "agent1"
    assert agent3.message_history[0]["message"] == "Message from agent1"
    assert agent3.message_history[1]["from"] == "agent2"
    assert agent3.message_history[1]["message"] == "Message from agent2"

    # Both should be from turn 1
    assert agent3.message_history[0]["turn"] == 1
    assert agent3.message_history[1]["turn"] == 1

    # Now in a NEW turn, agent1 talks to agent3 again
    sim.turn = 2
    action3 = {"action": "talk", "target": "agent3", "message": "New turn message"}
    debug_entry3 = {"notes": ""}
    sim._apply_action(agent1, action3, debug_entry3)

    # agent3's message_history should be reset to only the new message
    assert len(agent3.message_history) == 1
    assert agent3.message_history[0]["from"] == "agent1"
    assert agent3.message_history[0]["message"] == "New turn message"
    assert agent3.message_history[0]["turn"] == 2


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
