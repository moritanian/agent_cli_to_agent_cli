import { useEffect, useMemo, useState } from "react";
import type { CSSProperties } from "react";
import "./App.css";
import { resetSimulation, sendPlayerAction, stepSimulation } from "./api";
import type {
  AgentTrait,
  ConversationEntry,
  DebugEntry,
  PlayerRequest,
  ResetConfig,
  Snapshot,
  TurnResult,
} from "./types";

type ThemeMap = Record<string, AgentTrait>;
type PlayerActionPayload =
  | { action: "move"; direction: string }
  | { action: "talk"; target: string; message: string }
  | { action: "wait" };

const DEFAULT_CONFIG: ResetConfig = {
  gridSize: 3,
  numAgents: 2,
  seed: "",
  debug: false,
  backend: "gemini",
  playerAgent: false,
};

const FALLBACK_THEMES: AgentTrait[] = [
  {
    title: "Sentinel",
    icon: "🛡️",
    color: "#8ecae6",
    glow: "rgba(142, 202, 230, 0.6)",
    persona: "Speak naturally as the Sentinel and collaborate with the party.",
  },
  {
    title: "Rogue",
    icon: "🗡️",
    color: "#f9a03f",
    glow: "rgba(249, 160, 63, 0.6)",
    persona: "Speak naturally as the Rogue and collaborate with the party.",
  },
  {
    title: "Arcanist",
    icon: "🪄",
    color: "#bb6bd9",
    glow: "rgba(187, 107, 217, 0.6)",
    persona: "Speak naturally as the Arcanist and collaborate with the party.",
  },
  {
    title: "Ranger",
    icon: "🏹",
    color: "#6ee7b7",
    glow: "rgba(110, 231, 183, 0.55)",
    persona: "Speak naturally as the Ranger and collaborate with the party.",
  },
  {
    title: "Smith",
    icon: "⚒️",
    color: "#f97316",
    glow: "rgba(249, 115, 22, 0.5)",
    persona: "Speak naturally as the Smith and collaborate with the party.",
  },
  {
    title: "Bard",
    icon: "🎻",
    color: "#f472b6",
    glow: "rgba(244, 114, 182, 0.55)",
    persona: "Speak naturally as the Bard and collaborate with the party.",
  },
];

function buildThemeMap(snapshot: Snapshot | null): ThemeMap {
  if (!snapshot) {
    return {};
  }
  const traits = snapshot.traits ?? {};
  const map: ThemeMap = {};
  snapshot.agents.forEach((agent, index) => {
    const profile = traits[agent.name];
    if (profile) {
      map[agent.name] = profile;
    } else {
      map[agent.name] = FALLBACK_THEMES[index % FALLBACK_THEMES.length];
    }
  });
  return map;
}

function buildSpeechMap(snapshot: Snapshot | null): Record<string, string> {
  if (!snapshot) return {};
  const talkMap: Record<string, string> = {};
  const currentTurn = snapshot.turn;
  [...snapshot.messages]
    .reverse()
    .forEach((entry) => {
      if (entry.turn !== currentTurn) {
        return;
      }
      if (entry.from && !(entry.from in talkMap)) {
        talkMap[entry.from] = entry.message;
      }
    });
  return talkMap;
}

function buildGrid(
  snapshot: Snapshot | null,
  themes: ThemeMap,
  speeches: Record<string, string>,
): JSX.Element {
  if (!snapshot) {
    return (
      <div className="grid-placeholder">
        Press Reset to begin the simulation.
      </div>
    );
  }

  const { gridSize, agents } = snapshot;
  const rows = [];
  for (let y = 0; y < gridSize; y += 1) {
    const cells = [];
    for (let x = 0; x < gridSize; x += 1) {
      const occupant = agents.find(
        (agent) => agent.position.x === x && agent.position.y === y,
      );
      const theme = occupant ? themes[occupant.name] : undefined;
      const speech = occupant ? speeches[occupant.name] : undefined;
      const preview =
        speech && speech.length > 80 ? `${speech.slice(0, 77)}…` : speech;
      cells.push(
        <div
          className={`grid-cell ${theme ? "grid-cell--occupied" : ""}`}
          key={`${x}-${y}`}
          style={
            theme
              ? ({
                  "--tile-accent": theme.color,
                  "--tile-glow": theme.glow,
                } as CSSProperties)
              : {}
          }
        >
          <span className="cell-coord">{`${x},${y}`}</span>
          <div className="tile-bg" />
          {occupant ? (
            <div className="cell-agent">
              <span className="agent-icon">{theme?.icon ?? "⭐"}</span>
              <span className="agent-label">{theme?.title ?? occupant.name}</span>
              {preview && <span className="cell-dialogue">“{preview}”</span>}
            </div>
          ) : (
            <span className="cell-mote">✦</span>
          )}
        </div>,
      );
    }
    rows.push(
      <div className="grid-row" key={`row-${y}`}>
        {cells}
      </div>,
    );
  }

  return (
    <div className="grid" style={{ gridTemplateRows: `repeat(${gridSize}, 1fr)` }}>
      {rows}
    </div>
  );
}

function renderLegend(snapshot: Snapshot | null, themes: ThemeMap): JSX.Element {
  if (!snapshot) {
    return (
      <div className="legend">
        <span className="legend-empty">Empty tile</span>
      </div>
    );
  }

  const seen = new Set<string>();
  const badges = snapshot.agents
    .map((agent) => {
      if (seen.has(agent.name)) return null;
      seen.add(agent.name);
      const theme = themes[agent.name];
      return (
        <div
          key={agent.name}
          className="legend-item"
          style={
            theme
              ? ({
                  "--tile-accent": theme.color,
                  "--tile-glow": theme.glow,
                } as CSSProperties)
              : {}
          }
        >
          <span className="legend-icon">{theme?.icon ?? "★"}</span>
          <span>{theme?.title ?? agent.name}</span>
        </div>
      );
    })
    .filter(Boolean);

  return <div className="legend">{badges}</div>;
}

function ConversationLog({
  messages,
  themes,
}: {
  messages: ConversationEntry[];
  themes: ThemeMap;
}) {
  if (!messages.length) {
    return <p className="placeholder">No conversation yet.</p>;
  }
  return (
    <ul className="conversation-list">
      {messages.map((entry, idx) => (
        <li key={`${entry.turn}-${idx}`}>
          <span className="conversation-turn">Turn {entry.turn}</span>{" "}
          <strong>{themes[entry.from]?.title ?? entry.from}</strong> →{" "}
          <strong>{themes[entry.to]?.title ?? entry.to}</strong>: <span>{entry.message}</span>
        </li>
      ))}
    </ul>
  );
}

function DebugPanel({ history, themes }: { history: TurnResult[]; themes: ThemeMap }) {
  if (!history.length) {
    return <p className="placeholder">Advance a turn to view prompts and responses.</p>;
  }
  return (
    <div className="debug-list">
      {history.map((entry) => (
        <details key={entry.turn} open>
          <summary>Turn {entry.turn}</summary>
          {entry.debug.map((debug: DebugEntry, idx: number) => (
            <div className="debug-entry" key={`${entry.turn}-${debug.agent}-${idx}`}>
              <h4>{themes[debug.agent]?.title ?? debug.agent}</h4>
              <div className="debug-field">
                <span className="label">Legal actions:</span>
                <code>{JSON.stringify(debug.legal_actions)}</code>
              </div>
              <div className="debug-field">
                <span className="label">Prompt:</span>
                <pre>{debug.prompt}</pre>
              </div>
              <div className="debug-field">
                <span className="label">Response:</span>
                <pre>{debug.response}</pre>
              </div>
              <div className="debug-field">
                <span className="label">Parsed action:</span>
                <code>{JSON.stringify(debug.action)}</code>
              </div>
              {debug.notes && (
                <div className="debug-field">
                  <span className="label">Notes:</span>
                  <span>{debug.notes}</span>
                </div>
              )}
            </div>
          ))}
        </details>
      ))}
    </div>
  );
}

export default function App(): JSX.Element {
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [history, setHistory] = useState<TurnResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [playerRequest, setPlayerRequest] = useState<PlayerRequest | null>(null);
  const [messageDrafts, setMessageDrafts] = useState<Record<string, string>>({});

  const canStep = useMemo(() => Boolean(snapshot) && playerRequest === null, [snapshot, playerRequest]);
  const themeMap = useMemo(() => buildThemeMap(snapshot), [snapshot]);
  const speechMap = useMemo(() => buildSpeechMap(snapshot), [snapshot]);

  useEffect(() => {
    if (!playerRequest) {
      setMessageDrafts({});
      return;
    }
    setMessageDrafts((prev) => {
      const next: Record<string, string> = {};
      playerRequest.legal_actions.forEach((action) => {
        if (action.action === "talk" && action.target) {
          const title = themeMap[action.target]?.title ?? action.target;
          const fallback = `Hey ${title}, let's keep moving!`;
          next[action.target] = prev[action.target] ?? fallback;
        }
      });
      return next;
    });
  }, [playerRequest, themeMap]);

  useEffect(() => {
    handleReset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function handleReset() {
    try {
      setLoading(true);
      setError(null);
      setPlayerRequest(null);
      setMessageDrafts({});
      const response = await resetSimulation(config);
      setSnapshot(response.snapshot);
      setHistory([]);
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  async function handleStep() {
    if (!canStep) return;
    try {
      setLoading(true);
      setError(null);
      const result = await stepSimulation();
      setSnapshot(result.snapshot);
      if (result.requiresPlayer) {
        setPlayerRequest(result.player ?? null);
        return;
      }
      setPlayerRequest(null);
      setMessageDrafts({});
      setHistory((prev) => [...prev, result]);
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  async function handlePlayerAction(action: PlayerActionPayload) {
    try {
      setLoading(true);
      setError(null);
      const result = await sendPlayerAction(action);
      setSnapshot(result.snapshot);
      if (result.requiresPlayer) {
        setPlayerRequest(result.player ?? null);
        return;
      }
      setPlayerRequest(null);
      setMessageDrafts({});
      setHistory((prev) => [...prev, result]);
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Sandbox Agent Playground</h1>
        <p>Watch LLM-driven adventurers coordinate on a glowing battle grid.</p>
      </header>

      <section className="controls">
        <div className="field">
          <label htmlFor="grid">Grid Size</label>
          <input
            id="grid"
            type="number"
            min={2}
            max={8}
            value={config.gridSize}
            onChange={(event) =>
              setConfig((prev) => ({
                ...prev,
                gridSize: Number(event.target.value),
              }))
            }
          />
        </div>
        <div className="field">
          <label htmlFor="agents">Number of Agents{config.playerAgent ? " (including you)" : ""}</label>
          <input
            id="agents"
            type="number"
            min={2}
            max={6}
            value={config.numAgents}
            onChange={(event) =>
              setConfig((prev) => ({
                ...prev,
                numAgents: Number(event.target.value),
              }))
            }
          />
        </div>
        <div className="field">
          <label htmlFor="seed">Seed (optional)</label>
          <input
            id="seed"
            type="text"
            value={config.seed}
            onChange={(event) =>
              setConfig((prev) => ({
                ...prev,
                seed: event.target.value,
              }))
            }
          />
        </div>
        <div className="field checkbox">
          <label htmlFor="debug">
            <input
              id="debug"
              type="checkbox"
              checked={config.debug}
              onChange={(event) =>
                setConfig((prev) => ({
                  ...prev,
                  debug: event.target.checked,
                }))
              }
            />
            Enable CLI debug logs
          </label>
        </div>
        <div className="field">
          <label htmlFor="backend">LLM Backend</label>
          <select
            id="backend"
            value={config.backend}
            onChange={(event) =>
              setConfig((prev) => ({
                ...prev,
                backend: event.target.value,
              }))
            }
          >
            <option value="gemini">Gemini CLI</option>
            <option value="codex">Codex CLI</option>
            <option value="mock">Mock (offline)</option>
          </select>
        </div>
        <div className="field checkbox">
          <label htmlFor="playerAgent">
            <input
              id="playerAgent"
              type="checkbox"
              checked={config.playerAgent}
              onChange={(event) =>
                setConfig((prev) => ({
                  ...prev,
                  playerAgent: event.target.checked,
                }))
              }
            />
            Add player-controlled agent
          </label>
        </div>
        <div className="actions">
          <button onClick={handleReset} disabled={loading}>
            Reset
          </button>
          <button onClick={handleStep} disabled={!canStep || loading}>
            Step
          </button>
        </div>
        {error && <p className="error">Error: {error}</p>}
      </section>

      <main className="layout">
        <section className="board">
        <h2>
          Adventure Board
          {snapshot?.backend && (
            <span className="board-backend">{snapshot.backend.toUpperCase()} mode</span>
          )}
        </h2>
          <div className="board-diorama">
            <div className="board-overlay" />
            {buildGrid(snapshot, themeMap, speechMap)}
          </div>
          {renderLegend(snapshot, themeMap)}
          {playerRequest && (
            <div className="player-panel">
              <h3>Player Turn</h3>
              <p>
                Choose an action for {themeMap[playerRequest.agent]?.title ?? playerRequest.agent}.
              </p>
              <div className="player-actions">
                {(() => {
                  let waitRendered = false;
                  return playerRequest.legal_actions.map((action, idx) => {
                  if (action.action === "move" && action.direction) {
                    const dir = action.direction.charAt(0).toUpperCase() + action.direction.slice(1);
                    const label = `Move ${dir}`;
                    return (
                      <button
                        key={`move-${action.direction}`}
                        className="player-action-button"
                        onClick={() => handlePlayerAction({ action: "move", direction: action.direction! })}
                        disabled={loading}
                      >
                        {label}
                      </button>
                    );
                  }
                  if (action.action === "talk" && action.target) {
                    const title = themeMap[action.target]?.title ?? action.target;
                    const value = messageDrafts[action.target] ?? "";
                    return (
                      <div key={`talk-${action.target}`} className="player-talk-option">
                        <label>
                          Message to {title}
                          <input
                            className="player-message-input"
                            value={value}
                            onChange={(event) =>
                              setMessageDrafts((prev) => ({
                                ...prev,
                                [action.target!]: event.target.value,
                              }))
                            }
                          />
                        </label>
                        <button
                          className="player-action-button"
                          onClick={() =>
                            handlePlayerAction({
                              action: "talk",
                              target: action.target!,
                              message:
                                (messageDrafts[action.target!] ?? "").trim() ||
                                `Hey ${title}, let's keep moving!`,
                            })
                          }
                          disabled={loading}
                        >
                          Talk to {title}
                        </button>
                      </div>
                    );
                  }
                  if (action.action === "wait") {
                    if (waitRendered) {
                      return null;
                    }
                    waitRendered = true;
                    return (
                      <button
                        key="wait"
                        className="player-action-button"
                        onClick={() => handlePlayerAction({ action: "wait" })}
                        disabled={loading}
                      >
                        Wait this turn
                      </button>
                    );
                  }
                  return null;
                });
                })()}
              </div>
            </div>
          )}
        </section>
        <section className="panel">
          <div className="panel-section">
            <h2>Conversation Log</h2>
            <ConversationLog messages={snapshot?.messages ?? []} themes={themeMap} />
          </div>
          <div className="panel-section">
            <h2>Debug Panel</h2>
            <DebugPanel history={history} themes={themeMap} />
          </div>
        </section>
      </main>
    </div>
  );
}
