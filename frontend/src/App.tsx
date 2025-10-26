import { useEffect, useMemo, useState } from "react";
import type { CSSProperties } from "react";
import "./App.css";
import { resetSimulation, stepSimulation } from "./api";
import type {
  AgentTrait,
  ConversationEntry,
  DebugEntry,
  ResetConfig,
  Snapshot,
  TurnResult,
} from "./types";

type ThemeMap = Record<string, AgentTrait>;

const DEFAULT_CONFIG: ResetConfig = {
  gridSize: 3,
  numAgents: 2,
  seed: "",
  debug: false,
};

const FALLBACK_THEMES: AgentTrait[] = [
  {
    title: "Sentinel",
    icon: "🛡️",
    color: "#8ecae6",
    glow: "rgba(142, 202, 230, 0.6)",
    persona: "Sentinelとして自然に対話してください",
  },
  {
    title: "Rogue",
    icon: "🗡️",
    color: "#f9a03f",
    glow: "rgba(249, 160, 63, 0.6)",
    persona: "Rogueとして自然に対話してください",
  },
  {
    title: "Arcanist",
    icon: "🪄",
    color: "#bb6bd9",
    glow: "rgba(187, 107, 217, 0.6)",
    persona: "Arcanistとして自然に対話してください",
  },
  {
    title: "Ranger",
    icon: "🏹",
    color: "#6ee7b7",
    glow: "rgba(110, 231, 183, 0.55)",
    persona: "Rangerとして自然に対話してください",
  },
  {
    title: "Smith",
    icon: "⚒️",
    color: "#f97316",
    glow: "rgba(249, 115, 22, 0.5)",
    persona: "Smithとして自然に対話してください",
  },
  {
    title: "Bard",
    icon: "🎻",
    color: "#f472b6",
    glow: "rgba(244, 114, 182, 0.55)",
    persona: "Bardとして自然に対話してください",
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

function buildGrid(snapshot: Snapshot | null, themes: ThemeMap): JSX.Element {
  if (!snapshot) {
    return (
      <div className="grid-placeholder">
        Reset を押してシミュレーションを開始してください。
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
        <span className="legend-empty">空きマス</span>
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

function ConversationLog({ messages }: { messages: ConversationEntry[] }) {
  if (!messages.length) {
    return <p className="placeholder">まだ会話ログはありません。</p>;
  }
  return (
    <ul className="conversation-list">
      {messages.map((entry, idx) => (
        <li key={`${entry.turn}-${idx}`}>
          <span className="conversation-turn">Turn {entry.turn}</span>{" "}
          <strong>{entry.from}</strong> → <strong>{entry.to}</strong>: <span>{entry.message}</span>
        </li>
      ))}
    </ul>
  );
}

function DebugPanel({ history }: { history: TurnResult[] }) {
  if (!history.length) {
    return <p className="placeholder">ターンを進めるとプロンプトと応答が表示されます。</p>;
  }
  return (
    <div className="debug-list">
      {history.map((entry) => (
        <details key={entry.turn} open>
          <summary>Turn {entry.turn}</summary>
          {entry.debug.map((debug: DebugEntry, idx: number) => (
            <div className="debug-entry" key={`${entry.turn}-${debug.agent}-${idx}`}>
              <h4>{debug.agent}</h4>
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

  const canStep = useMemo(() => Boolean(snapshot), [snapshot]);
  const themeMap = useMemo(() => buildThemeMap(snapshot), [snapshot]);

  useEffect(() => {
    handleReset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function handleReset() {
    try {
      setLoading(true);
      setError(null);
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
        <p>LLMエージェントの挙動をファンタジーな冒険盤で見届けましょう。</p>
      </header>

      <section className="controls">
        <div className="field">
          <label htmlFor="grid">Grid サイズ</label>
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
          <label htmlFor="agents">Agent 数</label>
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
          <label htmlFor="seed">Seed (任意)</label>
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
            Gemini CLI debug
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
          <h2>冒険盤</h2>
          <div className="board-diorama">
            <div className="board-overlay" />
            {buildGrid(snapshot, themeMap)}
          </div>
          {renderLegend(snapshot, themeMap)}
        </section>
        <section className="panel">
          <div className="panel-section">
            <h2>会話ログ</h2>
            <ConversationLog messages={snapshot?.messages ?? []} />
          </div>
          <div className="panel-section">
            <h2>デバッグパネル</h2>
            <DebugPanel history={history} />
          </div>
        </section>
      </main>
    </div>
  );
}
