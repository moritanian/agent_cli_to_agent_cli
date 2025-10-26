import { useEffect, useMemo, useState } from "react";
import "./App.css";
import {
  resetSimulation,
  stepSimulation,
} from "./api";
import type {
  ConversationEntry,
  DebugEntry,
  ResetConfig,
  Snapshot,
  TurnResult,
} from "./types";

const DEFAULT_CONFIG: ResetConfig = {
  gridSize: 3,
  numAgents: 2,
  seed: "",
  debug: false,
};

function buildGrid(snapshot: Snapshot | null): JSX.Element {
  if (!snapshot) {
    return <div className="grid-placeholder">Reset を押してシミュレーションを開始してください。</div>;
  }

  const { gridSize, agents } = snapshot;
  const rows = [];
  for (let y = 0; y < gridSize; y += 1) {
    const cells = [];
    for (let x = 0; x < gridSize; x += 1) {
      const occupant = agents.find(
        (agent) => agent.position.x === x && agent.position.y === y
      );
      cells.push(
        <div className="grid-cell" key={`${x}-${y}`}>
          <span className="cell-coord">
            {x},{y}
          </span>
          {occupant && <span className="cell-agent">{occupant.name}</span>}
        </div>
      );
    }
    rows.push(
      <div className="grid-row" key={`row-${y}`}>
        {cells}
      </div>
    );
  }

  return (
    <div
      className="grid"
      style={{ gridTemplateRows: `repeat(${gridSize}, 1fr)` }}
    >
      {rows}
    </div>
  );
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
          <strong>{entry.from}</strong> → <strong>{entry.to}</strong>:{" "}
          <span>{entry.message}</span>
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
        <p>LLMエージェントの挙動をリアルタイムで確認できるデバッグUIです。</p>
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
            value={config.numAgents}
            disabled
            title="現在は2人のエージェントのみサポートしています。"
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
          <h2>サンドボックス</h2>
          {buildGrid(snapshot)}
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
