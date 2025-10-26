import type { ResetConfig, ResetResponse, TurnResult } from "./types";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

async function request<T>(
  path: string,
  options: RequestInit
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function resetSimulation(config: ResetConfig): Promise<ResetResponse> {
  const payload = {
    grid_size: config.gridSize,
    num_agents: config.numAgents,
    seed: config.seed ? Number(config.seed) : undefined,
    debug: config.debug,
    backend: config.backend,
    player_agent: config.playerAgent,
  };
  return request<ResetResponse>("/reset", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function stepSimulation(): Promise<TurnResult> {
  return request<TurnResult>("/step", { method: "POST" });
}

export async function sendPlayerAction(action: Record<string, unknown>): Promise<TurnResult> {
  return request<TurnResult>("/player_action", {
    method: "POST",
    body: JSON.stringify({ action }),
  });
}
