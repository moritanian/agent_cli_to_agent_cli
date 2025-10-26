export interface Position {
  x: number;
  y: number;
}

export interface AgentSnapshot {
  name: string;
  position: Position;
}

export interface ConversationEntry {
  from: string;
  to: string;
  message: string;
  turn: number;
}

export interface Snapshot {
  turn: number;
  gridSize: number;
  agents: AgentSnapshot[];
  messages: ConversationEntry[];
}

export interface LegalAction {
  action: "move" | "talk" | "wait";
  direction?: "up" | "down" | "left" | "right";
  target?: string;
}

export interface DebugEntry {
  agent: string;
  prompt: string;
  response: string;
  legal_actions: LegalAction[];
  action: Record<string, unknown>;
  notes?: string;
}

export interface TurnResult {
  turn: number;
  snapshot: Snapshot;
  turnMessages: ConversationEntry[];
  debug: DebugEntry[];
}

export interface ResetResponse {
  status: string;
  snapshot: Snapshot;
}

export interface ResetConfig {
  gridSize: number;
  numAgents: number;
  seed?: string;
  debug: boolean;
}
