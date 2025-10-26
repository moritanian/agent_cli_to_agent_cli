export interface Position {
  x: number;
  y: number;
}

export interface AgentSnapshot {
  name: string;
  position: Position;
}

export interface AgentTrait {
  title: string;
  icon: string;
  color: string;
  glow: string;
  persona: string;
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
  traits: Record<string, AgentTrait>;
  messages: ConversationEntry[];
  backend?: string;
  playerAgent?: boolean;
}

export interface LegalAction {
  action: "move" | "talk" | "wait";
  direction?: "up" | "down" | "left" | "right";
  target?: string;
  target_title?: string;
}

export interface PlayerRequest {
  agent: string;
  legal_actions: LegalAction[];
  traits?: AgentTrait;
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
  requiresPlayer?: boolean;
  player?: PlayerRequest;
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
  backend: string;
  playerAgent: boolean;
}
