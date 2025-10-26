from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project root (containing cli_clients, etc.) is importable when the package
# is executed from an installed location.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sandbox_simulation import SandboxSimulation

app = FastAPI(title="Sandbox Agent Simulator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

simulation: Optional[SandboxSimulation] = None


class ResetRequest(BaseModel):
    grid_size: int = 3
    num_agents: int = 2
    seed: Optional[int] = None
    debug: bool = False
    backend: str = "gemini"


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
async def reset(request: ResetRequest) -> dict[str, object]:
    global simulation
    simulation = SandboxSimulation(
        num_agents=request.num_agents,
        grid_size=request.grid_size,
        debug=request.debug,
        seed=request.seed,
        backend=request.backend,
    )
    snapshot = simulation.reset()
    return {"status": "ok", "snapshot": snapshot}


@app.post("/step")
async def step() -> dict[str, object]:
    if simulation is None:
        raise HTTPException(status_code=400, detail="Simulation not initialised. Call /reset first.")
    result = await simulation.step()
    return result


@app.get("/state")
async def state() -> dict[str, object]:
    if simulation is None:
        raise HTTPException(status_code=400, detail="Simulation not initialised. Call /reset first.")
    return {"snapshot": simulation.snapshot(), "history": simulation.history()}
