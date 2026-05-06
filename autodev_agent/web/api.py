"""
Web UI for AutoDevAgent using FastAPI.
Provides REST API and dashboard for task management, model configuration, and agent control.
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import asyncio
import json
from pathlib import Path
from datetime import datetime
import os

from ..models.config import get_config, reload_config, Config
from ..orchestrator.core import Orchestrator
from ..models.domain import Task, TaskStatus, AgentRole, Agent
from ..models.provider import ModelManager, OpenAIProvider, OllamaProvider


app = FastAPI(
    title="AutoDevAgent API",
    description="REST API for AutoDevAgent orchestration system with model and agent management",
    version="0.2.0"
)

# Global instances
orchestrator: Optional[Orchestrator] = None
config: Optional[Config] = None
active_connections: List[WebSocket] = []


class TaskRequest(BaseModel):
    """Request model for creating a task."""
    description: str
    title: Optional[str] = None
    decompose: bool = True
    steps: int = Field(ge=1, le=10, default=5)
    model: Optional[str] = None
    role: str = "developer"
    priority: str = "medium"


class TaskResponse(BaseModel):
    """Response model for task status."""
    id: str
    title: str
    status: str
    result: Optional[str] = None
    error_message: Optional[str] = None
    tokens_used: int = 0
    cost_usd: float = 0.0
    duration: Optional[float] = None
    created_at: str
    role: str = ""


class ModelConfigRequest(BaseModel):
    """Request to configure a model."""
    name: str
    provider: str = "openai"  # openai, ollama
    max_tokens: int = 4096
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class ModelStatus(BaseModel):
    """Model status information."""
    name: str
    provider: str
    available: bool
    is_local: bool
    configured: bool


class AgentConfigRequest(BaseModel):
    """Request to configure an agent."""
    name: str
    role: str
    model_name: str
    system_prompt: Optional[str] = None


class ConnectionTestRequest(BaseModel):
    """Request to test a connection."""
    provider: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize orchestrator and config on startup."""
    global orchestrator, config
    config = get_config()
    orchestrator = Orchestrator(config)
    print("🚀 AutoDevAgent Web UI started")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main dashboard."""
    html_path = Path(__file__).parent / "dashboard.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    else:
        return HTMLResponse(content=get_dashboard_html())


@app.get("/api/status")
async def get_status():
    """Get current system status."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    status = orchestrator.get_status()
    return {
        **status,
        "uptime": datetime.now().isoformat(),
        "version": "0.2.0"
    }


# ==================== TASK MANAGEMENT ====================

@app.post("/api/tasks", response_model=TaskResponse)
async def create_task(request: TaskRequest, background_tasks: BackgroundTasks):
    """Create and execute a new task."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Map role string to AgentRole
        try:
            role = AgentRole(request.role.upper())
        except ValueError:
            role = AgentRole.DEVELOPER
        
        # Create task directly through orchestrator
        task = orchestrator.create_task(
            title=request.title or request.description[:100],
            description=request.description,
            role=role,
            priority=request.priority
        )
        
        # Run task in background
        background_tasks.add_task(execute_task_background, task.id, request)
        
        return TaskResponse(
            id=task.id,
            title=task.title,
            status="pending",
            role=request.role,
            created_at=task.created_at.isoformat() if hasattr(task.created_at, 'isoformat') else str(task.created_at)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tasks")
async def list_tasks():
    """List all tasks."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    tasks = []
    for task_id, task in orchestrator.tasks.items():
        tasks.append(TaskResponse(
            id=task.id,
            title=task.title,
            status=task.status.value,
            result=task.result[:500] if task.result and len(task.result) > 500 else task.result,
            error_message=task.error_message,
            tokens_used=task.token_usage.total_tokens,
            cost_usd=task.token_usage.cost_usd,
            duration=task.duration,
            created_at=task.created_at.isoformat() if hasattr(task.created_at, 'isoformat') else str(task.created_at),
            role=task.role.value
        ))
    
    # Sort by created_at descending
    tasks.sort(key=lambda t: t.created_at, reverse=True)
    
    return {"tasks": tasks, "total": len(tasks)}


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    """Get specific task details."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    task = orchestrator.tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    return TaskResponse(
        id=task.id,
        title=task.title,
        status=task.status.value,
        result=task.result,
        error_message=task.error_message,
        tokens_used=task.token_usage.total_tokens,
        cost_usd=task.token_usage.cost_usd,
        duration=task.duration,
        created_at=task.created_at.isoformat() if hasattr(task.created_at, 'isoformat') else str(task.created_at),
        role=task.role.value
    )


@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a task."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    if task_id not in orchestrator.tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    del orchestrator.tasks[task_id]
    return {"message": f"Task {task_id} deleted"}


@app.post("/api/cache/clear")
async def clear_cache():
    """Clear the cache."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    orchestrator.clear_cache()
    return {"message": "Cache cleared successfully"}


# ==================== MODEL MANAGEMENT ====================

@app.get("/api/models")
async def list_models():
    """List all configured models with their status."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    models = []
    
    # Get OpenAI models
    if orchestrator.config.llm.openai:
        for model in orchestrator.config.llm.openai.models:
            is_available = 'openai' in orchestrator.model_manager.providers
            models.append({
                "name": model.name,
                "provider": "openai",
                "available": is_available,
                "is_local": False,
                "configured": bool(orchestrator.config.llm.openai.api_key),
                "max_tokens": model.max_tokens,
                "cost_per_1k_input": model.cost_per_1k_input,
                "cost_per_1k_output": model.cost_per_1k_output
            })
    
    # Get Ollama models
    if orchestrator.config.llm.ollama:
        ollama_available = 'ollama' in orchestrator.model_manager.providers
        for model in orchestrator.config.llm.ollama.models:
            models.append({
                "name": model.name,
                "provider": "ollama",
                "available": ollama_available,
                "is_local": True,
                "configured": True,
                "max_tokens": model.max_tokens,
                "cost_per_1k_input": model.cost_per_1k_input,
                "cost_per_1k_output": model.cost_per_1k_output
            })
    
    return {"models": models, "total": len(models)}


@app.post("/api/models/test")
async def test_model_connection(request: ConnectionTestRequest):
    """Test connection to a model provider."""
    try:
        if request.provider == "openai":
            # Test OpenAI connection
            api_key = request.api_key or os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                return {"success": False, "message": "API key not provided"}
            
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=request.base_url or "https://api.openai.com/v1")
            
            # Try a simple request
            model_name = request.model_name or "gpt-3.5-turbo"
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Hello, are you there?"}],
                max_tokens=10
            )
            
            return {
                "success": True,
                "message": f"Successfully connected to OpenAI ({model_name})",
                "model": model_name
            }
        
        elif request.provider == "ollama":
            # Test Ollama connection
            import requests
            base_url = request.base_url or "http://localhost:11434"
            model_name = request.model_name or "llama3"
            
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return {"success": False, "message": "Ollama server not responding"}
            
            # Check if model exists
            models = response.json().get("models", [])
            model_exists = any(m["name"] == model_name for m in models)
            
            return {
                "success": True,
                "message": f"Ollama is running. Model '{model_name}' {'found' if model_exists else 'not found'}",
                "model": model_name,
                "models_available": [m["name"] for m in models]
            }
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown provider: {request.provider}")
    
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.post("/api/models/configure")
async def configure_model(request: ModelConfigRequest):
    """Configure a new model (updates runtime config, not persistent)."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    # Note: This updates runtime config only. For persistence, update config.yaml
    if request.provider == "openai":
        if request.api_key:
            os.environ["OPENAI_API_KEY"] = request.api_key
        if request.base_url:
            os.environ["OPENAI_BASE_URL"] = request.base_url
        
        # Reload config
        reload_config()
        
        return {"message": f"OpenAI model '{request.name}' configured", "reload_required": True}
    
    elif request.provider == "ollama":
        if request.base_url:
            os.environ["OLLAMA_BASE_URL"] = request.base_url
        
        reload_config()
        return {"message": f"Ollama model '{request.name}' configured", "reload_required": True}
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {request.provider}")


# ==================== AGENT MANAGEMENT ====================

@app.get("/api/agents")
async def list_agents():
    """List all configured agents."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    agents = []
    for agent_id, agent in orchestrator.agents.items():
        agents.append({
            "id": agent.id,
            "name": agent.name,
            "role": agent.role.value,
            "model_name": agent.model_name,
            "available": agent.is_available,
            "tasks_completed": agent.tasks_completed,
            "success_rate": agent.success_rate,
            "system_prompt": agent.system_prompt[:100] + "..." if agent.system_prompt and len(agent.system_prompt) > 100 else agent.system_prompt
        })
    
    return {"agents": agents, "total": len(agents)}


@app.post("/api/agents", response_model=dict)
async def create_agent(request: AgentConfigRequest):
    """Create a new agent."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        role = AgentRole(request.role.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid role: {request.role}")
    
    agent = Agent(
        name=request.name,
        role=role,
        model_name=request.model_name,
        system_prompt=request.system_prompt
    )
    
    orchestrator.agents[agent.id] = agent
    
    return {
        "message": f"Agent '{agent.name}' created successfully",
        "agent_id": agent.id
    }


@app.put("/api/agents/{agent_id}")
async def update_agent(agent_id: str, request: AgentConfigRequest):
    """Update an existing agent."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    if agent_id not in orchestrator.agents:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    try:
        role = AgentRole(request.role.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid role: {request.role}")
    
    agent = orchestrator.agents[agent_id]
    agent.name = request.name
    agent.role = role
    agent.model_name = request.model_name
    agent.system_prompt = request.system_prompt
    
    return {"message": f"Agent '{agent.name}' updated successfully"}


@app.delete("/api/agents/{agent_id}")
async def delete_agent(agent_id: str):
    """Delete an agent."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    if agent_id not in orchestrator.agents:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    del orchestrator.agents[agent_id]
    return {"message": f"Agent {agent_id} deleted"}


# ==================== CONFIGURATION ====================

@app.get("/api/config")
async def get_configuration():
    """Get current configuration."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    config = orchestrator.config
    
    return {
        "llm": {
            "default_model": config.llm.default_model,
            "openai_configured": bool(config.llm.openai and config.llm.openai.api_key),
            "ollama_configured": bool(config.llm.ollama)
        },
        "cost": {
            "max_cost_per_task": config.cost.max_cost_per_task,
            "max_daily_cost": config.cost.max_daily_cost,
            "routing_strategy": config.cost.routing_strategy,
            "fallback_to_local": config.cost.fallback_to_local
        },
        "cache": {
            "enabled": config.cache.enabled,
            "semantic_search": config.cache.semantic_search
        },
        "orchestrator": {
            "max_retries": config.orchestrator.max_retries,
            "parallel": config.orchestrator.parallel,
            "default_complexity": config.orchestrator.default_complexity,
            "mode": config.orchestrator.mode
        }
    }


@app.post("/api/config/reload")
async def reload_configuration():
    """Reload configuration from file."""
    try:
        new_config = reload_config()
        # Reinitialize orchestrator with new config
        global orchestrator
        orchestrator = Orchestrator(new_config)
        return {"message": "Configuration reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload config: {str(e)}")


# ==================== WEBSOCKET ====================

@app.websocket("/ws/tasks")
async def task_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time task updates."""
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # Send periodic updates
            if orchestrator:
                status = orchestrator.get_status()
                await websocket.send_json({
                    "type": "status_update",
                    "data": status
                })
            await asyncio.sleep(3)
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        print("Client disconnected from WebSocket")


async def execute_task_background(task_id: str, request: TaskRequest):
    """Execute a task in the background."""
    if not orchestrator:
        return
    
    try:
        # Run the task
        task = orchestrator.run(
            task_description=request.description,
            decompose=request.decompose,
            num_steps=request.steps
        )
        
        # Notify connected clients
        for conn in active_connections:
            try:
                await conn.send_json({
                    "type": "task_complete",
                    "task_id": task.id,
                    "status": task.status.value,
                    "result": task.result
                })
            except:
                pass
    
    except Exception as e:
        # Notify about failure
        for conn in active_connections:
            try:
                await conn.send_json({
                    "type": "task_failed",
                    "task_id": task_id,
                    "error": str(e)
                })
            except:
                pass


def get_dashboard_html() -> str:
    """Generate the full dashboard HTML with model and agent management."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoDevAgent Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        header { background: linear-gradient(135deg, #2c3e50, #3498db); color: white; padding: 25px; margin-bottom: 25px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h1 { font-size: 28px; margin-bottom: 8px; }
        .subtitle { opacity: 0.9; font-size: 14px; }
        .nav-tabs { display: flex; gap: 10px; margin-bottom: 20px; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
        .nav-tab { padding: 10px 20px; border: none; background: #ecf0f1; cursor: pointer; border-radius: 5px; font-weight: 500; transition: all 0.3s; }
        .nav-tab:hover { background: #bdc3c7; }
        .nav-tab.active { background: #3498db; color: white; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 25px; }
        .stat-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); transition: transform 0.2s; }
        .stat-card:hover { transform: translateY(-3px); }
        .stat-value { font-size: 36px; font-weight: bold; color: #3498db; }
        .stat-label { color: #7f8c8d; font-size: 13px; margin-top: 8px; text-transform: uppercase; letter-spacing: 0.5px; }
        .card { background: white; padding: 25px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 20px; }
        .card h2 { margin-bottom: 20px; color: #2c3e50; font-size: 20px; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }
        textarea, input, select { width: 100%; padding: 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; margin-bottom: 12px; font-family: inherit; }
        textarea:focus, input:focus, select:focus { outline: none; border-color: #3498db; box-shadow: 0 0 0 3px rgba(52,152,219,0.1); }
        textarea { min-height: 120px; resize: vertical; }
        button { background: #3498db; color: white; border: none; padding: 12px 24px; border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: 500; transition: all 0.3s; display: inline-flex; align-items: center; gap: 8px; }
        button:hover { background: #2980b9; transform: translateY(-1px); box-shadow: 0 4px 8px rgba(0,0,0,0.15); }
        button.secondary { background: #95a5a6; }
        button.secondary:hover { background: #7f8c8d; }
        button.success { background: #27ae60; }
        button.success:hover { background: #229954; }
        button.danger { background: #e74c3c; }
        button.danger:hover { background: #c0392b; }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .btn-group { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 15px; }
        .model-item, .agent-item, .task-item { padding: 15px; border: 1px solid #eee; border-radius: 8px; margin-bottom: 12px; transition: all 0.3s; }
        .model-item:hover, .agent-item:hover, .task-item:hover { border-color: #3498db; box-shadow: 0 2px 8px rgba(52,152,219,0.15); }
        .item-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
        .item-title { font-weight: 600; color: #2c3e50; font-size: 16px; }
        .badge { display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; text-transform: uppercase; }
        .badge-success { background: #d5f5e3; color: #27ae60; }
        .badge-warning { background: #fdebd0; color: #f39c12; }
        .badge-danger { background: #fadbd8; color: #e74c3c; }
        .badge-info { background: #d6eaf8; color: #3498db; }
        .status-indicator { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }
        .status-online { background: #27ae60; }
        .status-offline { background: #e74c3c; }
        .item-meta { font-size: 12px; color: #7f8c8d; margin-top: 8px; }
        .grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }
        .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .loading { text-align: center; padding: 40px; color: #7f8c8d; }
        .spinner { border: 3px solid #f3f3f3; border-top: 3px solid #3498db; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 15px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .alert { padding: 15px 20px; border-radius: 8px; margin-bottom: 20px; display: none; }
        .alert-success { background: #d5f5e3; color: #27ae60; border: 1px solid #27ae60; }
        .alert-error { background: #fadbd8; color: #e74c3c; border: 1px solid #e74c3c; }
        .progress-bar { height: 6px; background: #ecf0f1; border-radius: 3px; overflow: hidden; margin-top: 10px; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #3498db, #2ecc71); width: 0%; transition: width 0.3s; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #eee; }
        th { background: #f8f9fa; font-weight: 600; color: #2c3e50; }
        tr:hover { background: #f8f9fa; }
        .config-section { margin-bottom: 25px; }
        .config-section h3 { margin-bottom: 15px; color: #34495e; font-size: 16px; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🚀 AutoDevAgent Dashboard</h1>
            <p class="subtitle">Autonomous Development System - Manage models, agents, and tasks from one interface</p>
        </header>
        
        <div class="nav-tabs">
            <button class="nav-tab active" onclick="showTab('tasks')">📋 Tasks</button>
            <button class="nav-tab" onclick="showTab('models')">🤖 Models</button>
            <button class="nav-tab" onclick="showTab('agents')">👥 Agents</button>
            <button class="nav-tab" onclick="showTab('config')">⚙️ Config</button>
        </div>
        
        <!-- TASKS TAB -->
        <div id="tasks-tab" class="tab-content active">
            <div class="stats" id="stats"></div>
            
            <div class="card">
                <h2>Create New Task</h2>
                <div class="form-row">
                    <div>
                        <label style="font-weight: 500; margin-bottom: 5px; display: block;">Title (optional)</label>
                        <input type="text" id="taskTitle" placeholder="Short task title">
                    </div>
                    <div>
                        <label style="font-weight: 500; margin-bottom: 5px; display: block;">Agent Role</label>
                        <select id="taskRole">
                            <option value="developer">Developer</option>
                            <option value="analyst">Analyst</option>
                            <option value="tester">Tester</option>
                            <option value="documentation">Documentation</option>
                        </select>
                    </div>
                </div>
                <label style="font-weight: 500; margin-bottom: 5px; display: block;">Description</label>
                <textarea id="taskDescription" placeholder="Describe what you want to accomplish..."></textarea>
                <div class="form-row">
                    <div>
                        <label style="font-weight: 500; margin-bottom: 5px; display: block;">Decompose into steps</label>
                        <select id="taskDecompose">
                            <option value="true">Yes</option>
                            <option value="false">No</option>
                        </select>
                    </div>
                    <div>
                        <label style="font-weight: 500; margin-bottom: 5px; display: block;">Number of steps</label>
                        <input type="number" id="taskSteps" min="1" max="10" value="5">
                    </div>
                </div>
                <button onclick="createTask()" style="margin-top: 15px;">▶️ Run Task</button>
            </div>
            
            <div class="card">
                <h2>Recent Tasks</h2>
                <div id="tasksList"><div class="loading">Loading tasks...</div></div>
            </div>
        </div>
        
        <!-- MODELS TAB -->
        <div id="models-tab" class="tab-content">
            <div class="card">
                <h2>Available Models</h2>
                <div id="modelsList"><div class="loading">Loading models...</div></div>
            </div>
            
            <div class="card">
                <h2>Test Connection</h2>
                <div class="form-row">
                    <div>
                        <label style="font-weight: 500; margin-bottom: 5px; display: block;">Provider</label>
                        <select id="testProvider">
                            <option value="openai">OpenAI</option>
                            <option value="ollama">Ollama (Local)</option>
                        </select>
                    </div>
                    <div>
                        <label style="font-weight: 500; margin-bottom: 5px; display: block;">Model Name</label>
                        <input type="text" id="testModel" placeholder="gpt-3.5-turbo or llama3">
                    </div>
                </div>
                <div id="apiKeyField">
                    <label style="font-weight: 500; margin-bottom: 5px; display: block;">API Key</label>
                    <input type="password" id="testApiKey" placeholder="sk-...">
                </div>
                <button onclick="testConnection()" class="success" style="margin-top: 15px;">🔌 Test Connection</button>
                <div id="testResult" class="alert" style="margin-top: 15px;"></div>
            </div>
        </div>
        
        <!-- AGENTS TAB -->
        <div id="agents-tab" class="tab-content">
            <div class="card">
                <h2>Create New Agent</h2>
                <div class="form-row">
                    <div>
                        <label style="font-weight: 500; margin-bottom: 5px; display: block;">Name</label>
                        <input type="text" id="agentName" placeholder="My Agent">
                    </div>
                    <div>
                        <label style="font-weight: 500; margin-bottom: 5px; display: block;">Role</label>
                        <select id="agentRole">
                            <option value="developer">Developer</option>
                            <option value="analyst">Analyst</option>
                            <option value="tester">Tester</option>
                            <option value="documentation">Documentation</option>
                            <option value="reviewer">Reviewer</option>
                        </select>
                    </div>
                </div>
                <label style="font-weight: 500; margin-bottom: 5px; display: block;">Model</label>
                <select id="agentModel"></select>
                <label style="font-weight: 500; margin-bottom: 5px; display: block;">System Prompt (optional)</label>
                <textarea id="agentPrompt" placeholder="Custom instructions for this agent..." style="min-height: 80px;"></textarea>
                <button onclick="createAgent()" style="margin-top: 15px;">➕ Create Agent</button>
            </div>
            
            <div class="card">
                <h2>Existing Agents</h2>
                <div id="agentsList"><div class="loading">Loading agents...</div></div>
            </div>
        </div>
        
        <!-- CONFIG TAB -->
        <div id="config-tab" class="tab-content">
            <div class="card">
                <h2>Current Configuration</h2>
                <div id="configDisplay"><div class="loading">Loading configuration...</div></div>
            </div>
            
            <div class="card">
                <h2>Actions</h2>
                <div class="btn-group">
                    <button onclick="reloadConfig()" class="success">🔄 Reload Configuration</button>
                    <button onclick="clearCache()" class="secondary">🗑️ Clear Cache</button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let ws = null;
        
        function showTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.nav-tab').forEach(btn => btn.classList.remove('active'));
            document.getElementById(tabName + '-tab').classList.add('active');
            event.target.classList.add('active');
            loadData();
        }
        
        async function loadData() {
            try {
                const statusRes = await fetch('/api/status');
                const status = await statusRes.json();
                
                document.getElementById('stats').innerHTML = `
                    <div class="stat-card">
                        <div class="stat-value">${status.total_tasks || 0}</div>
                        <div class="stat-label">Total Tasks</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">$${(status.total_cost_usd || 0).toFixed(4)}</div>
                        <div class="stat-label">Total Cost</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${status.total_tokens?.toLocaleString() || 0}</div>
                        <div class="stat-label">Tokens Used</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${status.available_agents || 0}/${status.total_agents || 0}</div>
                        <div class="stat-label">Available Agents</div>
                    </div>
                `;
                
                loadTasks();
                loadModels();
                loadAgents();
                loadConfig();
            } catch (error) {
                console.error('Error loading data:', error);
            }
        }
        
        async function loadTasks() {
            try {
                const res = await fetch('/api/tasks');
                const data = await res.json();
                const tasksList = document.getElementById('tasksList');
                
                if (data.tasks.length === 0) {
                    tasksList.innerHTML = '<div style="padding: 20px; text-align: center; color: #7f8c8d;">No tasks yet. Create your first task above!</div>';
                } else {
                    tasksList.innerHTML = data.tasks.map(task => `
                        <div class="task-item">
                            <div class="item-header">
                                <span class="item-title">${escapeHtml(task.title)}</span>
                                <span class="badge badge-${getTaskStatusClass(task.status)}">${task.status}</span>
                            </div>
                            <div class="item-meta">
                                Role: ${task.role} | Tokens: ${task.tokens_used} | Cost: $${task.cost_usd.toFixed(4)} | 
                                ${task.duration ? task.duration.toFixed(1) + 's' : ''} | 
                                ${new Date(task.created_at).toLocaleString()}
                            </div>
                            ${task.result ? `<div style="margin-top: 10px; padding: 10px; background: #f8f9fa; border-radius: 5px; font-size: 13px;">${escapeHtml(task.result.substring(0, 200))}${task.result.length > 200 ? '...' : ''}</div>` : ''}
                            ${task.error_message ? `<div style="margin-top: 10px; padding: 10px; background: #fadbd8; border-radius: 5px; font-size: 13px; color: #e74c3c;">Error: ${escapeHtml(task.error_message)}</div>` : ''}
                        </div>
                    `).join('');
                }
            } catch (error) {
                console.error('Error loading tasks:', error);
            }
        }
        
        function getTaskStatusClass(status) {
            const classes = { pending: 'warning', running: 'info', completed: 'success', failed: 'danger' };
            return classes[status] || 'info';
        }
        
        async function loadModels() {
            try {
                const res = await fetch('/api/models');
                const data = await res.json();
                const modelsList = document.getElementById('modelsList');
                const agentModel = document.getElementById('agentModel');
                
                if (data.models.length === 0) {
                    modelsList.innerHTML = '<div style="padding: 20px; text-align: center; color: #7f8c8d;">No models configured</div>';
                    return;
                }
                
                modelsList.innerHTML = data.models.map(model => `
                    <div class="model-item">
                        <div class="item-header">
                            <div>
                                <span class="status-indicator ${model.available ? 'status-online' : 'status-offline'}"></span>
                                <span class="item-title">${model.name}</span>
                            </div>
                            <span class="badge badge-${model.provider === 'ollama' ? 'success' : 'info'}">${model.provider}</span>
                        </div>
                        <div class="item-meta">
                            Max tokens: ${model.max_tokens.toLocaleString()} | 
                            Input: $${model.cost_per_1k_input}/1k | Output: $${model.cost_per_1k_output}/1k |
                            ${model.is_local ? '🏠 Local' : '☁️ Cloud'} |
                            ${model.configured ? '✅ Configured' : '❌ Not configured'}
                        </div>
                    </div>
                `).join('');
                
                // Populate agent model dropdown
                agentModel.innerHTML = data.models.filter(m => m.configured).map(model => 
                    `<option value="${model.name}">${model.name} (${model.provider})</option>`
                ).join('');
            } catch (error) {
                console.error('Error loading models:', error);
            }
        }
        
        async function loadAgents() {
            try {
                const res = await fetch('/api/agents');
                const data = await res.json();
                const agentsList = document.getElementById('agentsList');
                
                if (data.agents.length === 0) {
                    agentsList.innerHTML = '<div style="padding: 20px; text-align: center; color: #7f8c8d;">No agents configured</div>';
                    return;
                }
                
                agentsList.innerHTML = data.agents.map(agent => `
                    <div class="agent-item">
                        <div class="item-header">
                            <div>
                                <span class="item-title">${escapeHtml(agent.name)}</span>
                                <span class="badge badge-info" style="margin-left: 10px;">${agent.role}</span>
                            </div>
                            <span class="status-indicator ${agent.available ? 'status-online' : 'status-offline'}" style="display:inline-block;"></span>
                        </div>
                        <div class="item-meta">
                            Model: ${agent.model_name} | Tasks completed: ${agent.tasks_completed} | 
                            Success rate: ${(agent.success_rate * 100).toFixed(1)}%
                        </div>
                        ${agent.system_prompt ? `<div style="margin-top: 8px; font-size: 12px; color: #7f8c8d; font-style: italic;">"${escapeHtml(agent.system_prompt)}"</div>` : ''}
                    </div>
                `).join('');
            } catch (error) {
                console.error('Error loading agents:', error);
            }
        }
        
        async function loadConfig() {
            try {
                const res = await fetch('/api/config');
                const config = await res.json();
                const configDisplay = document.getElementById('configDisplay');
                
                configDisplay.innerHTML = `
                    <div class="config-section">
                        <h3>LLM Configuration</h3>
                        <table>
                            <tr><td>Default Model</td><td><strong>${config.llm.default_model}</strong></td></tr>
                            <tr><td>OpenAI</td><td>${config.llm.openai_configured ? '<span class="badge badge-success">Configured</span>' : '<span class="badge badge-danger">Not configured</span>'}</td></tr>
                            <tr><td>Ollama</td><td>${config.llm.ollama_configured ? '<span class="badge badge-success">Configured</span>' : '<span class="badge badge-danger">Not configured</span>'}</td></tr>
                        </table>
                    </div>
                    <div class="config-section">
                        <h3>Cost Limits</h3>
                        <table>
                            <tr><td>Max per Task</td><td>$${config.cost.max_cost_per_task.toFixed(2)}</td></tr>
                            <tr><td>Max Daily</td><td>$${config.cost.max_daily_cost.toFixed(2)}</td></tr>
                            <tr><td>Strategy</td><td><span class="badge badge-info">${config.cost.routing_strategy}</span></td></tr>
                            <tr><td>Fallback to Local</td><td>${config.cost.fallback_to_local ? '✅ Yes' : '❌ No'}</td></tr>
                        </table>
                    </div>
                    <div class="config-section">
                        <h3>Orchestrator</h3>
                        <table>
                            <tr><td>Max Retries</td><td>${config.orchestrator.max_retries}</td></tr>
                            <tr><td>Parallel Execution</td><td>${config.orchestrator.parallel ? '✅ Enabled' : '❌ Disabled'}</td></tr>
                            <tr><td>Default Complexity</td><td>${config.orchestrator.default_complexity}/5</td></tr>
                            <tr><td>Mode</td><td><span class="badge badge-info">${config.orchestrator.mode}</span></td></tr>
                        </table>
                    </div>
                `;
            } catch (error) {
                console.error('Error loading config:', error);
            }
        }
        
        async function createTask() {
            const description = document.getElementById('taskDescription').value.trim();
            if (!description) {
                alert('Please enter a task description');
                return;
            }
            
            const btn = event.target;
            btn.disabled = true;
            btn.innerHTML = '⏳ Creating...';
            
            try {
                const res = await fetch('/api/tasks', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        title: document.getElementById('taskTitle').value.trim(),
                        description: description,
                        role: document.getElementById('taskRole').value,
                        decompose: document.getElementById('taskDecompose').value === 'true',
                        steps: parseInt(document.getElementById('taskSteps').value)
                    })
                });
                
                const data = await res.json();
                
                if (res.ok) {
                    document.getElementById('taskDescription').value = '';
                    document.getElementById('taskTitle').value = '';
                    showAlert(`Task created successfully! ID: ${data.id}`, 'success');
                    setTimeout(() => loadData(), 500);
                } else {
                    showAlert('Error: ' + data.detail, 'error');
                }
            } catch (error) {
                showAlert('Error creating task: ' + error.message, 'error');
            } finally {
                btn.disabled = false;
                btn.innerHTML = '▶️ Run Task';
            }
        }
        
        async function testConnection() {
            const provider = document.getElementById('testProvider').value;
            const btn = event.target;
            btn.disabled = true;
            btn.innerHTML = '⏳ Testing...';
            
            try {
                const res = await fetch('/api/models/test', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        provider: provider,
                        api_key: document.getElementById('testApiKey').value,
                        model_name: document.getElementById('testModel').value || undefined
                    })
                });
                
                const data = await res.json();
                const resultDiv = document.getElementById('testResult');
                resultDiv.style.display = 'block';
                
                if (data.success) {
                    resultDiv.className = 'alert alert-success';
                    resultDiv.innerHTML = '✅ ' + data.message;
                    if (data.models_available) {
                        resultDiv.innerHTML += '<br><br><strong>Available models:</strong> ' + data.models_available.join(', ');
                    }
                } else {
                    resultDiv.className = 'alert alert-error';
                    resultDiv.innerHTML = '❌ ' + data.message;
                }
            } catch (error) {
                showAlert('Error testing connection: ' + error.message, 'error');
            } finally {
                btn.disabled = false;
                btn.innerHTML = '🔌 Test Connection';
            }
        }
        
        async function createAgent() {
            const name = document.getElementById('agentName').value.trim();
            if (!name) {
                alert('Please enter an agent name');
                return;
            }
            
            try {
                const res = await fetch('/api/agents', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name: name,
                        role: document.getElementById('agentRole').value,
                        model_name: document.getElementById('agentModel').value,
                        system_prompt: document.getElementById('agentPrompt').value.trim()
                    })
                });
                
                const data = await res.json();
                
                if (res.ok) {
                    document.getElementById('agentName').value = '';
                    document.getElementById('agentPrompt').value = '';
                    showAlert(data.message, 'success');
                    setTimeout(() => loadAgents(), 500);
                } else {
                    showAlert('Error: ' + data.detail, 'error');
                }
            } catch (error) {
                showAlert('Error creating agent: ' + error.message, 'error');
            }
        }
        
        async function reloadConfig() {
            try {
                const res = await fetch('/api/config/reload', { method: 'POST' });
                const data = await res.json();
                if (res.ok) {
                    showAlert(data.message, 'success');
                    setTimeout(loadData, 1000);
                } else {
                    showAlert('Error: ' + data.detail, 'error');
                }
            } catch (error) {
                showAlert('Error reloading config: ' + error.message, 'error');
            }
        }
        
        async function clearCache() {
            try {
                const res = await fetch('/api/cache/clear', { method: 'POST' });
                const data = await res.json();
                showAlert(data.message, 'success');
            } catch (error) {
                showAlert('Error clearing cache: ' + error.message, 'error');
            }
        }
        
        function showAlert(message, type) {
            const alert = document.createElement('div');
            alert.className = `alert alert-${type}`;
            alert.style.display = 'block';
            alert.textContent = message;
            document.querySelector('.container').insertBefore(alert, document.querySelector('.container').firstChild);
            setTimeout(() => alert.remove(), 5000);
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Toggle API key field based on provider
        document.addEventListener('change', (e) => {
            if (e.target.id === 'testProvider') {
                const apiKeyField = document.getElementById('apiKeyField');
                apiKeyField.style.display = e.target.value === 'openai' ? 'block' : 'none';
            }
        });
        
        // Connect WebSocket for real-time updates
        function connectWebSocket() {
            ws = new WebSocket(`ws://${window.location.host}/ws/tasks`);
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type === 'status_update') {
                    // Update stats in real-time
                } else if (data.type === 'task_complete') {
                    showAlert(`Task ${data.task_id} completed!`, 'success');
                    loadTasks();
                } else if (data.type === 'task_failed') {
                    showAlert(`Task ${data.task_id} failed: ${data.error}`, 'error');
                    loadTasks();
                }
            };
            ws.onclose = () => {
                setTimeout(connectWebSocket, 3000);
            };
        }
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            loadData();
            connectWebSocket();
            setInterval(loadData, 10000); // Refresh every 10 seconds
        });
    </script>
</body>
</html>"""


def get_simple_dashboard_html() -> str:
    """Fallback simple dashboard."""
    return get_dashboard_html()
