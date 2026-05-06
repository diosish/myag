"""
Web UI for AutoDevAgent using FastAPI.
Provides REST API and simple dashboard for task management.
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
import json
from pathlib import Path
from datetime import datetime

from ..models.config import get_config
from ..orchestrator.core import Orchestrator
from ..models.domain import Task, TaskStatus, AgentRole


app = FastAPI(
    title="AutoDevAgent API",
    description="REST API for AutoDevAgent orchestration system",
    version="0.1.0"
)

# Global orchestrator instance
orchestrator: Optional[Orchestrator] = None
task_queue: asyncio.Queue = asyncio.Queue()
active_tasks: Dict[str, Dict[str, Any]] = {}


class TaskRequest(BaseModel):
    """Request model for creating a task."""
    description: str
    decompose: bool = True
    steps: int = 5
    model: Optional[str] = None


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


@app.on_event("startup")
async def startup_event():
    """Initialize orchestrator on startup."""
    global orchestrator
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
        return HTMLResponse(content=get_simple_dashboard_html())


@app.get("/api/status")
async def get_status():
    """Get current system status."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    status = orchestrator.get_status()
    return {
        **status,
        "uptime": datetime.now().isoformat(),
        "version": "0.1.0"
    }


@app.post("/api/tasks", response_model=TaskResponse)
async def create_task(request: TaskRequest):
    """Create and execute a new task."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Store task info
    active_tasks[task_id] = {
        "description": request.description,
        "status": "pending",
        "created_at": datetime.now().isoformat()
    }
    
    try:
        # Run task in background
        asyncio.create_task(execute_task(task_id, request))
        
        return TaskResponse(
            id=task_id,
            title=request.description[:100],
            status="pending",
            created_at=active_tasks[task_id]["created_at"]
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
            created_at=task.created_at.isoformat() if hasattr(task.created_at, 'isoformat') else str(task.created_at)
        ))
    
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
        created_at=task.created_at.isoformat() if hasattr(task.created_at, 'isoformat') else str(task.created_at)
    )


@app.post("/api/cache/clear")
async def clear_cache():
    """Clear the cache."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    orchestrator.clear_cache()
    return {"message": "Cache cleared successfully"}


@app.websocket("/ws/tasks")
async def task_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time task updates."""
    await websocket.accept()
    try:
        while True:
            # Send periodic updates
            if orchestrator:
                status = orchestrator.get_status()
                await websocket.send_json({
                    "type": "status_update",
                    "data": status
                })
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        print("Client disconnected from WebSocket")


async def execute_task(task_id: str, request: TaskRequest):
    """Execute a task in the background."""
    global active_tasks
    
    try:
        active_tasks[task_id]["status"] = "running"
        
        # Run the task
        task = orchestrator.run(
            task_description=request.description,
            decompose=request.decompose,
            num_steps=request.steps
        )
        
        # Update status
        active_tasks[task_id]["status"] = task.status.value
        active_tasks[task_id]["result"] = task.result
        active_tasks[task_id]["error_message"] = task.error_message
        
    except Exception as e:
        active_tasks[task_id]["status"] = "failed"
        active_tasks[task_id]["error_message"] = str(e)


def get_simple_dashboard_html() -> str:
    """Generate a simple dashboard HTML."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoDevAgent Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        header { background: #2c3e50; color: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; }
        h1 { font-size: 24px; margin-bottom: 10px; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .stat-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .stat-value { font-size: 32px; font-weight: bold; color: #3498db; }
        .stat-label { color: #7f8c8d; font-size: 14px; margin-top: 5px; }
        .form-section { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        textarea { width: 100%; padding: 12px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; resize: vertical; min-height: 100px; }
        button { background: #3498db; color: white; border: none; padding: 12px 24px; border-radius: 4px; cursor: pointer; font-size: 16px; margin-top: 10px; }
        button:hover { background: #2980b9; }
        .tasks-list { background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); overflow: hidden; }
        .task-item { padding: 15px 20px; border-bottom: 1px solid #eee; }
        .task-item:last-child { border-bottom: none; }
        .task-title { font-weight: 600; margin-bottom: 5px; }
        .task-status { display: inline-block; padding: 3px 8px; border-radius: 3px; font-size: 12px; }
        .status-pending { background: #f39c12; color: white; }
        .status-running { background: #3498db; color: white; }
        .status-completed { background: #27ae60; color: white; }
        .status-failed { background: #e74c3c; color: white; }
        .task-meta { font-size: 12px; color: #7f8c8d; margin-top: 8px; }
        .refresh-btn { position: fixed; bottom: 20px; right: 20px; background: #27ae60; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🚀 AutoDevAgent Dashboard</h1>
            <p>Autonomous Development System</p>
        </header>
        
        <div class="stats" id="stats"></div>
        
        <div class="form-section">
            <h2>Create New Task</h2>
            <textarea id="taskDescription" placeholder="Describe what you want to accomplish..."></textarea>
            <button onclick="createTask()">Run Task</button>
        </div>
        
        <div class="tasks-list">
            <h2 style="padding: 15px 20px;">Recent Tasks</h2>
            <div id="tasksList"></div>
        </div>
    </div>
    
    <button class="refresh-btn" onclick="loadData()">🔄 Refresh</button>
    
    <script>
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
                
                const tasksRes = await fetch('/api/tasks');
                const tasksData = await tasksRes.json();
                
                const tasksList = document.getElementById('tasksList');
                if (tasksData.tasks.length === 0) {
                    tasksList.innerHTML = '<div style="padding: 20px; text-align: center; color: #7f8c8d;">No tasks yet</div>';
                } else {
                    tasksList.innerHTML = tasksData.tasks.map(task => `
                        <div class="task-item">
                            <div class="task-title">${escapeHtml(task.title)}</div>
                            <span class="task-status status-${task.status}">${task.status}</span>
                            <div class="task-meta">
                                Tokens: ${task.tokens_used} | Cost: $${task.cost_usd.toFixed(4)} | 
                                ${task.duration ? task.duration.toFixed(1) + 's' : ''}
                            </div>
                        </div>
                    `).join('');
                }
            } catch (error) {
                console.error('Error loading data:', error);
            }
        }
        
        async function createTask() {
            const description = document.getElementById('taskDescription').value.trim();
            if (!description) {
                alert('Please enter a task description');
                return;
            }
            
            try {
                const res = await fetch('/api/tasks', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ description, decompose: true, steps: 5 })
                });
                
                const data = await res.json();
                
                if (res.ok) {
                    document.getElementById('taskDescription').value = '';
                    alert('Task created! ID: ' + data.id);
                    setTimeout(loadData, 1000);
                } else {
                    alert('Error: ' + data.detail);
                }
            } catch (error) {
                alert('Error creating task: ' + error.message);
            }
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Load data on page load and refresh every 5 seconds
        loadData();
        setInterval(loadData, 5000);
    </script>
</body>
</html>"""
