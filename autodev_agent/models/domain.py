"""
Core domain models for AutoDevAgent.
Defines Task, Agent, and related entities.
"""

import hashlib
import time
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Set
from pydantic import BaseModel, Field


def generate_task_id() -> str:
    """Generate a unique task ID."""
    return hashlib.md5(f"{time.time()}-{uuid.uuid4()}".encode()).hexdigest()[:12]


def generate_agent_id() -> str:
    """Generate a unique agent ID."""
    return f"agent_{hashlib.md5(f'{time.time()}-{uuid.uuid4()}'.encode()).hexdigest()[:8]}"


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"  # Waiting for user input
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentRole(str, Enum):
    """Predefined agent roles."""
    ORCHESTRATOR = "orchestrator"
    ANALYST = "analyst"
    DEVELOPER = "developer"
    TESTER = "tester"
    DOCUMENTATION = "documentation"
    REVIEWER = "reviewer"


class TokenUsage(BaseModel):
    """Token usage statistics."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    
    def model_post_init(self, __context):
        """Calculate total_tokens if not provided."""
        self.total_tokens = self.input_tokens + self.output_tokens
    
    def add(self, other: 'TokenUsage') -> 'TokenUsage':
        """Add another TokenUsage to this one."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=self.cost_usd + other.cost_usd
        )


class TaskArtifact(BaseModel):
    """An artifact produced by a task."""
    name: str
    type: str  # file, code, doc, test, etc.
    path: Optional[str] = None
    content: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True


class Task(BaseModel):
    """Represents a task to be executed by an agent."""
    
    id: str = Field(default_factory=generate_task_id)
    title: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    role: AgentRole = AgentRole.DEVELOPER
    
    # Parent/child relationships
    parent_id: Optional[str] = None
    child_ids: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)  # Other task IDs this depends on
    
    # Execution context
    context: Dict[str, Any] = Field(default_factory=dict)
    files_involved: List[str] = Field(default_factory=list)
    
    # Results
    result: Optional[str] = None
    artifacts: List[TaskArtifact] = Field(default_factory=list)
    error_message: Optional[str] = None
    
    # Cost tracking
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Retry info
    retry_count: int = 0
    max_retries: int = 3
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate task duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def is_blocked(self) -> bool:
        """Check if task is blocked."""
        return self.status == TaskStatus.BLOCKED
    
    @property
    def can_start(self) -> bool:
        """Check if task can start (all dependencies completed)."""
        # In a real implementation, check if all dependency tasks are completed
        return self.status == TaskStatus.PENDING and len(self.dependencies) == 0
    
    def mark_started(self):
        """Mark task as started."""
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now()
    
    def mark_completed(self, result: str):
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = datetime.now()
    
    def mark_failed(self, error: str):
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.error_message = error
        self.completed_at = datetime.now()
    
    def mark_blocked(self, reason: str = "Waiting for user input"):
        """Mark task as blocked."""
        self.status = TaskStatus.BLOCKED
        self.context['blocked_reason'] = reason
    
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries and self.status == TaskStatus.FAILED
    
    def reset_for_retry(self):
        """Reset task state for retry."""
        self.status = TaskStatus.PENDING
        self.error_message = None
        self.started_at = None
        self.completed_at = None
        self.retry_count += 1


class Agent(BaseModel):
    """Represents an autonomous agent with specific skills."""
    
    id: str = Field(default_factory=generate_agent_id)
    name: str
    role: AgentRole
    skills: List[str] = Field(default_factory=list)
    
    # Model assignment
    model_name: str = "gpt-4"
    system_prompt: Optional[str] = None
    
    # Constraints
    max_context_tokens: int = 4096
    allowed_file_patterns: List[str] = Field(default_factory=list)
    
    # Statistics
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_token_usage: TokenUsage = Field(default_factory=TokenUsage)
    
    # State
    active_task: Optional[str] = None  # Task ID
    is_available: bool = True
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def success_rate(self) -> float:
        """Calculate agent success rate."""
        total = self.tasks_completed + self.tasks_failed
        if total == 0:
            return 1.0
        return self.tasks_completed / total
    
    def assign_task(self, task_id: str):
        """Assign a task to this agent."""
        self.active_task = task_id
        self.is_available = False
    
    def release_task(self):
        """Release the current task."""
        self.active_task = None
        self.is_available = True
    
    def record_completion(self):
        """Record successful task completion."""
        self.tasks_completed += 1
        self.release_task()
    
    def record_failure(self):
        """Record task failure."""
        self.tasks_failed += 1
        self.release_task()


class Skill(BaseModel):
    """Represents a skill that agents can have."""
    
    name: str
    description: str
    system_prompt_template: Optional[str] = None
    tools: List[str] = Field(default_factory=list)
    file_patterns: List[str] = Field(default_factory=list)
    
    # Example prompts for this skill
    example_prompts: List[str] = Field(default_factory=list)


class DecompositionPlan(BaseModel):
    """Result of task decomposition."""
    
    original_task: Task
    subtasks: List[Task] = Field(default_factory=list)
    strategy: str = "sequential"  # sequential, parallel, dag
    estimated_cost: float = 0.0
    estimated_tokens: int = 0
    
    @property
    def total_estimated_tokens(self) -> int:
        """Sum of estimated tokens for all subtasks."""
        return sum(task.token_usage.total_tokens for task in self.subtasks)
