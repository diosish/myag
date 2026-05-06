"""Tests for domain models."""

import pytest
from datetime import datetime, timedelta

from autodev_agent.models.domain import (
    Task, TaskStatus, TaskPriority, AgentRole, TokenUsage,
    Agent, Skill, DecompositionPlan, TaskArtifact
)


class TestTokenUsage:
    """Tests for TokenUsage model."""
    
    def test_creation(self):
        """Test basic creation."""
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.cost_usd == 0.0
    
    def test_addition(self):
        """Test adding two TokenUsage objects."""
        usage1 = TokenUsage(input_tokens=100, output_tokens=50, cost_usd=0.01)
        usage2 = TokenUsage(input_tokens=200, output_tokens=100, cost_usd=0.02)
        
        total = usage1.add(usage2)
        
        assert total.input_tokens == 300
        assert total.output_tokens == 150
        assert total.cost_usd == 0.03


class TestTask:
    """Tests for Task model."""
    
    def test_creation(self):
        """Test basic task creation."""
        task = Task(title="Test Task", description="Do something")
        
        assert task.title == "Test Task"
        assert task.description == "Do something"
        assert task.status == TaskStatus.PENDING
        assert task.priority == TaskPriority.MEDIUM
        assert task.role == AgentRole.DEVELOPER
        assert task.retry_count == 0
    
    def test_mark_started(self):
        """Test marking task as started."""
        task = Task(title="Test", description="Test desc")
        
        assert task.status == TaskStatus.PENDING
        assert task.started_at is None
        
        task.mark_started()
        
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.started_at is not None
    
    def test_mark_completed(self):
        """Test marking task as completed."""
        task = Task(title="Test", description="Test desc")
        task.mark_started()
        
        task.mark_completed("Done!")
        
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "Done!"
        assert task.completed_at is not None
    
    def test_mark_failed(self):
        """Test marking task as failed."""
        task = Task(title="Test", description="Test desc")
        task.mark_started()
        
        task.mark_failed("Something went wrong")
        
        assert task.status == TaskStatus.FAILED
        assert task.error_message == "Something went wrong"
    
    def test_can_retry(self):
        """Test retry logic."""
        task = Task(title="Test", description="Test desc", max_retries=3)
        task.mark_failed("Error")
        
        assert task.can_retry() is True
        
        # Simulate max retries reached
        task.retry_count = 3
        assert task.can_retry() is False
    
    def test_reset_for_retry(self):
        """Test resetting task for retry."""
        task = Task(title="Test", description="Test desc")
        task.mark_started()
        task.mark_failed("Error")
        task.retry_count = 1
        
        task.reset_for_retry()
        
        assert task.status == TaskStatus.PENDING
        assert task.error_message is None
        assert task.started_at is None
        assert task.retry_count == 2
    
    def test_duration(self):
        """Test duration calculation."""
        task = Task(title="Test", description="Test desc")
        task.started_at = datetime.now() - timedelta(seconds=30)
        task.completed_at = datetime.now()
        
        assert task.duration is not None
        assert 29 <= task.duration <= 31  # Allow some timing variance


class TestAgent:
    """Tests for Agent model."""
    
    def test_creation(self):
        """Test basic agent creation."""
        agent = Agent(name="TestBot", role=AgentRole.DEVELOPER)
        
        assert agent.name == "TestBot"
        assert agent.role == AgentRole.DEVELOPER
        assert agent.is_available is True
        assert agent.tasks_completed == 0
        assert agent.tasks_failed == 0
    
    def test_assign_task(self):
        """Test assigning task to agent."""
        agent = Agent(name="TestBot", role=AgentRole.DEVELOPER)
        
        agent.assign_task("task-123")
        
        assert agent.active_task == "task-123"
        assert agent.is_available is False
    
    def test_release_task(self):
        """Test releasing task from agent."""
        agent = Agent(name="TestBot", role=AgentRole.DEVELOPER)
        agent.assign_task("task-123")
        
        agent.release_task()
        
        assert agent.active_task is None
        assert agent.is_available is True
    
    def test_success_rate(self):
        """Test success rate calculation."""
        agent = Agent(name="TestBot", role=AgentRole.DEVELOPER)
        
        # No tasks yet
        assert agent.success_rate == 1.0
        
        # Record some completions and failures
        agent.tasks_completed = 8
        agent.tasks_failed = 2
        
        assert agent.success_rate == 0.8


class TestDecompositionPlan:
    """Tests for DecompositionPlan model."""
    
    def test_creation(self):
        """Test plan creation."""
        original = Task(title="Main Task", description="Do everything")
        subtask1 = Task(title="Step 1", description="First step")
        subtask2 = Task(title="Step 2", description="Second step")
        
        plan = DecompositionPlan(
            original_task=original,
            subtasks=[subtask1, subtask2],
            strategy="sequential"
        )
        
        assert len(plan.subtasks) == 2
        assert plan.strategy == "sequential"
    
    def test_total_estimated_tokens(self):
        """Test token estimation."""
        original = Task(title="Main", description="Desc")
        subtask1 = Task(title="Step 1", description="Desc")
        subtask1.token_usage.input_tokens = 100
        subtask1.token_usage.output_tokens = 50
        subtask1.token_usage.total_tokens = 150  # Set explicitly after modification
        
        subtask2 = Task(title="Step 2", description="Desc")
        subtask2.token_usage.input_tokens = 200
        subtask2.token_usage.output_tokens = 100
        subtask2.token_usage.total_tokens = 300  # Set explicitly after modification
        
        plan = DecompositionPlan(
            original_task=original,
            subtasks=[subtask1, subtask2]
        )
        
        assert plan.total_estimated_tokens == 450
