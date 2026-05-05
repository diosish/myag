"""
Core Orchestrator for AutoDevAgent.
Manages task execution, agent assignment, and workflow coordination.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

from ..models.config import Config, get_config
from ..models.domain import (
    Task, TaskStatus, Agent, AgentRole, TokenUsage, DecompositionPlan
)
from ..models.provider import ModelManager, ModelResponse


logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Main orchestrator that coordinates task execution.
    
    Responsibilities:
    - Accept tasks from users
    - Decompose complex tasks into subtasks
    - Assign tasks to appropriate agents
    - Track progress and costs
    - Handle failures and retries
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.model_manager = ModelManager(self.config)
        
        # Task storage
        self.tasks: Dict[str, Task] = {}
        self.agents: Dict[str, Agent] = {}
        
        # Execution state
        self.current_task_id: Optional[str] = None
        self.execution_history: List[Dict[str, Any]] = []
        
        # Cost tracking
        self.total_cost_usd: float = 0.0
        self.total_tokens: int = 0
        
        # Initialize default agents
        self._initialize_default_agents()
        
        # Setup logging
        self._setup_logging()
        
        logger.info("Orchestrator initialized")
    
    def _setup_logging(self):
        """Configure logging based on config."""
        log_config = self.config.logging
        
        # Create logs directory if needed
        log_path = Path(log_config.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, log_config.level),
            format=log_config.format,
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
    
    def _initialize_default_agents(self):
        """Create default agents for common roles."""
        default_agents = [
            Agent(name="Analyst", role=AgentRole.ANALYST, model_name=self.config.llm.default_model),
            Agent(name="Developer", role=AgentRole.DEVELOPER, model_name=self.config.llm.default_model),
            Agent(name="Tester", role=AgentRole.TESTER, model_name=self.config.llm.default_model),
            Agent(name="Documenter", role=AgentRole.DOCUMENTATION, model_name=self.config.llm.default_model),
        ]
        
        for agent in default_agents:
            self.agents[agent.id] = agent
        
        logger.info(f"Initialized {len(default_agents)} default agents")
    
    def create_task(
        self,
        title: str,
        description: str,
        role: AgentRole = AgentRole.DEVELOPER,
        priority: str = "medium",
        context: Optional[Dict[str, Any]] = None
    ) -> Task:
        """Create a new task."""
        task = Task(
            title=title,
            description=description,
            role=role,
            priority=priority,
            context=context or {},
            max_retries=self.config.orchestrator.max_retries
        )
        
        self.tasks[task.id] = task
        logger.info(f"Created task {task.id}: {title}")
        
        return task
    
    def decompose_task(self, task: Task, num_subtasks: int = 3) -> DecompositionPlan:
        """
        Decompose a complex task into smaller subtasks.
        
        Uses LLM to break down the task into manageable steps.
        """
        logger.info(f"Decomposing task {task.id} into ~{num_subtasks} subtasks")
        
        system_prompt = """You are an expert task planner. Your job is to break down complex tasks into clear, actionable subtasks.
Each subtask should be:
- Specific and actionable
- Independently executable
- Ordered logically (if there are dependencies)

Respond with a JSON array of subtasks, each containing:
- title: Short title for the subtask
- description: Detailed description of what needs to be done
- role: The agent role best suited for this task (analyst, developer, tester, documentation)
- dependencies: List of indices (0-based) of subtasks this one depends on"""

        prompt = f"""Break down the following task into {num_subtasks} subtasks:

Task: {task.title}
Description: {task.description}

Provide exactly {num_subtasks} subtasks in JSON format."""

        try:
            response = self.model_manager.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3  # Lower temperature for more structured output
            )
            
            # Parse the response
            content = response.content.strip()
            
            # Try to extract JSON from the response
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                subtask_data = json.loads(json_str)
            else:
                # Fallback: create generic subtasks
                logger.warning("Could not parse JSON from decomposition response, creating generic subtasks")
                subtask_data = [
                    {"title": f"Step {i+1}", "description": f"Complete step {i+1} of the task", "role": "developer", "dependencies": [i-1] if i > 0 else []}
                    for i in range(num_subtasks)
                ]
            
            # Create subtask objects
            subtasks = []
            for i, data in enumerate(subtask_data):
                role_str = data.get('role', 'developer')
                try:
                    role = AgentRole(role_str.upper())
                except ValueError:
                    role = AgentRole.DEVELOPER
                
                subtask = Task(
                    title=data.get('title', f'Step {i+1}'),
                    description=data.get('description', ''),
                    role=role,
                    parent_id=task.id,
                    dependencies=[]  # Will be resolved below
                )
                
                # Set up dependencies (relative to subtask list)
                dep_indices = data.get('dependencies', [])
                for dep_idx in dep_indices:
                    if 0 <= dep_idx < i:  # Only allow backward dependencies
                        subtask.dependencies.append(f"{task.id}_sub_{dep_idx}")
                
                # Generate ID that includes parent task ID
                subtask.id = f"{task.id}_sub_{i}"
                
                subtasks.append(subtask)
                self.tasks[subtask.id] = subtask
            
            # Update parent task with child IDs
            task.child_ids = [st.id for st in subtasks]
            
            # Update dependencies to use actual subtask IDs
            for i, subtask in enumerate(subtasks):
                dep_indices = subtask_data[i].get('dependencies', [])
                subtask.dependencies = [f"{task.id}_sub_{idx}" for idx in dep_indices if 0 <= idx < i]
            
            # Calculate estimates
            plan = DecompositionPlan(
                original_task=task,
                subtasks=subtasks,
                strategy="sequential",
                estimated_tokens=response.total_tokens
            )
            
            # Update cost tracking
            self._record_token_usage(response, task)
            
            logger.info(f"Decomposition complete: {len(subtasks)} subtasks created")
            
            return plan
            
        except Exception as e:
            logger.error(f"Failed to decompose task: {str(e)}")
            # Return a simple sequential plan as fallback
            subtasks = []
            for i in range(num_subtasks):
                subtask = Task(
                    title=f"Step {i+1}",
                    description=f"Complete step {i+1} of: {task.title}",
                    role=task.role,
                    parent_id=task.id,
                    dependencies=[f"{task.id}_sub_{i-1}"] if i > 0 else []
                )
                subtask.id = f"{task.id}_sub_{i}"
                subtasks.append(subtask)
                self.tasks[subtask.id] = subtask
            
            task.child_ids = [st.id for st in subtasks]
            
            return DecompositionPlan(
                original_task=task,
                subtasks=subtasks,
                strategy="sequential"
            )
    
    def execute_task(self, task: Task, agent: Optional[Agent] = None) -> Task:
        """
        Execute a single task using an appropriate agent.
        """
        logger.info(f"Executing task {task.id}: {task.title}")
        
        # Select agent
        if not agent:
            agent = self._select_agent_for_task(task)
        
        if not agent:
            task.mark_failed("No available agent found")
            return task
        
        # Check dependencies
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if dep_task and dep_task.status != TaskStatus.COMPLETED:
                logger.info(f"Task {task.id} blocked by incomplete dependency {dep_id}")
                task.mark_blocked(f"Waiting for dependency {dep_id}")
                return task
        
        # Mark as started
        task.mark_started()
        agent.assign_task(task.id)
        self.current_task_id = task.id
        
        try:
            # Build prompt for the agent
            system_prompt = self._build_system_prompt(agent)
            user_prompt = self._build_user_prompt(task)
            
            # Execute with LLM
            response = self.model_manager.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                model_name=agent.model_name,
                temperature=0.7
            )
            
            # Process result
            task.mark_completed(response.content)
            agent.record_completion()
            
            # Record token usage
            self._record_token_usage(response, task)
            
            logger.info(f"Task {task.id} completed successfully")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Task {task.id} failed: {error_msg}")
            
            if task.can_retry():
                task.reset_for_retry()
                logger.info(f"Retrying task {task.id} (attempt {task.retry_count}/{task.max_retries})")
                return self.execute_task(task, agent)
            else:
                task.mark_failed(error_msg)
                agent.record_failure()
        
        finally:
            agent.release_task()
            self.current_task_id = None
        
        return task
    
    def execute_plan(self, plan: DecompositionPlan) -> List[Task]:
        """
        Execute all subtasks in a decomposition plan.
        
        Respects dependencies and executes sequentially (for now).
        """
        logger.info(f"Executing plan with {len(plan.subtasks)} subtasks")
        
        completed_tasks = []
        failed_tasks = []
        
        # Simple sequential execution (P2 will add parallel DAG execution)
        for subtask in plan.subtasks:
            # Refresh task state from storage
            current_task = self.tasks.get(subtask.id)
            if not current_task:
                continue
            
            # Check if dependencies are met
            deps_met = all(
                self.tasks.get(dep_id, Task(id="unknown", title="", description="", status=TaskStatus.FAILED)).status == TaskStatus.COMPLETED
                for dep_id in current_task.dependencies
            )
            
            if not deps_met:
                logger.warning(f"Skipping {subtask.id}: dependencies not met")
                failed_tasks.append(subtask)
                continue
            
            # Execute the task
            result = self.execute_task(current_task)
            
            if result.status == TaskStatus.COMPLETED:
                completed_tasks.append(result)
            else:
                failed_tasks.append(result)
                if plan.strategy == "sequential":
                    logger.info("Sequential strategy: stopping on first failure")
                    break
        
        # Update original task
        original = self.tasks.get(plan.original_task.id)
        if original:
            if failed_tasks:
                original.mark_failed(f"{len(failed_tasks)} subtasks failed")
            else:
                original.mark_completed(f"All {len(completed_tasks)} subtasks completed")
        
        logger.info(f"Plan execution complete: {len(completed_tasks)} succeeded, {len(failed_tasks)} failed")
        
        return completed_tasks
    
    def run(self, task_description: str, decompose: bool = True, num_steps: int = 5) -> Task:
        """
        High-level method to run a task.
        
        Args:
            task_description: Natural language description of the task
            decompose: Whether to break into subtasks
            num_steps: Number of subtasks to create (if decomposing)
        
        Returns:
            The completed (or failed) task
        """
        logger.info(f"Running task: {task_description}")
        
        # Create main task
        task = self.create_task(
            title=task_description[:100],  # Truncate for title
            description=task_description,
            role=AgentRole.ANALYST if decompose else AgentRole.DEVELOPER
        )
        
        if decompose:
            # Decompose and execute subtasks
            plan = self.decompose_task(task, num_subtasks=num_steps)
            self.execute_plan(plan)
        else:
            # Execute directly
            self.execute_task(task)
        
        return task
    
    def _select_agent_for_task(self, task: Task) -> Optional[Agent]:
        """Select the best available agent for a task."""
        # Find agents with matching role
        candidates = [
            agent for agent in self.agents.values()
            if agent.role == task.role and agent.is_available
        ]
        
        if not candidates:
            # Fallback to any available agent
            candidates = [a for a in self.agents.values() if a.is_available]
        
        if not candidates:
            return None
        
        # Select agent with best success rate
        return max(candidates, key=lambda a: a.success_rate)
    
    def _build_system_prompt(self, agent: Agent) -> str:
        """Build system prompt for an agent based on role."""
        role_prompts = {
            AgentRole.ANALYST: """You are an expert analyst. Your job is to understand requirements, identify ambiguities, and create clear specifications.
Always think step-by-step and ask clarifying questions when needed.""",
            
            AgentRole.DEVELOPER: """You are an expert software developer. Write clean, efficient, well-documented code.
Follow best practices, include error handling, and write tests when appropriate.""",
            
            AgentRole.TESTER: """You are an expert QA engineer. Design comprehensive test cases and identify edge cases.
Focus on both positive and negative scenarios.""",
            
            AgentRole.DOCUMENTATION: """You are an expert technical writer. Create clear, concise documentation.
Include examples, explain concepts thoroughly, and organize information logically.""",
            
            AgentRole.REVIEWER: """You are an expert code reviewer. Identify bugs, security issues, and improvements.
Provide constructive feedback with specific suggestions.""",
            
            AgentRole.ORCHESTRATOR: """You are the coordinator. Manage tasks, delegate work, and ensure quality.""",
        }
        
        base_prompt = role_prompts.get(agent.role, "You are a helpful AI assistant.")
        
        if agent.system_prompt:
            base_prompt += f"\n\n{agent.system_prompt}"
        
        return base_prompt
    
    def _build_user_prompt(self, task: Task) -> str:
        """Build user prompt for a task."""
        prompt_parts = [
            f"Task: {task.title}",
            f"Description: {task.description}",
        ]
        
        if task.context:
            prompt_parts.append("\nContext:")
            for key, value in task.context.items():
                prompt_parts.append(f"- {key}: {value}")
        
        if task.files_involved:
            prompt_parts.append("\nRelevant files:")
            for file_path in task.files_involved:
                prompt_parts.append(f"- {file_path}")
        
        if task.parent_id:
            parent = self.tasks.get(task.parent_id)
            if parent:
                prompt_parts.append(f"\nParent task: {parent.title}")
        
        prompt_parts.append("\nPlease complete this task. Provide your solution with clear explanations.")
        
        return "\n".join(prompt_parts)
    
    def _record_token_usage(self, response: ModelResponse, task: Task):
        """Record token usage for cost tracking."""
        usage = TokenUsage(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=getattr(response, '_cost_usd', 0.0)
        )
        
        task.token_usage = task.token_usage.add(usage)
        self.total_tokens += response.total_tokens
        self.total_cost_usd += usage.cost_usd
        
        # Also update agent stats if we can find which agent was used
        # (This would need to be passed in or tracked differently)
        
        logger.debug(f"Recorded {response.total_tokens} tokens (${usage.cost_usd:.4f}) for task {task.id}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""
        task_counts = {}
        for task in self.tasks.values():
            status = task.status.value
            task_counts[status] = task_counts.get(status, 0) + 1
        
        return {
            'total_tasks': len(self.tasks),
            'task_status_counts': task_counts,
            'total_agents': len(self.agents),
            'available_agents': sum(1 for a in self.agents.values() if a.is_available),
            'total_cost_usd': self.total_cost_usd,
            'total_tokens': self.total_tokens,
            'current_task': self.current_task_id
        }
    
    def clear_cache(self):
        """Clear model provider caches."""
        for provider in self.model_manager.providers.values():
            provider._cache.clear()
        logger.info("Cache cleared")
