"""
Obsidian sync module for AutoDevAgent.
Saves tasks and artifacts as Markdown files with YAML frontmatter.
Creates bidirectional links between related items.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import yaml

from ..models.config import Config, get_config
from ..models.domain import Task, TaskStatus, TaskArtifact


logger = logging.getLogger(__name__)


class ObsidianSync:
    """
    Synchronizes tasks and artifacts to Obsidian-compatible Markdown files.
    
    Features:
    - YAML frontmatter with metadata
    - Bidirectional [[wikilinks]]
    - Tag support
    - Incremental updates
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        
        # Get directories from config
        sync_config = self.config.sync.obsidian
        self.graph_dir = Path(sync_config.graph_directory)
        self.decisions_dir = Path(sync_config.decisions_directory)
        
        # Initialize directories
        self._initialize()
        
        logger.info(f"ObsidianSync initialized: graph={self.graph_dir}, decisions={self.decisions_dir}")
    
    def _initialize(self):
        """Create necessary directories."""
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.decisions_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organization
        (self.graph_dir / "tasks").mkdir(exist_ok=True)
        (self.graph_dir / "artifacts").mkdir(exist_ok=True)
        (self.graph_dir / "agents").mkdir(exist_ok=True)
    
    def save_task(self, task: Task, update_links: bool = True) -> Path:
        """
        Save a task as a Markdown file with YAML frontmatter.
        
        Args:
            task: The task to save
            update_links: Whether to update bidirectional links
        
        Returns:
            Path to the saved file
        """
        # Generate filename from task ID and title
        safe_title = self._sanitize_filename(task.title)
        filename = f"{task.id}_{safe_title}.md"
        filepath = self.graph_dir / "tasks" / filename
        
        # Build frontmatter
        frontmatter = {
            'id': task.id,
            'title': task.title,
            'status': task.status.value,
            'role': task.role.value,
            'priority': task.priority.value,
            'created_at': task.created_at.isoformat(),
            'tags': self._build_tags(task),
        }
        
        # Add optional fields
        if task.parent_id:
            frontmatter['parent_id'] = task.parent_id
        
        if task.dependencies:
            frontmatter['dependencies'] = task.dependencies
        
        if task.token_usage.total_tokens > 0:
            frontmatter['tokens_used'] = task.token_usage.total_tokens
            frontmatter['cost_usd'] = round(task.token_usage.cost_usd, 4)
        
        if task.started_at:
            frontmatter['started_at'] = task.started_at.isoformat()
        
        if task.completed_at:
            frontmatter['completed_at'] = task.completed_at.isoformat()
            if task.duration:
                frontmatter['duration_seconds'] = round(task.duration, 2)
        
        if task.error_message:
            frontmatter['error'] = task.error_message
        
        if task.retry_count > 0:
            frontmatter['retry_count'] = task.retry_count
        
        # Build content
        content_parts = [
            "---",
            yaml.dump(frontmatter, sort_keys=False, allow_unicode=True).rstrip(),
            "---",
            ""
        ]
        
        # Add description
        content_parts.append(f"# {task.title}")
        content_parts.append("")
        content_parts.append(task.description)
        content_parts.append("")
        
        # Add status badge
        status_emoji = {
            TaskStatus.PENDING: "⏳",
            TaskStatus.IN_PROGRESS: "🔄",
            TaskStatus.COMPLETED: "✅",
            TaskStatus.FAILED: "❌",
            TaskStatus.BLOCKED: "🚫",
            TaskStatus.CANCELLED: "🛑"
        }.get(task.status, "📋")
        
        content_parts.append(f"**Status:** {status_emoji} {task.status.value.upper()}")
        content_parts.append("")
        
        # Add links to dependencies
        if task.dependencies and update_links:
            content_parts.append("## Dependencies")
            content_parts.append("")
            for dep_id in task.dependencies:
                dep_task = None  # Would need to look this up from storage
                if dep_task:
                    dep_link = self._create_wikilink(dep_task.title, dep_task.id)
                    content_parts.append(f"- {dep_link}")
                else:
                    content_parts.append(f"- `[[{dep_id}]]` (not found)")
            content_parts.append("")
        
        # Add links to child tasks
        if task.child_ids:
            content_parts.append("## Subtasks")
            content_parts.append("")
            for child_id in task.child_ids:
                # Would need to look up child task
                content_parts.append(f"- `[[{child_id}]]`")
            content_parts.append("")
        
        # Add result if completed
        if task.result:
            content_parts.append("## Result")
            content_parts.append("")
            content_parts.append(task.result)
            content_parts.append("")
        
        # Add artifacts
        if task.artifacts:
            content_parts.append("## Artifacts")
            content_parts.append("")
            for artifact in task.artifacts:
                if artifact.path:
                    artifact_link = self._create_wikilink(artifact.name, path=artifact.path)
                    content_parts.append(f"- {artifact_link}")
                else:
                    content_parts.append(f"- 📄 {artifact.name} ({artifact.type})")
            content_parts.append("")
        
        # Add metadata section
        content_parts.append("---")
        content_parts.append("*Generated by AutoDevAgent*")
        
        # Write file
        content = "\n".join(content_parts)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Update bidirectional links
        if update_links and self.config.sync.obsidian.auto_link:
            self._update_backlinks(task)
        
        logger.debug(f"Saved task {task.id} to {filepath}")
        
        return filepath
    
    def save_artifact(self, artifact: TaskArtifact, task: Optional[Task] = None) -> Path:
        """
        Save an artifact as a Markdown file.
        
        Args:
            artifact: The artifact to save
            task: Optional parent task for linking
        
        Returns:
            Path to the saved file
        """
        safe_name = self._sanitize_filename(artifact.name)
        filename = f"{artifact.created_at.strftime('%Y%m%d_%H%M%S')}_{safe_name}.md"
        filepath = self.graph_dir / "artifacts" / filename
        
        # Build frontmatter
        frontmatter = {
            'name': artifact.name,
            'type': artifact.type,
            'created_at': artifact.created_at.isoformat(),
            'tags': [f"artifact/{artifact.type}"],
        }
        
        if task:
            frontmatter['task_id'] = task.id
            frontmatter['task_title'] = task.title
        
        if artifact.path:
            frontmatter['source_path'] = artifact.path
        
        # Build content
        content_parts = [
            "---",
            yaml.dump(frontmatter, sort_keys=False, allow_unicode=True).rstrip(),
            "---",
            "",
            f"# {artifact.name}",
            "",
            f"*Type: {artifact.type}* | *Created: {artifact.created_at.strftime('%Y-%m-%d %H:%M')}*",
            ""
        ]
        
        if task:
            task_link = self._create_wikilink(task.title, task.id)
            content_parts.append(f"**Parent Task:** {task_link}")
            content_parts.append("")
        
        if artifact.content:
            content_parts.append("## Content")
            content_parts.append("")
            
            # If it looks like code, wrap in code block
            if artifact.type in ['code', 'python', 'javascript', 'test']:
                lang = artifact.type if artifact.type != 'code' else ''
                content_parts.append(f"```{lang}")
                content_parts.append(artifact.content)
                content_parts.append("```")
            else:
                content_parts.append(artifact.content)
            
            content_parts.append("")
        
        if artifact.path:
            content_parts.append(f"**Source:** `{artifact.path}`")
            content_parts.append("")
        
        content = "\n".join(content_parts)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.debug(f"Saved artifact {artifact.name} to {filepath}")
        
        return filepath
    
    def save_decision(self, decision_id: str, title: str, context: str, decision: str, alternatives: Optional[List[str]] = None) -> Path:
        """
        Save a decision record (Architecture Decision Record style).
        
        Args:
            decision_id: Unique identifier
            title: Decision title
            context: Context and problem statement
            decision: The decision made
            alternatives: Alternative options considered
        
        Returns:
            Path to the saved file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_title = self._sanitize_filename(title)
        filename = f"ADR_{timestamp}_{safe_title}.md"
        filepath = self.decisions_dir / filename
        
        # Build content
        content_parts = [
            "---",
            yaml.dump({
                'id': decision_id,
                'title': title,
                'date': datetime.now().isoformat(),
                'status': 'accepted',
                'tags': ['decision', 'adr']
            }, sort_keys=False).rstrip(),
            "---",
            "",
            f"# {title}",
            "",
            f"*Decision ID: {decision_id}* | *Date: {datetime.now().strftime('%Y-%m-%d')}*",
            "",
            "## Context",
            "",
            context,
            "",
            "## Decision",
            "",
            decision,
            ""
        ]
        
        if alternatives:
            content_parts.append("## Alternatives Considered")
            content_parts.append("")
            for alt in alternatives:
                content_parts.append(f"- {alt}")
            content_parts.append("")
        
        content_parts.append("## Consequences")
        content_parts.append("")
        content_parts.append("*To be documented as we implement this decision.*")
        content_parts.append("")
        content_parts.append("---")
        content_parts.append("*Generated by AutoDevAgent*")
        
        content = "\n".join(content_parts)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Saved decision {decision_id} to {filepath}")
        
        return filepath
    
    def _create_wikilink(self, title: str, id: Optional[str] = None, path: Optional[str] = None) -> str:
        """Create an Obsidian wikilink."""
        if path:
            # Link to external file
            return f"[{title}]({path})"
        elif id:
            # Include ID in link for disambiguation
            return f"[[{title}|{id}]]"
        else:
            return f"[[{title}]]"
    
    def _build_tags(self, task: Task) -> List[str]:
        """Build tag list for a task."""
        tags = []
        
        # Status tag
        tags.append(f"status/{task.status.value}")
        
        # Role tag
        tags.append(f"role/{task.role.value}")
        
        # Priority tag
        if task.priority.value in ['high', 'critical']:
            tags.append(f"priority/{task.priority.value}")
        
        # Custom tags
        tags.extend(task.tags)
        
        return tags
    
    def _update_backlinks(self, task: Task):
        """Update backlinks in related task files."""
        # This would scan other task files and add backlinks
        # For now, it's a placeholder for future implementation
        pass
    
    def _sanitize_filename(self, title: str, max_length: int = 50) -> str:
        """Sanitize a string for use in filenames."""
        # Remove or replace problematic characters
        sanitized = title.replace('/', '-').replace('\\', '-')
        sanitized = sanitized.replace(':', '-').replace('*', '-')
        sanitized = sanitized.replace('?', '-').replace('"', "'")
        sanitized = sanitized.replace('<', '(').replace('>', ')')
        sanitized = sanitized.replace('|', '-')
        
        # Truncate if too long
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        # Clean up whitespace
        sanitized = ' '.join(sanitized.split())
        
        return sanitized
    
    def get_task_file(self, task_id: str) -> Optional[Path]:
        """Find the file path for a task by ID."""
        # Search in tasks directory
        for filepath in (self.graph_dir / "tasks").glob(f"{task_id}_*.md"):
            return filepath
        return None
    
    def list_tasks(self) -> List[Dict[str, Any]]:
        """List all tasks in the graph."""
        tasks = []
        
        for filepath in (self.graph_dir / "tasks").glob("*.md"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse frontmatter
                if content.startswith('---'):
                    parts = content.split('---', 2)
                    if len(parts) >= 3:
                        frontmatter = yaml.safe_load(parts[1])
                        tasks.append({
                            'id': frontmatter.get('id', 'unknown'),
                            'title': frontmatter.get('title', 'Untitled'),
                            'status': frontmatter.get('status', 'unknown'),
                            'file': str(filepath)
                        })
            except Exception as e:
                logger.warning(f"Error reading task file {filepath}: {e}")
        
        return tasks
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the Obsidian graph."""
        task_files = list((self.graph_dir / "tasks").glob("*.md"))
        artifact_files = list((self.graph_dir / "artifacts").glob("*.md"))
        decision_files = list(self.decisions_dir.glob("*.md"))
        
        return {
            'total_tasks': len(task_files),
            'total_artifacts': len(artifact_files),
            'total_decisions': len(decision_files),
            'graph_directory': str(self.graph_dir),
            'decisions_directory': str(self.decisions_dir)
        }
