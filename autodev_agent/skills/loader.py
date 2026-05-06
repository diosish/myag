"""
Skills loader module for AutoDevAgent.
Loads skill definitions from YAML files and provides programmatic access.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

import yaml

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """Represents a skill with its configuration."""
    
    name: str
    description: str
    file_patterns: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    system_prompt_template: Optional[str] = None
    example_prompts: List[str] = field(default_factory=list)
    
    def matches_file(self, filepath: str) -> bool:
        """Check if this skill matches a given file path."""
        from fnmatch import fnmatch
        
        for pattern in self.file_patterns:
            if fnmatch(filepath, pattern):
                return True
        return False
    
    def build_system_prompt(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Build the system prompt for this skill."""
        if not self.system_prompt_template:
            return f"You are an expert with the {self.name} skill."
        
        # Simple template substitution (can be enhanced later)
        prompt = self.system_prompt_template
        if context:
            for key, value in context.items():
                prompt = prompt.replace(f"{{{{{key}}}}}", str(value))
        
        return prompt


class SkillLoader:
    """
    Loads and manages skills from YAML configuration files.
    
    Usage:
        loader = SkillLoader()
        python_skill = loader.get_skill('python')
        prompt = python_skill.build_system_prompt()
    """
    
    def __init__(self, skills_file: Optional[str] = None):
        """
        Initialize the skill loader.
        
        Args:
            skills_file: Path to the skills YAML file. If None, uses default location.
        """
        if skills_file is None:
            # Default location
            skills_file = Path(__file__).parent / "skills.yaml"
        
        self.skills_file = Path(skills_file)
        self._skills: Dict[str, Skill] = {}
        
        if self.skills_file.exists():
            self._load_skills()
        else:
            logger.warning(f"Skills file not found: {self.skills_file}")
    
    def _load_skills(self):
        """Load skills from the YAML file."""
        try:
            with open(self.skills_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if not data:
                logger.warning("Skills file is empty")
                return
            
            for skill_id, skill_data in data.items():
                skill = Skill(
                    name=skill_data.get('name', skill_id),
                    description=skill_data.get('description', ''),
                    file_patterns=skill_data.get('file_patterns', []),
                    tools=skill_data.get('tools', []),
                    system_prompt_template=skill_data.get('system_prompt_template'),
                    example_prompts=skill_data.get('example_prompts', [])
                )
                self._skills[skill_id] = skill
            
            logger.info(f"Loaded {len(self._skills)} skills from {self.skills_file}")
            
        except Exception as e:
            logger.error(f"Error loading skills: {e}")
            raise
    
    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """Get a skill by ID."""
        return self._skills.get(skill_id)
    
    def list_skills(self) -> List[str]:
        """List all available skill IDs."""
        return list(self._skills.keys())
    
    def get_skills_for_file(self, filepath: str) -> List[Skill]:
        """Get all skills that match a given file path."""
        matching = []
        for skill in self._skills.values():
            if skill.matches_file(filepath):
                matching.append(skill)
        return matching
    
    def get_skill_names(self) -> List[str]:
        """Get all skill names."""
        return [skill.name for skill in self._skills.values()]
    
    def reload(self):
        """Reload skills from file."""
        self._skills.clear()
        self._load_skills()


# Global instance for convenience
_default_loader: Optional[SkillLoader] = None


def get_skill_loader() -> SkillLoader:
    """Get the default skill loader instance."""
    global _default_loader
    if _default_loader is None:
        _default_loader = SkillLoader()
    return _default_loader


def get_skill(skill_id: str) -> Optional[Skill]:
    """Get a skill from the default loader."""
    return get_skill_loader().get_skill(skill_id)
