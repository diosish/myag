"""Tests for skills loader."""

import pytest
import tempfile
from pathlib import Path
import yaml

from autodev_agent.skills.loader import SkillLoader, Skill


class TestSkill:
    """Tests for Skill dataclass."""
    
    def test_creation(self):
        """Test creating a skill."""
        skill = Skill(
            name="Python Development",
            description="Expert Python programming",
            file_patterns=["*.py", "requirements.txt"],
            tools=["pytest", "black", "mypy"],
            system_prompt_template="You are a Python expert..."
        )
        
        assert skill.name == "Python Development"
        assert len(skill.file_patterns) == 2
        assert len(skill.tools) == 3
    
    def test_matches_file(self):
        """Test file pattern matching."""
        skill = Skill(
            name="Python",
            description="Test",
            file_patterns=["*.py", "requirements.txt"]
        )
        
        assert skill.matches_file("main.py") is True
        assert skill.matches_file("test_utils.py") is True
        assert skill.matches_file("requirements.txt") is True
        assert skill.matches_file("README.md") is False
        assert skill.matches_file("src/app.js") is False
    
    def test_build_system_prompt_without_template(self):
        """Test building prompt without template."""
        skill = Skill(
            name="Generic Skill",
            description="Test"
        )
        
        prompt = skill.build_system_prompt()
        assert "Generic Skill" in prompt
    
    def test_build_system_prompt_with_template(self):
        """Test building prompt with template and context."""
        skill = Skill(
            name="Python",
            description="Test",
            system_prompt_template="You are {{expertise_level}} Python developer who knows {{frameworks}}"
        )
        
        prompt = skill.build_system_prompt({
            'expertise_level': 'an expert',
            'frameworks': 'Django and FastAPI'
        })
        
        assert "an expert" in prompt
        assert "Django and FastAPI" in prompt


class TestSkillLoader:
    """Tests for SkillLoader class."""
    
    @pytest.fixture
    def temp_skills_file(self):
        """Create temporary skills YAML file."""
        skills_data = {
            'python': {
                'name': 'Python Development',
                'description': 'Expert Python programming',
                'file_patterns': ['*.py'],
                'tools': ['pytest', 'black'],
                'system_prompt_template': 'You are a Python expert.'
            },
            'javascript': {
                'name': 'JavaScript Development',
                'description': 'Modern JS/TS development',
                'file_patterns': ['*.js', '*.ts'],
                'tools': ['eslint', 'prettier']
            }
        }
        
        tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(skills_data, tmpfile)
        tmpfile.close()
        
        yield tmpfile.name
        
        # Cleanup
        Path(tmpfile.name).unlink()
    
    def test_load_skills(self, temp_skills_file):
        """Test loading skills from file."""
        loader = SkillLoader(temp_skills_file)
        
        assert len(loader._skills) == 2
        assert 'python' in loader._skills
        assert 'javascript' in loader._skills
    
    def test_get_skill(self, temp_skills_file):
        """Test getting a skill by ID."""
        loader = SkillLoader(temp_skills_file)
        
        skill = loader.get_skill('python')
        assert skill is not None
        assert skill.name == 'Python Development'
        assert 'pytest' in skill.tools
        
        # Non-existent skill
        skill_none = loader.get_skill('nonexistent')
        assert skill_none is None
    
    def test_list_skills(self, temp_skills_file):
        """Test listing skill IDs."""
        loader = SkillLoader(temp_skills_file)
        
        skill_ids = loader.list_skills()
        assert 'python' in skill_ids
        assert 'javascript' in skill_ids
        assert len(skill_ids) == 2
    
    def test_get_skills_for_file(self, temp_skills_file):
        """Test getting skills that match a file."""
        loader = SkillLoader(temp_skills_file)
        
        # Python file
        python_skills = loader.get_skills_for_file('app.py')
        assert len(python_skills) == 1
        assert python_skills[0].name == 'Python Development'
        
        # TypeScript file
        ts_skills = loader.get_skills_for_file('component.ts')
        assert len(ts_skills) == 1
        assert ts_skills[0].name == 'JavaScript Development'
        
        # Markdown file (no match)
        md_skills = loader.get_skills_for_file('README.md')
        assert len(md_skills) == 0
    
    def test_reload(self, temp_skills_file):
        """Test reloading skills."""
        loader = SkillLoader(temp_skills_file)
        
        # Modify the file
        new_data = {
            'new_skill': {
                'name': 'New Skill',
                'description': 'A new skill'
            }
        }
        with open(temp_skills_file, 'w') as f:
            yaml.dump(new_data, f)
        
        # Reload
        loader.reload()
        
        assert len(loader._skills) == 1
        assert 'new_skill' in loader._skills
    
    def test_default_location_not_found(self, caplog):
        """Test behavior when default skills file doesn't exist."""
        # Point to non-existent file
        loader = SkillLoader('/nonexistent/path/skills.yaml')
        
        assert len(loader._skills) == 0
        assert "Skills file not found" in caplog.text
    
    def test_empty_file(self, caplog):
        """Test loading empty skills file."""
        tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        tmpfile.write("")  # Empty file
        tmpfile.close()
        
        loader = SkillLoader(tmpfile.name)
        
        assert len(loader._skills) == 0
        
        Path(tmpfile.name).unlink()
