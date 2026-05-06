"""AutoDevAgent CLI and main entry point."""

import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='adev',
        description='AutoDevAgent - Autonomous Development System'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run a task')
    run_parser.add_argument('task', type=str, help='Task description')
    run_parser.add_argument('--no-decompose', action='store_true', help='Skip task decomposition')
    run_parser.add_argument('--steps', type=int, default=5, help='Number of subtasks to create')
    run_parser.add_argument('--model', type=str, help='Model to use')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    
    # Cache commands
    cache_parser = subparsers.add_parser('cache', help='Cache management')
    cache_subparsers = cache_parser.add_subparsers(dest='cache_command')
    
    cache_subparsers.add_parser('clear', help='Clear cache')
    cache_subparsers.add_parser('stats', help='Show cache statistics')
    cache_list = cache_subparsers.add_parser('list', help='List cache entries')
    cache_list.add_argument('--limit', type=int, default=20, help='Max entries to show')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_parser.add_argument('--show', action='store_true', help='Show current config')
    config_parser.add_argument('--init', action='store_true', help='Initialize config file')
    
    # Web UI command
    web_parser = subparsers.add_parser('web', help='Start Web UI')
    web_parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    web_parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    web_parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Handle commands
    if args.command == 'run':
        return cmd_run(args)
    elif args.command == 'status':
        return cmd_status(args)
    elif args.command == 'cache':
        return cmd_cache(args)
    elif args.command == 'config':
        return cmd_config(args)
    elif args.command == 'web':
        return cmd_web(args)
    else:
        parser.print_help()
        return 1


def cmd_run(args):
    """Execute the run command."""
    try:
        from .models.config import get_config
        from .orchestrator.core import Orchestrator
        
        print(f"🚀 AutoDevAgent starting task: {args.task[:100]}...")
        
        # Initialize orchestrator
        config = get_config()
        
        # Override model if specified
        if args.model:
            config.llm.default_model = args.model
        
        orchestrator = Orchestrator(config)
        
        # Run the task
        task = orchestrator.run(
            task_description=args.task,
            decompose=not args.no_decompose,
            num_steps=args.steps
        )
        
        # Print result
        print("\n" + "="*60)
        if task.status.value == 'completed':
            print("✅ Task completed successfully!")
        else:
            print(f"❌ Task failed: {task.error_message}")
        
        print(f"\n📊 Statistics:")
        print(f"   Tokens used: {task.token_usage.total_tokens}")
        print(f"   Cost: ${task.token_usage.cost_usd:.4f}")
        if task.duration:
            print(f"   Duration: {task.duration:.1f}s")
        
        if task.result:
            print(f"\n📝 Result:\n{task.result[:500]}{'...' if len(task.result) > 500 else ''}")
        
        return 0 if task.status.value == 'completed' else 1
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_status(args):
    """Execute the status command."""
    try:
        from .models.config import get_config
        from .orchestrator.core import Orchestrator
        
        config = get_config()
        orchestrator = Orchestrator(config)
        
        status = orchestrator.get_status()
        
        print("📊 AutoDevAgent Status")
        print("="*40)
        print(f"Total tasks: {status['total_tasks']}")
        print(f"Task breakdown:")
        for state, count in status['task_status_counts'].items():
            print(f"   {state}: {count}")
        print(f"\nAgents: {status['available_agents']}/{status['total_agents']} available")
        print(f"Total cost: ${status['total_cost_usd']:.4f}")
        print(f"Total tokens: {status['total_tokens']:,}")
        
        if status['current_task']:
            print(f"\nCurrent task: {status['current_task']}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1


def cmd_cache(args):
    """Execute cache commands."""
    try:
        from .models.config import get_config
        from .cache.simple import SimpleCache
        
        config = get_config()
        cache = SimpleCache(
            cache_dir=config.cache.directory,
            ttl_seconds=config.cache.ttl,
            max_size_mb=config.cache.max_size_mb
        )
        
        if args.cache_command == 'clear':
            cache.clear()
            print("✅ Cache cleared")
        
        elif args.cache_command == 'stats':
            stats = cache.stats()
            print("📊 Cache Statistics")
            print("="*40)
            print(f"Entries: {stats['entries']}")
            print(f"Size: {stats['size_mb']:.2f} MB / {stats['max_size_mb']} MB")
            print(f"Directory: {stats['directory']}")
        
        elif args.cache_command == 'list':
            entries = cache.list_entries()
            if not entries:
                print("Cache is empty")
            else:
                print(f"📋 Recent cache entries (showing {min(len(entries), args.limit)} of {len(entries)}):")
                print("="*60)
                for entry in entries[:args.limit]:
                    expired = " ⚠️ EXPIRED" if entry['expired'] else ""
                    print(f"{entry['created_at'][:19]} | {entry['model']:15} | {entry['tokens']:5} tokens | {entry['key']}{expired}")
        
        else:
            print("Use: adev cache [clear|stats|list]")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1


def cmd_config(args):
    """Execute config commands."""
    try:
        from .models.config import get_config, load_config, Config
        import yaml
        
        if args.show:
            config = get_config()
            print("Current configuration:")
            print("="*60)
            print(f"Default model: {config.llm.default_model}")
            print(f"Cache enabled: {config.cache.enabled}")
            print(f"Cache directory: {config.cache.directory}")
            print(f"Max cost per task: ${config.cost.max_cost_per_task}")
            print(f"Routing strategy: {config.cost.routing_strategy}")
            print(f"Log level: {config.logging.level}")
        
        elif args.init:
            config_path = Path("config.yaml")
            if config_path.exists():
                print(f"⚠️  Config file already exists: {config_path}")
                response = input("Overwrite? (y/N): ")
                if response.lower() != 'y':
                    return 0
            
            # Create default config
            config = Config()
            with open(config_path, 'w') as f:
                yaml.dump(config.model_dump(), f, default_flow_style=False)
            
            print(f"✅ Created config file: {config_path}")
        
        else:
            print("Use: adev config [--show|--init]")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1


def cmd_web(args):
    """Start the Web UI."""
    try:
        import uvicorn
        from .web.api import app
        
        print(f"🚀 Starting AutoDevAgent Web UI...")
        print(f"   Host: {args.host}")
        print(f"   Port: {args.port}")
        print(f"   Reload: {args.reload}")
        print(f"\n   Open http://localhost:{args.port} in your browser\n")
        print("="*60)
        
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info"
        )
        
        return 0
        
    except ImportError:
        print("❌ Web dependencies not installed.")
        print("   Install with: pip install 'autodev-agent[web]'")
        return 1
    except Exception as e:
        print(f"❌ Error starting web UI: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
