#!/usr/bin/env python3
"""
Utility script to manage artifact directories.

Helps keep the artifacts directory clean by removing old simulation results.
"""

import argparse
import shutil
from datetime import datetime, timedelta
from pathlib import Path


def list_artifacts(artifacts_dir: Path = Path("artifacts")) -> list[Path]:
    """List all artifact directories with their creation times."""
    if not artifacts_dir.exists():
        return []
    
    artifacts = []
    for path in artifacts_dir.iterdir():
        if path.is_dir() and path.name.startswith("202"):  # Date-based naming
            artifacts.append(path)
    
    return sorted(artifacts, key=lambda p: p.name)


def clean_old_artifacts(max_age_days: int = 7, dry_run: bool = True):
    """Remove artifact directories older than max_age_days."""
    artifacts_dir = Path("artifacts")
    
    if not artifacts_dir.exists():
        print("No artifacts directory found.")
        return
    
    cutoff_date = datetime.now() - timedelta(days=max_age_days)
    cutoff_str = cutoff_date.strftime("%Y%m%d")
    
    artifacts = list_artifacts(artifacts_dir)
    old_artifacts = [a for a in artifacts if a.name < cutoff_str]
    
    if not old_artifacts:
        print(f"No artifacts older than {max_age_days} days found.")
        return
    
    print(f"Found {len(old_artifacts)} artifacts older than {max_age_days} days:")
    total_size = 0
    
    for artifact_path in old_artifacts:
        size = sum(f.stat().st_size for f in artifact_path.rglob('*') if f.is_file())
        total_size += size
        size_mb = size / (1024 * 1024)
        print(f"  {artifact_path.name} ({size_mb:.1f} MB)")
    
    print(f"Total size: {total_size / (1024 * 1024):.1f} MB")
    
    if dry_run:
        print("\n[DRY RUN] Use --delete to actually remove these directories.")
    else:
        confirm = input(f"\nDelete {len(old_artifacts)} artifact directories? [y/N]: ")
        if confirm.lower() == 'y':
            for artifact_path in old_artifacts:
                shutil.rmtree(artifact_path)
                print(f"Removed {artifact_path.name}")
            print(f"Cleaned up {total_size / (1024 * 1024):.1f} MB")
        else:
            print("Cancelled.")


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(description="Manage simulation artifacts")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    subparsers.add_parser('list', help='List all artifact directories')
    
    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean old artifact directories')
    clean_parser.add_argument('--max-age', type=int, default=7, 
                             help='Maximum age in days (default: 7)')
    clean_parser.add_argument('--delete', action='store_true',
                             help='Actually delete directories (default: dry run)')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        artifacts = list_artifacts()
        if not artifacts:
            print("No artifact directories found.")
        else:
            print(f"Found {len(artifacts)} artifact directories:")
            for artifact_path in artifacts:
                size = sum(f.stat().st_size for f in artifact_path.rglob('*') if f.is_file())
                size_mb = size / (1024 * 1024)
                print(f"  {artifact_path.name} ({size_mb:.1f} MB)")
    
    elif args.command == 'clean':
        clean_old_artifacts(max_age_days=args.max_age, dry_run=not args.delete)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
