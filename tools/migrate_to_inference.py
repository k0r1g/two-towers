#!/usr/bin/env python
"""
Migration script to help transition from the old structure to the new inference structure.

This script will:
1. Check for imports using the old structure and suggest updates
2. Output warnings for deprecated modules
3. Help migrate configuration files
"""

import os
import re
import argparse
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger('migrate')

# Patterns to search for
OLD_IMPORTS = {
    r'from twotower\.glove_search import GloVeSearch': 'from inference.search import GloVeSearch',
    r'from twotower\.search import SemanticSearch': 'from inference.search import TwoTowerSearch',
    r'from twotower import .*GloVeSearch': 'from inference.search import GloVeSearch',
    r'from twotower import .*SemanticSearch': 'from inference.search import TwoTowerSearch',
}

def scan_file(filepath):
    """Scan a file for old imports and suggest replacements."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        logger.warning(f"Skipping {filepath}: Unable to decode as UTF-8")
        return False
    
    changes_needed = False
    
    for old_pattern, new_import in OLD_IMPORTS.items():
        if re.search(old_pattern, content):
            logger.warning(f"Found old import pattern in {filepath}: {old_pattern}")
            logger.info(f"  Suggested replacement: {new_import}")
            changes_needed = True
    
    # Check for retrieve.py usage
    if 'retrieve.py' in content:
        logger.warning(f"Found reference to retrieve.py in {filepath}")
        logger.info("  Suggested replacement: Use the new CLI: python -m inference.cli.retrieve")
        changes_needed = True
    
    return changes_needed

def scan_directory(directory):
    """Recursively scan a directory for Python files."""
    logger.info(f"Scanning directory: {directory}")
    
    py_files = list(Path(directory).glob('**/*.py'))
    changes_needed = False
    
    for py_file in py_files:
        try:
            if scan_file(py_file):
                changes_needed = True
        except Exception as e:
            logger.error(f"Error scanning {py_file}: {e}")
    
    if not changes_needed:
        logger.info("No migration issues found!")
    
    return changes_needed

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Migration helper for the new inference structure')
    parser.add_argument('--directory', '-d', default='.', help='Directory to scan')
    args = parser.parse_args()
    
    logger.info("Migration Helper for Two-Tower Inference Structure")
    logger.info("=" * 60)
    
    # Print explanation of changes
    logger.info("The following changes have been made to the codebase:")
    logger.info("1. GloVeSearch moved from twotower.glove_search to inference.search.GloVeSearch")
    logger.info("2. SemanticSearch moved from twotower.search to inference.search.TwoTowerSearch")
    logger.info("3. retrieve.py functionality moved to inference.cli.retrieve")
    logger.info("4. Added a new evaluation module at twotower.evaluate")
    logger.info("=" * 60)
    
    # Scan directory
    changes_needed = scan_directory(args.directory)
    
    # Print summary
    if changes_needed:
        logger.info("=" * 60)
        logger.info("Migration needed. See warnings above for details.")
        logger.info("Consider updating your imports to the new structure.")
        logger.info("See inference/README.md for more information on the new structure.")
        return 1
    else:
        logger.info("=" * 60)
        logger.info("No migration issues found!")
        return 0

if __name__ == "__main__":
    exit(main()) 