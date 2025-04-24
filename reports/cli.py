#!/usr/bin/env python
"""
Command-line interface for Two-Tower model reports.

This module provides a CLI for creating W&B reports for Two-Tower model runs,
whether for individual runs or comparisons between multiple runs.

Usage:
    # Create a single-run report
    python -m reports.cli single --run-id RUN_ID

    # Create a comparison report for specific runs
    python -m reports.cli compare --run-ids RUN_ID1 RUN_ID2 RUN_ID3

    # Create a comparison report using 5 most recent runs
    python -m reports.cli compare
"""

import argparse
import logging
import sys
from typing import Optional, List

from .single_report import create_two_tower_report
from .compare_report import create_comparison_report

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('two_tower.report_cli')

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Create W&B reports for Two-Tower model experiments")
    
    # Common arguments
    parser.add_argument("--project", default="two-tower-retrieval", help="W&B project name")
    parser.add_argument("--entity", default=None, help="W&B entity (username or team name)")
    parser.add_argument("--title", default=None, help="Report title")
    parser.add_argument("--description", default=None, help="Report description")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Single report command
    single_parser = subparsers.add_parser("single", help="Create a report for a single run")
    single_parser.add_argument("--run-id", required=True, help="W&B run ID for the report")
    
    # Comparison report command
    compare_parser = subparsers.add_parser("compare", help="Create a comparison report for multiple runs")
    compare_parser.add_argument("--run-ids", nargs="+", help="List of W&B run IDs to include in comparison")
    compare_parser.add_argument("--num-recent", type=int, default=5, 
                              help="Number of recent runs to compare (if run-ids not specified)")
    
    return parser.parse_args()

def run_single_report(args) -> Optional[str]:
    """Run a single report command."""
    logger.info(f"Creating single report for run {args.run_id}")
    
    try:
        url = create_two_tower_report(
            project_name=args.project,
            entity=args.entity,
            title=args.title,
            description=args.description,
            run_id=args.run_id
        )
        
        if url:
            logger.info(f"Report created successfully at: {url}")
            print(f"Report created successfully at: {url}")
            return url
        else:
            logger.error("Failed to create report")
            print("Failed to create report. See logs for details.")
            return None
    except Exception as e:
        logger.error(f"Error creating single report: {str(e)}")
        print(f"Error: {str(e)}")
        return None

def run_comparison_report(args) -> Optional[str]:
    """Run a comparison report command."""
    run_ids = args.run_ids
    
    if not run_ids:
        logger.info(f"No run IDs provided, using {args.num_recent} most recent runs")
    else:
        logger.info(f"Creating comparison report for {len(run_ids)} runs")
    
    try:
        url = create_comparison_report(
            project_name=args.project,
            entity=args.entity,
            title=args.title,
            description=args.description,
            run_ids=run_ids
        )
        
        if url:
            logger.info(f"Comparison report created successfully at: {url}")
            print(f"Comparison report created successfully at: {url}")
            return url
        else:
            logger.error("Failed to create comparison report")
            print("Failed to create comparison report. See logs for details.")
            return None
    except Exception as e:
        logger.error(f"Error creating comparison report: {str(e)}")
        print(f"Error: {str(e)}")
        return None

def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    if args.command == "single":
        run_single_report(args)
    elif args.command == "compare":
        run_comparison_report(args)
    else:
        print("Please specify a command: 'single' or 'compare'")
        sys.exit(1)

if __name__ == "__main__":
    main() 