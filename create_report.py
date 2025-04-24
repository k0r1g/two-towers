#!/usr/bin/env python
"""
Create W&B reports for Two-Tower model runs.

This is a transition wrapper that forwards to the new modular reports package.
For more details, run:
    python -m reports.cli --help
"""

# Import the main entry point from the reports package
from reports.cli import main

if __name__ == "__main__":
    # Forward control to the main function in reports.cli
    main() 