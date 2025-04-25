#!/usr/bin/env python
"""
Test runner for the Two-Tower model tests.
"""

import unittest
import sys
import os

# Add the parent directory to the path to import modules correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    # Discover and run all tests
    test_suite = unittest.defaultTestLoader.discover('tests', pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    
    # Return non-zero exit code if any tests failed
    sys.exit(0 if result.wasSuccessful() else 1) 