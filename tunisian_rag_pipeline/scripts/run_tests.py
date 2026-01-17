#!/usr/bin/env python3
"""
Run All Tests Script
====================
Convenience script to run all tests with proper configuration.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --verbose    # Run with verbose output
    python run_tests.py --slow       # Include slow tests
"""

import os
import sys
import subprocess
from pathlib import Path


def main():
    """Run the test suite."""
    project_root = Path(__file__).parent.parent
    tests_dir = project_root / "tests"
    
    # Change to project root
    os.chdir(project_root)
    
    # Build pytest command
    cmd = [sys.executable, "-m", "pytest", str(tests_dir)]
    
    # Add arguments
    if "--verbose" in sys.argv or "-v" in sys.argv:
        cmd.append("-v")
    
    if "--slow" in sys.argv:
        cmd.append("--runslow")
    
    if "--integration" in sys.argv:
        cmd.append("--runintegration")
    
    # Add coverage if available
    try:
        import pytest_cov
        cmd.extend(["--cov=src", "--cov-report=term-missing"])
    except ImportError:
        pass
    
    # Run tests
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)
    
    result = subprocess.run(cmd)
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
