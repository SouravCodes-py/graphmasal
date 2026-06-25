"""
Pytest configuration for pipeline tests.
Adds the pipeline directory to sys.path so that imports like
`from state import StudentState` and `from db.neo4j_helper import run_query`
resolve correctly — matching the same import style the agents use at runtime.
"""

import sys
import os

# Add pipeline/ to the front of sys.path
pipeline_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pipeline_dir not in sys.path:
    sys.path.insert(0, pipeline_dir)
