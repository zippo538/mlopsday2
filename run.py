import os
import sys
from pathlib import Path

# Add src directory to path
src_path = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, src_path)

from run_pipeline import run_pipeline, create_directories

if __name__ == "__main__":
    create_directories()
    success = run_pipeline()
    
    if success:
        print("Pipeline completed successfully")
    else:
        print("Pipeline failed")