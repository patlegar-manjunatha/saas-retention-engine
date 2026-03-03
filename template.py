import os
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    ".github/workflows/ci.yaml",
    "config/training.yaml",
    "config/serving.yaml",
    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_transformation.py",
    "src/components/model_training.py",
    "src/components/model_evaluation.py",
    "src/helpers/__init__.py",
    "src/helpers/load_save.py",
    "src/helpers/initiate_stage.py",
    "src/helpers/exception_handling.py",
    "app/__init__.py",
    "app/main.py",
    "tests/__init__.py",
    "tests/test_api.py",
    "Dockerfile",
    "requirements.txt"
]

empty_directories = [
    "data",
    "artifacts"
]

# Create the files and their parent directories
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            
            # Auto-fill the dummy test for CI/CD
            if filename == "test_api.py":
                f.write("def test_dummy():\n    assert 1 == 1\n")
            
            # Auto-fill basic requirements
            elif filename == "requirements.txt":
                f.write("pandas\nscikit-learn\nxgboost\nmlflow\nfastapi\nuvicorn\npytest\n")
            
            else:
                pass # Create empty file
                
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")

# Create the empty directories
for directory in empty_directories:
    os.makedirs(directory, exist_ok=True)
    logging.info(f"Creating empty directory: {directory}")

logging.info("SUCCESS: MLOps Architecture generated perfectly.")
