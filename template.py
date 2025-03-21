from pathlib import Path
import os
project_name  = "src"


list_of_files = [
    f"{project_name}/__init__.py",
    f"{project_name}/components/data_preprocessing.py",
    f"{project_name}/components/inference.py",
    f"{project_name}/components/model.py",
    f"{project_name}/components/train.py",
    f"{project_name}/components/api.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/data/raw",
    f"{project_name}/data/processed",
    f"{project_name}/data/dvc.yaml",
    f"{project_name}/data/__init__.py",
    f"{project_name}/tests/test_model.py",
    f"{project_name}/tests/test_data.py",
    f"{project_name}/tests/__init__.py",
    f"{project_name}/airflow/pipeline_dag.py",
    f"{project_name}/airflow/__init__.py",
    f"{project_name}/logger/__init__.py",
    ".github/workflows/ci-cd.yaml",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "dockerignore",
    "demo.py"

]

for file in list_of_files:
    file = Path(file)
    filedir, filename = os.path.split(file)
    if filedir:
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(file)) or (os.path.getsize(file) == 0):
        with open(file, "w") as f:
            f.write("")
    else:
        print(f"File {file} already exists")
        continue