import mlflow
import os
import glob
import sys

# Add src to path
sys.path.append(os.getcwd())

mlflow.set_tracking_uri("file:./mlruns")
experiment = mlflow.get_experiment_by_name("fraud_detection_baseline")

if not experiment:
    print("Experiment not found!")
    sys.exit(1)

runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1)

if runs.empty:
    print("No runs found!")
    sys.exit(1)

run_id = runs.iloc[0].run_id
artifact_uri = runs.iloc[0].artifact_uri

print(f"Run ID: {run_id}")
print(f"Artifact URI: {artifact_uri}")

# Check local path matching the URI
if artifact_uri.startswith("file:///"):
    local_path = artifact_uri.replace("file:///", "").replace("/", os.sep)
    # Fix for windows drive letter potentially
    # If C:/, it might become C:\ on windows replace of / to \
    
    print(f"Checking path: {local_path}")
    if os.path.exists(local_path):
        print("Directory exists.")
        print("Contents:", os.listdir(local_path))
    else:
        print("Directory DOES NOT exist.")

# Glob check in mlruns
glob_path = f"mlruns/{experiment.experiment_id}/{run_id}/artifacts/*"
print(f"Globbing {glob_path}...")
found = glob.glob(glob_path)
print("Found via glob:", found)
