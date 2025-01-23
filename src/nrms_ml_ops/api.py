from fastapi import FastAPI
import subprocess
import re

app = FastAPI()

@app.post("/evaluate/")
async def evaluate_model():
    result = subprocess.run(["python", "evaluate_cloud.py"], capture_output=True, text=True)
    # print("STDOUT:", result.stdout)
    # print("STDERR:", result.stderr)
    # Extract the last "Test AUC: " line with numbers from stdout
    matches = re.findall(r"Test AUC: \d+\.\d+", result.stdout)
    if matches:
        last_match = matches[-1]
    else:
        last_match = "No AUC found"
    if result.returncode == 0:
        return {"auc": last_match}
    else:
        return {"error": result.stderr.strip()}