from fastapi import FastAPI
import subprocess
import re

app = FastAPI()


@app.post("/evaluate/")
async def evaluate_model():
    try:
        result = subprocess.run(
            ["python", "evaluate_cloud.py"],
            capture_output=True,
            text=True,
            check=True,  # Levanta un error si el script falla
        )
        matches = re.findall(r"Test AUC: \d+\.\d+", result.stdout)
        last_match = matches[-1] if matches else "No AUC found"
        return {"auc": last_match}
    except subprocess.CalledProcessError as e:
        return {"error": f"Subprocess failed: {e.stderr.strip()}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}
