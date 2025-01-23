# from fastapi import FastAPI, UploadFile, File
# from typing import Optional
# from enum import Enum
# from http import HTTPStatus
# import cv2

# app = FastAPI()

# @app.get("/")
# def root():
#     return {"Hello": "World"}

# class ItemEnum(Enum):
#     alexnet = "alexnet"
#     resnet = "resnet"
#     lenet = "lenet"

# @app.get("/restric_items/{item_id}")
# def read_item(item_id: ItemEnum):
#     return {"item_id": item_id}

# @app.post("/cv_model/")
# async def cv_model(data: UploadFile = File(...)):
#     with open('image.jpg', 'wb') as image:
#         content = await data.read()
#         image.write(content)
#         image.close()
#     response = {
#         "input": data,
#         "message": HTTPStatus.OK.phrase,
#         "status-code": HTTPStatus.OK,
#     }
#     return response


from fastapi import FastAPI
import subprocess
import re

app = FastAPI()

@app.post("/evaluate/")
async def evaluate_model():
    result = subprocess.run(["python", "evaluate.py"], capture_output=True, text=True)
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