import os

import numpy as np
import onnxruntime as rt
import pandas as pd
from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse

from src.exam_project.data import find_sparse_rows_by_initials, load_and_preprocess_data

app = FastAPI()


@app.get("/predict/{initials}")
def predict(model_path: str = "models/model.onnx", initials: str = "APPL"):
    try:
        print("Current Working Directory:", os.getcwd())
        column_transformer, _, _, X_test, _, _, y_test, _ = load_and_preprocess_data()
        X, y = find_sparse_rows_by_initials(
            X_test, y_test, column_transformer, initials
        )

        ort_model = rt.InferenceSession(model_path)
        input_names = [i.name for i in ort_model.get_inputs()]
        output_names = [i.name for i in ort_model.get_outputs()]
        batch = {input_names[0]: X}
        out = ort_model.run(output_names, batch)

        labels = pd.read_csv("data/processed/sector_names.csv")
        labels = labels.to_numpy().flatten()
        prediction = labels[np.argmax(out)]
        correct = labels[y]

        print("Correct class:", correct)
        print("Predicted class:", prediction)

        return {
            "prediction": prediction,
            "correct": correct,
        }
    except ValueError as e:
        # Extract and format the available initials from the error message
        available_initials = (
            str(e).split("Available initials to choose from:\n")[1].split(", ")
        )

        # Return a user-friendly JSON response
        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid initials provided.",
                "message": "The provided initials were not found in the company list.",
                "available_initials": available_initials,
            },
        )
    except Exception as e:
        # Handle other exceptions gracefully
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Sector Prediction API!"
        """Use the endpoint /predict/{initials} to predict
        the sector of a company based on its stock symbol initials."""
    }
