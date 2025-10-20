from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from .schemas import PredictionResponse
from .model_service import model_service

app = FastAPI(
    title="Plant Disease Detection API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/", tags=["Health Check"])
async def root():
    return {"status": "ok", "message": "Plant Disease API is running."}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_disease(
    file: UploadFile = File(...)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type.")
    try:
        image_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=500, detail="Error reading file.")
    
    try:
        prediction_result = model_service.predict(image_bytes)
        return prediction_result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error.")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)