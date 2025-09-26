from typing import Union
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import RedirectResponse
import cv2
import numpy as np
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize, ToTensor
import uvicorn
from medapp.neural_net import Net
from pathlib import Path
import logging
from PIL import Image
import pydantic 
from pydantic import BaseModel
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

class Prediction(BaseModel):
    class_index: int
    class_name: str
    confidence: float

class PredictionResponse(BaseModel):
    predictions: list[Prediction]
    predicted_class_index: int
    
app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
data_file = Path("data/final_model.pt")
# Define transforms (copied from task.py)
pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Dermamnist class labels (7 classes)
CLASS_LABELS = [
    "melanoma", 
    "melanocytic nevi", 
    "benign keratosis-like lesions", 
    "basal cell carcinoma", 
    "actinic keratoses",
    "vascular lesions", 
    "dermatofibroma"
]

# Define transforms (same as in task.py)
pytorch_transforms = Compose([
    ToTensor(), 
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def load_model(model_path: str):
    """Load the trained model from final_model.pt"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net(num_classes=7)  # Dermamnist has 7 classes
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    # Read the uploaded file
    contents = await file.read()
    
    # Open the image using OpenCV
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Print image size (as requested)
    print(f"Image size: {width} x {height} pixels")
    
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "image_size": {
            "width": width,
            "height": height,
            "total_pixels": width * height
        }
    }


def load_and_preprocess_image(image: Image):
    """Load and preprocess the image for inference"""
    # Load image
    image = image.convert('RGB')
    
    # Resize to 64x64 (to match model architecture)
    image = image.resize((64, 64))
    
    # Apply transforms
    input_tensor = pytorch_transforms(image).unsqueeze(0)  # Add batch dimension
    
    return input_tensor

def predict_image(model, input_tensor, device):
    """Run inference on the preprocessed image"""
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence, probabilities[0]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict the class of an uploaded medical image"""
    try:
        # Read the uploaded file
        contents = await file.read()
        model_path = "data/final_model.pt"
        
        image = Image.open(io.BytesIO(contents))
        
        # Load model
        model, device = load_model(model_path)
        print(f"Model loaded on device: {device}")
        
        # Load and preprocess image
        input_tensor = load_and_preprocess_image(image)
        print(f"Image loaded and preprocessed. Shape: {input_tensor.shape}")
        predicted_class, confidence, all_probabilities = predict_image(model, input_tensor, device)
        
        # Display results
        print("\n" + "="*50)
        print("INFERENCE RESULTS")
        print("="*50)
        print(f"Predicted Class: {predicted_class}")
        print(f"Predicted Label: {CLASS_LABELS[predicted_class]}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        
        print("\nAll Class Probabilities:")
        print("-" * 30)
        for i, prob in enumerate(all_probabilities):
            print(f"{i}: {CLASS_LABELS[i]:<30} {prob:.4f} ({prob*100:.2f}%)")
        
        print("\nNote: This image is from class_0, so the true label is:", CLASS_LABELS[0])
        predictions = []
        for i, prob in enumerate(all_probabilities):
            predictions.append(Prediction(class_index=i, class_name=CLASS_LABELS[i], confidence=prob))
        result = PredictionResponse(
            predictions=predictions,
            predicted_class_index=predicted_class
        )
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
