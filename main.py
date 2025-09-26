from typing import Union
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import RedirectResponse
from PIL import Image
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize, ToTensor
import uvicorn
from medapp.neural_net import Net
from pathlib import Path
app = FastAPI()
data_file = Path("data/final_model.pt")
# Define transforms (copied from task.py)
pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# PathMNIST class labels (9 classes)
CLASS_LABELS = [
    "adipose", "background", "debris", "lymphocytes", "mucus", 
    "smooth muscle", "normal colon mucosa", "cancer-associated stroma", "colorectal adenocarcinoma epithelium"
]

# Load the model
def load_model():
    """Load the trained model from final_model.pt"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net(num_classes=9)  # PathMNIST has 9 classes
    model.load_state_dict(torch.load(data_file, map_location=device))
    model.to(device)
    model.eval()
    return model, device

# Load model at startup
model, device = load_model()

@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    # Read the uploaded file
    contents = await file.read()
    
    # Open the image using PIL
    image = Image.open(io.BytesIO(contents))
    
    # Get image dimensions
    width, height = image.size
    
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

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Predict the class of an uploaded medical image"""
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Open the image using PIL
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        input_tensor = pytorch_transforms(image).unsqueeze(0)  # Add batch dimension
        input_tensor = input_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Get class label
        predicted_label = CLASS_LABELS[predicted_class]
        
        # Get all class probabilities
        all_probabilities = {
            CLASS_LABELS[i]: probabilities[0][i].item() 
            for i in range(len(CLASS_LABELS))
        }
        
        return {
            "filename": file.filename,
            "predicted_class": predicted_class,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "all_probabilities": all_probabilities,
            "image_size": {
                "width": image.size[0],
                "height": image.size[1]
            }
        }
        
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
