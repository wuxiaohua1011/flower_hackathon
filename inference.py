import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor
from medapp.neural_net import Net
from pathlib import Path

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

def load_and_preprocess_image(image_path: str):
    """Load and preprocess the image for inference"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
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

def main():
    # Paths
    model_path = "data/final_model.pt"
    image_path = "data/test/class_0/test_image_00003.png"
    
    print(f"Loading model from: {model_path}")
    print(f"Loading image from: {image_path}")
    
    # Load model
    model, device = load_model(model_path)
    print(f"Model loaded on device: {device}")
    
    # Load and preprocess image
    input_tensor = load_and_preprocess_image(image_path)
    print(f"Image loaded and preprocessed. Shape: {input_tensor.shape}")
    
    # Run inference
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

if __name__ == "__main__":
    main()
