import torch
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
from model import SimpleCNN

# Define transformations for CIFAR-10 images
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN()
model.load_state_dict(torch.load('./models/cnn_cifar10.pth', map_location=device))
model.to(device)
model.eval()

# CIFAR-10 classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Function to preprocess a user-uploaded image
def preprocess_user_image(image_data):
    try:
        image = Image.open(BytesIO(image_data))
        image = transform(image).unsqueeze(0)  # Add batch dimension
        return image
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

# Function to make predictions
def predict_user_image(image_data, model):
    try:
        image_tensor = preprocess_user_image(image_data)
        if image_tensor is None:
            return None

        image_tensor = image_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs.data, 1)
            return classes[predicted.item()]
    except Exception as e:
        print(f"Error predicting image: {str(e)}")
        return None

# Example usage
# Assume you have received image_data (byte content of the image)
# Replace 'path_to_your_image.jpg' with the actual image path
with open('./tests/test1.jpg', 'rb') as f:
    image_data = f.read()

predicted_class = predict_user_image(image_data, model)
if predicted_class:
    print(f'Predicted class: {predicted_class}')
