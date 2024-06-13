import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from io import BytesIO
import torch
import torchvision.transforms as transforms
from model import SimpleCNN

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN()
model.load_state_dict(torch.load('./models/cnn_cifar10.pth', map_location=device))
model.to(device)
model.eval()

# CIFAR-10 classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def preprocess_user_image(image_data):
    try:
        image = Image.open(BytesIO(image_data))
        image = transform(image).unsqueeze(0)  # Add batch dimension
        return image
    except Exception as e:
        messagebox.showerror("Error", f"Error preprocessing image: {str(e)}")
        return None

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
        messagebox.showerror("Error", f"Error predicting image: {str(e)}")
        return None

def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        with open(file_path, 'rb') as f:
            image_data = f.read()
        image = Image.open(BytesIO(image_data))
        image.thumbnail((256, 256))
        photo = ImageTk.PhotoImage(image)

        image_label.config(image=photo)
        image_label.image = photo

        predicted_class = predict_user_image(image_data, model)
        if predicted_class:
            result_label.config(text=f"Predicted class: {predicted_class}")

window = tk.Tk()
window.title("CIFAR-10 Image Classifier")
window.minsize(256,256)
window.maxsize(256,356)
window.eval('tk::PlaceWindow . center')

frame = tk.Frame(window)
frame.pack(pady=10)

btn = tk.Button(frame, text="Open Image", command=open_image)
btn.pack()

image_label = tk.Label(frame)
image_label.pack(pady=10)

result_label = tk.Label(frame, text="")
result_label.pack(pady=10)

window.mainloop()
