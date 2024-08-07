import torch
from torchvision import transforms
from PIL import Image
from model import VIT
from Utils.config import parse_args
import tkinter as tk
from tkinter import filedialog
from torchvision import datasets

# Load configuration
args = parse_args()

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VIT.VIT()  # Initialize your model architecture
model.load_state_dict(torch.load('ViT_Model/model_weight_MNIST.pt', map_location=device))
model.to(device)
model.eval()


# Define image transformation
transform = transforms.Compose([
    transforms.Resize((args.im_s, args.im_s)),
    transforms.ToTensor()
])

# Function to get class names
def get_class_names(dataset_path):
    dataset = datasets.ImageFolder(dataset_path)
    return dataset.classes

# Define paths and get class names
train_data_path = args.train_data
class_names = get_class_names(train_data_path)  # Get class names from the dataset

def classify_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, dim=1)
    
    return class_names[predicted_class.item()]

def choose_file():
    # Create a Tk root widget (hidden)
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    # Open file dialog to choose an image file
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    return file_path

# Main function to run the script
if __name__ == "__main__":
    image_path = choose_file()
    if image_path:
        class_name = classify_image(image_path)
        print(f'Predicted class: {class_name}')
    else:
        print("No file selected")
