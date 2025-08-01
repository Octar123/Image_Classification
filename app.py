import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# --- Define the same CNN model structure ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# --- End of model structure ---

# Load the trained model
PATH = './cifar_net.pth'
model = SimpleCNN()
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval() # Set the model to evaluation mode

# Define the classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# --- Streamlit App ---
st.title("CIFAR-10 Image Classifier")
st.write("Upload an image, and the model will predict what it is.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image and make a prediction
    img_tensor = transform(image).unsqueeze(0) # Add a batch dimension
    outputs = model(img_tensor)
    _, predicted_index = torch.max(outputs, 1)

    prediction = classes[predicted_index.item()]

    st.write(f"**Prediction: {prediction}**")