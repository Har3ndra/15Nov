#Downloading and Preparing Dataset
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Hardcoded absolute or relative path to your dataset folder
data_dir = r"C:\Users\HARENDRA SINGH\OneDrive\Desktop\DemographAI\UTKFace"

ethnicity_labels = ['White', 'Black', 'Asian', 'Indian', 'Others']  # UTKFace labels

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
            # Ethnicity label is the second part of the filename (adjusted for some files)
            parts = img_name.split('_')
            if len(parts) < 2:
                raise ValueError("Invalid filename format")
            label = int(parts[1])
            if label < 0 or label >= len(ethnicity_labels):
                raise ValueError("Invalid label")
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_name}: {e}")
            # Return a dummy or skip; for simplicity, raise to skip
            raise

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

try:
    dataset = FaceDataset(data_dir, transform=transform)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
except Exception as e:
    print(f"Error setting up dataset: {e}")
    train_loader = None


#Defining CNN Model
import torch.nn as nn

class EthnicityCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(EthnicityCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = EthnicityCNN(num_classes=len(ethnicity_labels))

#Training Loop
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 30
if train_loader is not None:
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

else:
    print("Training skipped due to dataset error.")

# Save trained model
torch.save(model.state_dict(), "ethnicity_cnn.pth")

#Taking Image
import cv2
from deepface import DeepFace
from tkinter import Tk, filedialog

def predict_ethnicity_img(image_path):
    try:
        result = DeepFace.analyze(img_path=image_path, actions=['race'])
        race_predictions = result['race']
        ethnicity = max(race_predictions, key=race_predictions.get)
        confidence = race_predictions[ethnicity]
        print(f"Ethnicity: {ethnicity} ({confidence:.2f}%)")
    except Exception as e:
        print("Error or No face detected in image.")

def predict_ethnicity_camera():
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive camera frame. Exiting ...")
            break

        try:
            result = DeepFace.analyze(frame, actions=['race'], enforce_detection=False)
            if isinstance(result, list):
                result = result[0]
            race_predictions = result['race']
            ethnicity = max(race_predictions, key=race_predictions.get)
            confidence = race_predictions[ethnicity]
            msg = f"Ethnicity: {ethnicity} ({confidence:.2f}%)"
        except Exception:
            msg = "No face detected"

        cv2.putText(frame, msg, (30, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Ethnicity Prediction', frame)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):  # ESC or q to quit
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    print("Choose an option:")
    print("1. Use Camera")
    print("2. Upload a Photo")
    choice = input("Enter 1 or 2: ")

    if choice == "1":
        predict_ethnicity_camera()
    elif choice == "2":
        # File dialog for photo upload
        Tk().withdraw()  # we don't want a full GUI, so keep the root window hidden
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            predict_ethnicity_img(file_path)
        else:
            print("No image selected.")
    else:
        print("Invalid choice. Please enter either 1 or 2.")

if __name__ == "__main__":
    main()

#Alternate
from deepface import DeepFace
result = DeepFace.analyze(img_path="path/to/test_image.jpg", actions=['race'])
print(result['race'])
