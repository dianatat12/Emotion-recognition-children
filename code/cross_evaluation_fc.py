import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from facenet_pytorch import MTCNN
from hsemotion.facial_emotions import HSEmotionRecognizer
from torch.nn.functional import one_hot


# Initialize wandb
wandb.init(project="cross_dataset_app2five_fc_girls", entity="dtat-universitat-oberta-de-catalunya")

# Define the dataset class
class EmotionDataset(Dataset):
    def __init__(self, images, labels, transform=None, num_classes=3):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.images)  # Returns the number of samples in the dataset

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = np.array(image, dtype=np.float32)
        image = np.transpose(image, (1, 2, 0))  # Convert to HWC
        if self.transform:
            image = self.transform(image)

        # Assuming labels are one-hot encoded to match (batch_size, num_classes)
        label = torch.tensor(label, dtype=torch.float32)
        return image, label



# Function to detect face in a frame
def detect_face(frame, mtcnn):
    bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)
    if bounding_boxes is None:
        return []
    else:
        bounding_boxes = bounding_boxes[probs > 0.9]
        return bounding_boxes


# Function to process frames and extract face images
def process_frames_in_subfolder(subfolder_path, label, mtcnn):
    subfolder_results = []
    for root, _, files in os.walk(subfolder_path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                frame_path = os.path.join(root, file)
                frame_bgr = cv2.imread(frame_path)
                if frame_bgr is None:
                    continue
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                bounding_boxes = detect_face(frame_rgb, mtcnn)
                for bbox in bounding_boxes:
                    x1, y1, x2, y2 = bbox.astype(int)
                    face_img = frame_rgb[y1:y2, x1:x2, :]
                    if face_img.size == 0:
                        continue
                    face_img = cv2.resize(face_img, (224, 224))
                    face_img = np.transpose(face_img, (2, 0, 1))
                    face_img = face_img / 255.0
                    subfolder_results.append((face_img, label))
    return subfolder_results


# Function to read and process the labels
def read_and_process_labels(file_path, emotions_list):
    import pandas as pd
    df = pd.read_excel(file_path)
    df.columns = ['Video Name'] + emotions_list
    df = df.set_index('Video Name')
    emotion_data = df.applymap(lambda x: 1 if x in [1, 2] else 0)
    return emotion_data


# Function to find the matching video name from the subfolder
def find_matching_video(subfolder_name, labels_df):
    base_name = subfolder_name.rsplit('.', 1)[0].lower()
    for video_name in labels_df.index:
        if base_name in video_name.lower():
            return video_name
    return None


# Function to load and process the dataset
def load_dataset(directory, labels_path, emotions_list, mtcnn):
    labels_df = read_and_process_labels(labels_path, emotions_list)
    images, labels = [], []

    for subfolder_name in os.listdir(directory):
        subfolder_path = os.path.join(directory, subfolder_name)
        if os.path.isdir(subfolder_path):
            matching_video = find_matching_video(subfolder_name, labels_df)
            if matching_video:
                label = labels_df.loc[matching_video].values
                frame_results = process_frames_in_subfolder(subfolder_path, label, mtcnn)
                for face_img, lbl in frame_results:
                    images.append(face_img)
                    labels.append(lbl)

    return np.array(images), np.array(labels)


# Function to load the model and weights
def load_model(model_name, num_classes, device, weights_path):
    model = HSEmotionRecognizer(model_name=model_name, device=device).model
    
    # Define the classifier with the correct structure
    model.classifier = torch.nn.Sequential(
        torch.nn.BatchNorm1d(1280),
        torch.nn.Linear(1280, 640),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),  # Assuming Dropout layer is used
        torch.nn.BatchNorm1d(640),
        torch.nn.Linear(640, num_classes),
        torch.nn.Softmax(dim=1)  # Softmax for multi-class classification
    )
    
    # Load the model state dictionary with strict=False to avoid mismatch issues
    model.load_state_dict(torch.load(weights_path), strict=False)
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model


import torch.nn.functional as F

def evaluate_model(model, dataloader, emotions_list, device):
    all_preds = []
    all_labels = []
    total_loss = 0  # To accumulate loss
    criterion = torch.nn.CrossEntropyLoss()  # Use your training loss here

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = torch.softmax(outputs, dim=1)
            
            # Compute loss for the current batch
            batch_loss = criterion(outputs, labels.argmax(dim=1))  # Assuming labels are one-hot encoded
            total_loss += batch_loss.item() * inputs.size(0)  # Accumulate loss

            # Collect predictions and labels
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Calculate average test loss
    avg_loss = total_loss / len(dataloader.dataset)
    
    # Convert lists to arrays for evaluation
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    # Convert one-hot to single-class labels
    y_true_single_class = np.argmax(all_labels, axis=1)
    y_pred_single_class = np.argmax(all_preds, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_true_single_class, y_pred_single_class)
    f1 = f1_score(y_true_single_class, y_pred_single_class, average='weighted')
    auc = roc_auc_score(
        all_labels, all_preds, average='weighted', multi_class='ovr'
    )

    # Print test metrics
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test AUC: {auc:.4f}")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'enet_b0_8_best_afew'
    num_classes = 3
    emotions_list = ['Happiness', 'Surprise', 'Frustration']

    # Path to the saved model weights
    weights_path = '/home/diana/Documents/pth_reselection/emoreact_boys_scratch.pth'

    # Load the model with saved weights
    print("Loading model...")
    model = load_model(model_name, num_classes, device, 
                       
                       weights_path)

    # Define paths to validation data
    val_directory = '/home/diana/Documents/App2Five_Reselection/boys/frames'
    val_labels_path = '/home/diana/Documents/App2Five_Reselection/boys/app2five_boys_labels.xlsx'

    # Load validation dataset
    print("Loading validation dataset...")
    mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=device)
    X_val, y_val = load_dataset(val_directory, val_labels_path, emotions_list, mtcnn)

    # Define the transformation for validation dataset
    transform_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_dataset = EmotionDataset(X_val, y_val, transform=transform_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Evaluate the model on the validation dataset
    print("Evaluating model...")
    evaluate_model(model, val_loader, emotions_list, device)

    # Finish wandb run
    wandb.finish()
