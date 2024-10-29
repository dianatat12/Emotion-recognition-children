import os
import torch
import cv2
import pandas as pd
import numpy as np
import wandb
from facenet_pytorch import MTCNN
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F
import copy
from sklearn.preprocessing import label_binarize

# Initialize Weights & Biases project
wandb.init(project="emoreact_girls_scratch", entity="dtat-universitat-oberta-de-catalunya")

# Custom Dataset class for Emotion Dataset
class EmotionDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = np.array(image, dtype=np.float32)
        image = np.transpose(image, (1, 2, 0))
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.float32)
        return image, label

# Modify the model head for fine-tuning
class EmotionClassifierEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(EmotionClassifierEfficientNet, self).__init__()
        self.model = models.efficientnet_b0(weights=None)
        num_features = self.model.classifier[1].in_features
        
        # Updated head with Dropout applied before the last layer
        self.model.classifier = nn.Sequential(
            nn.Linear(num_features, 256),  # Additional linear layer with 256 output features
            nn.ReLU(inplace=True),         # Activation function
            nn.Dropout(0.5),               # Dropout layer with 50% probability
            nn.Linear(256, num_classes)    # Final prediction layer
        )

    def forward(self, x):
        return self.model(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        cross_entropy_loss = nn.functional.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-cross_entropy_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * cross_entropy_loss
        return focal_loss.mean()

def CB_loss(labels, logits, samples_per_cls, no_of_classes, beta=0.9999, gamma=2.0):
    device = logits.device
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    weights = torch.tensor(weights).float().to(device)
    labels = labels.argmax(dim=1).to(device)
    
    cross_entropy_loss = nn.functional.cross_entropy(logits, labels, weight=weights, reduction='none')
    pt = torch.exp(-cross_entropy_loss)
    focal_loss = (1 - pt) ** gamma * cross_entropy_loss
    return focal_loss.mean()

def calculate_metrics(y_true, y_pred, emotions_list, phase='train'):
    y_true_single_class = y_true.argmax(axis=1)
    y_pred_single_class = y_pred.argmax(axis=1)
    
    accuracy = accuracy_score(y_true_single_class, y_pred_single_class)
    f1 = f1_score(y_true_single_class, y_pred_single_class, average='weighted')
    # try:
    #     auc = roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovr')
    # except ValueError:
    #     auc = 0
    
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    # print(f"AUC: {auc}")
    
    wandb.log({
        f"{phase}_accuracy": accuracy,
        f"{phase}_f1": f1,
        # f"{phase}_auc": auc
    })

    cm = confusion_matrix(y_true_single_class, y_pred_single_class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=emotions_list, yticklabels=emotions_list)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - All Emotions ({phase.capitalize()})')
    
    # Instead of saving the image, directly log it to wandb
    wandb.log({f"confusion_matrix_all_{phase}": wandb.Image(plt)})
    plt.close()
    
    return f1


# Load labels and data
def read_and_process_labels(file_path):
    df = pd.read_excel(file_path, header=None)
    df.columns = df.iloc[0]
    df = df.set_index(df.columns[0])
    df = df.iloc[1:]
    df.columns = df.columns.astype(str)
    df.columns = ['0', '1', '2']
    emotion_data = df.apply(pd.to_numeric, errors='coerce')
    return emotion_data

def detect_face(frame, mtcnn):
    bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)
    if bounding_boxes is None:
        return []
    else:
        bounding_boxes = bounding_boxes[probs > 0.9]
        return bounding_boxes


def process_frames_in_subfolder(subfolder_path, label, mtcnn):
    subfolder_results = []
    for root, _, files in os.walk(subfolder_path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                frame_path = os.path.join(root, file)
                frame_bgr = cv2.imread(frame_path)
                if frame_bgr is None:
                    print(f"Error reading image: {frame_path}")
                    continue
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                bounding_boxes = detect_face(frame_rgb, mtcnn)
                for bbox in bounding_boxes:
                    x1, y1, x2, y2 = bbox.astype(int)
                    face_img = frame_rgb[y1:y2, x1:x2, :]
                    if face_img.size == 0:
                        print(f"Empty face image extracted from: {frame_path}")
                        continue
                    face_img = cv2.resize(face_img, (224, 224))
                    face_img = np.transpose(face_img, (2, 0, 1))
                    face_img = face_img / 255.0
                    subfolder_results.append((face_img, label))
    return subfolder_results

def train_model(model, dataloaders, samples_per_cls, no_of_classes, optimizer, scheduler, num_epochs=100, patience=10, loss_function="focal"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
    epochs_no_improve = 0
    emotion_labels = ['Happiness', 'Surprise', 'Frustration']

    # Define the loss function
    if loss_function == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    
    if loss_function == 'focal':
        criterion = FocalLoss(alpha=0.25, gamma=2)

    elif loss_function == 'cb_loss':
        criterion = lambda logits, labels: CB_loss(labels, logits, samples_per_cls, no_of_classes)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_preds = []
            all_labels = []

            loader = dataloaders[phase]
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                preds = torch.softmax(outputs, dim=1).detach().cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.detach().cpu().numpy())

            epoch_loss = running_loss / len(loader.dataset)
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)

            # Call calculate_metrics for metrics logging and confusion matrix, and capture the f1 score
            f1 = calculate_metrics(all_labels, all_preds, emotions_list=emotion_labels, phase=phase)

            wandb.log({f'{phase}_loss': epoch_loss})

            if phase == 'val':
                scheduler.step()
                if f1 > best_f1:
                    best_f1 = f1
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    print(f'Best val F1: {best_f1:.4f}')
    model.load_state_dict(best_model_wts)
    return model


# Data loading and preparation
def find_matching_video(subfolder_name, labels_df):
    base_name = subfolder_name.rsplit('.', 1)[0].lower()
    for video_name in labels_df.index:
        if base_name in video_name.lower():
            return video_name
    return None

# Main training script
if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=device)

    emotions_list = ['Happiness', 'Surprise', 'Frustration']
    num_classes = len(emotions_list)

    # Define paths to datasets
    train_directory = '/home/diana/Documents/EmoReact_Reselection/girls/girls_train'
    val_directory = '/home/diana/Documents/EmoReact_Reselection/girls/girls_test'

    train_path = '/home/diana/Documents/EmoReact_Reselection/girls/emoreact_girls_train.xlsx'
    val_path = '/home/diana/Documents/EmoReact_Reselection/girls/emoreact_girls_test.xlsx'

    # Process datasets
    train_labels_df = read_and_process_labels(train_path)
    val_labels_df = read_and_process_labels(val_path)

    train_images, train_labels = [], []
    val_images, val_labels = [], []

    # Process training and validation data
    def load_data(folder, labels_df):
        images, labels = [], []
        for subfolder_name in os.listdir(folder):
            subfolder_path = os.path.join(folder, subfolder_name)
            if os.path.isdir(subfolder_path):
                print(f"Processing frames in subfolder: {subfolder_name}")
                matching_video = find_matching_video(subfolder_name, labels_df)
                if matching_video:
                    label = labels_df.loc[matching_video].values
                    frame_results = process_frames_in_subfolder(subfolder_path, label, mtcnn)
                    for face_img, lbl in frame_results:
                        images.append(face_img)
                        labels.append(lbl)
                else:
                    print(f"Warning: No label found for {subfolder_name}")
        return np.array(images), np.array(labels)

    train_images, train_labels = load_data(train_directory, train_labels_df)
    val_images, val_labels = load_data(val_directory, val_labels_df)

    # Define transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = EmotionDataset(train_images, train_labels, transform=train_transform)
    val_dataset = EmotionDataset(val_images, val_labels, transform=val_transform)

    # Define batch_size
    batch_size = 32

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    # Initialize model, optimizer, scheduler, and parameters
    model = EmotionClassifierEfficientNet(num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=100)

    # Update WandB config
    wandb.config.update({
        "batch_size": batch_size,
        "learning_rate": 1e-4,
        "weight_decay": 1e-2,
        "num_epochs": 100,
        "patience": 10,
        "loss_function": "focal"  # Switch to "cb_loss" as needed
    })

    # Train the model with early stopping
    model = train_model(model, dataloaders, np.sum(train_labels, axis=0), num_classes, optimizer, scheduler, num_epochs=100, patience=10, loss_function="focal")

    # Save the model
    torch.save(model.state_dict(), f'/home/diana/Documents/pth_reselection/emoreact_girls_scratch.pth')
    wandb.save(f'/home/diana/Documents/pth_reselection/emoreact_girls_scratch.pth')

    wandb.finish()

print("Training, evaluation, and logging completed.")
