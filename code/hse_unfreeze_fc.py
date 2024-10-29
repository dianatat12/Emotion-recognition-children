import os
import torch
import cv2
import pandas as pd
import numpy as np
import wandb
from facenet_pytorch import MTCNN
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from hsemotion.facial_emotions import HSEmotionRecognizer
from torch.optim.lr_scheduler import OneCycleLR

# Initialize wandb
wandb.init(project="emoreact_girls_fc", entity="dtat-universitat-oberta-de-catalunya")

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


def read_and_process_labels(file_path, emotions_list):
    df = pd.read_excel(file_path)
    df.columns = ['Video Name'] + emotions_list
    df = df.set_index('Video Name')
    
    # Map the labels
    emotion_data = df.applymap(lambda x: 1 if x in [1, 2] else 0)
    
    print(f"Processed labels shape: {emotion_data.shape}")
    print(f"Sample of processed labels:\n{emotion_data.head()}")
    return emotion_data

def find_matching_video(subfolder_name, labels_df):
    base_name = subfolder_name.rsplit('.', 1)[0].lower().strip()  # Strip any whitespace
    for video_name in labels_df.index:
        if base_name in video_name.lower().strip():
            return video_name
    print(f"No match found for subfolder: {subfolder_name} with base name: {base_name}")
    return None


def load_dataset(directory, labels_path, emotions_list, mtcnn):
    labels_df = read_and_process_labels(labels_path, emotions_list)
    images, labels = [], []

    for subfolder_name in os.listdir(directory):
        subfolder_path = os.path.join(directory, subfolder_name)
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


def manipulate_layers(model_name, num_classes, device):
    fer = HSEmotionRecognizer(model_name=model_name, device=device)
    
    # Freeze all layers
    for param in fer.model.parameters():
        param.requires_grad = False
    
    # Identify all convolutional layers
    conv_layers = [module for module in fer.model.modules() if isinstance(module, nn.Conv2d)]
    
    # Unfreeze the last 2 convolutional layers
    last_2_convs = conv_layers[-2:]
    for conv in last_2_convs:
        for param in conv.parameters():
            param.requires_grad = True
    
    # Unfreeze the classifier (fully connected layer) and add Dropout
    fer.model.classifier = nn.Sequential(
        nn.BatchNorm1d(1280),
        nn.Linear(1280, 640),
        nn.ReLU(),
        nn.Dropout(p=0.5),  # Add Dropout with a probability of 0.5
        nn.BatchNorm1d(640),
        nn.Linear(640, num_classes),
        nn.Softmax(dim=1)  # Add Softmax for multi-class classification
    )
    
    for param in fer.model.classifier.parameters():
        param.requires_grad = True
    
    # Print trainable and frozen layers
    for name, param in fer.model.named_parameters():
        if param.requires_grad:
            print(f"Trainable layer: {name}")
        else:
            print(f"Frozen layer: {name}")
    
    return fer.model.to(device)


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=100, patience=20):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0
    best_loss = float('inf')
    epochs_no_improve = 0
    
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

            for inputs, labels in dataloaders[phase]:
                if inputs.size(0) == 1:
                    print(f"Skipping batch with size 1 in {phase} phase")
                    continue  # Skip this batch
                
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
                preds = torch.softmax(outputs, dim=1)
                all_preds.extend(preds.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            if all_preds and all_labels:
                all_preds = np.array(all_preds)
                all_labels = np.array(all_labels)
                
                # Calculate and log metrics
                accuracy = accuracy_score(all_labels.argmax(axis=1), all_preds.argmax(axis=1))
                f1 = f1_score(all_labels.argmax(axis=1), all_preds.argmax(axis=1), average='weighted')
                # auc = roc_auc_score(all_labels, all_preds, average='weighted', multi_class='ovr')

                print(f'{phase} Loss: {epoch_loss:.4f} Accuracy: {accuracy:.4f} F1: {f1:.4f} ')

                wandb.log({f"{phase}_loss": epoch_loss, 
                           f"{phase}_accuracy": accuracy,
                           f"{phase}_f1": f1,
                        #    f"{phase}_auc": auc,
                           "epoch": epoch,
                           "learning_rate": optimizer.param_groups[0]['lr']})

                # Log confusion matrix
                calculate_metrics(all_labels, all_preds, emotions_list, phase=phase)
                
            else:
                print(f"No valid predictions for {phase} phase in epoch {epoch}.")

            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if f1 > best_f1:
                    best_f1 = f1
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        print(f"Epoch {epoch} complete. Best Loss: {best_loss:.4f}. Best F1: {best_f1:.4f}.")

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}. No improvement in {patience} consecutive epochs.")
            break

    model.load_state_dict(best_model_wts)
    
    return model


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=device)

    emotions_list = ['Happiness', 'Surprise', 'Frustration']
    num_classes = len(emotions_list)

    train_directory = '/home/diana/Documents/EmoReact_Reselection/girls/girls_train'
    val_directory = '/home/diana/Documents/EmoReact_Reselection/girls/girls_test'

    train_path = '/home/diana/Documents/EmoReact_Reselection/girls/emoreact_girls_train.xlsx'
    val_path = '/home/diana/Documents/EmoReact_Reselection/girls/emoreact_girls_test.xlsx'

    # Load training data
    X_train, y_train = load_dataset(train_directory, train_path, emotions_list, mtcnn)
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of y_train: {y_train.shape}")

    # Load validation data
    X_val, y_val = load_dataset(val_directory, val_path, emotions_list, mtcnn)
    print(f"Shape of X_val: {X_val.shape}")
    print(f"Shape of y_val: {y_val.shape}")

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = EmotionDataset(X_train, y_train, transform=transform_train)
    val_dataset = EmotionDataset(X_val, y_val, transform=transform_val)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    model_name = 'enet_b0_8_best_afew'
    model = manipulate_layers(model_name, num_classes, device)

    loss_function = 'focal'  

    if loss_function == 'crossentropy':
        criterion = nn.CrossEntropyLoss()

    elif loss_function == 'focal':
        criterion = FocalLoss(alpha=0.25, gamma=2)

    elif loss_function == 'cb_loss':
        samples_per_cls = np.sum(y_train, axis=0)
        criterion = lambda logits, labels: CB_loss(labels, logits, samples_per_cls, num_classes)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-2)
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=100)

    wandb.config.update({
        "batch_size": batch_size,
        "initial_learning_rate": 1e-4,
        "weight_decay": 1e-2,
        "num_epochs": 100,
        "patience": 20,
        "model": "enet_b0_8_best_afew",
        "loss_function": loss_function
    })

    model = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=100, patience=20)

    # Save the final model weights with the specified name
    torch.save(model.state_dict(), f'/home/diana/Documents/pth_reselection/emoreact_girls_fc.pth')
    wandb.save(f'/home/diana/Documents/pth_reselection/emoreact_girls_fc.pth')

    wandb.finish()

print("Training, evaluation, and logging completed.")

