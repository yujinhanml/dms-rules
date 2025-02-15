import argparse
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import wandb
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import re

# Command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Train a neural network for latent space verification.")
    parser.add_argument('--training_epoch', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--base_path', type=str, default= "/cpfs04/user/hanyujin/causal-dm/results/sunshadow_lfd_lnd_rfd_rnd/vis/epoch_400_1737189887.87018/", help='Path to the folder containing prediction files')
    parser.add_argument('--file_name', type=str, default= "prediction_1000.npz", help='Path to the folder containing prediction files')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--output_dim', type=int, default=2, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--use_gpu', type=int, default=1, help='Which GPU to use (if available)')
    parser.add_argument('--weight', type=float, default=0.5, help='Weight of the contrastive loss')
    return parser.parse_args()

from sklearn.preprocessing import StandardScaler

# Load and process data (same as your original)
def load_and_process_data(file_path, args):
    npz_data = np.load(file_path)

    x = npz_data['x']
    label = npz_data['label']

    # 标准化数据
    scaler = StandardScaler()
    original_shape = x.shape
    x = x.reshape(x.shape[0], -1)
    x = scaler.fit_transform(x)
    x = x.reshape(original_shape)

    # Assign labels (same as your original)
    new_labels = np.zeros_like(label, dtype=int)
    if args.output_dim == 5:
        new_labels[(label >= 0) & (label < 0.4)] = 2
        new_labels[(label >= 0.4) & (label < 0.8)] = 1
        new_labels[(label >= 0.8) & (label < 1.2)] = 0
        new_labels[(label >= 1.2) & (label < 1.6)] = 3
        new_labels[(label >= 1.6) & (label <= 2.0)] = 4
    if args.output_dim == 3:
        new_labels[(label <= 0)] = 1
        new_labels[(label >= 0.8) & (label <= 1.2)] = 0
        new_labels[((label < 0.8) & (label > 0)) | (label > 1.2)] = 2
    if args.output_dim == 4:
        new_labels[(label <= 0)] = 1
        new_labels[(label >= 0.8) & (label <= 1.2)] = 0
        new_labels[((label < 0.8) & (label > 0))] = 2
        new_labels[(label > 1.2)] = 3
    if args.output_dim == 2:
        new_labels[(label < 0.2) | (label >= 1.2)] = 1
        new_labels[(label >= 0.8) & (label < 1.2)] = 0

    return x, new_labels

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Contrastive Loss function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = 0.5 * (label.float() * torch.pow(euclidean_distance, 2) +
                      (1 - label).float() * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss.mean()

# Dataset class (same as your original)
class PredictionDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# Training function
def train(model, train_loader, test_loader, num_epochs, optimizer, criterion, contrastive_criterion, device, args):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        class_correct_train = [0] * args.output_dim
        class_total_train = [0] * args.output_dim

        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(data)
            # Classification loss
            loss = criterion(outputs, labels)

            # Contrastive loss (Pairwise distance)
            # Create positive and negative pairs from the batch (you can customize this logic)
            # For simplicity, we will take all pairs of samples from the same batch and compare their outputs.
            contrastive_loss = 0.0
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    if labels[i] == labels[j]:
                        contrastive_label = torch.tensor([1]).to(device)  # Same class
                    else:
                        contrastive_label = torch.tensor([0]).to(device)  # Different class
                    contrastive_loss += contrastive_criterion(outputs[i:i+1], outputs[j:j+1], contrastive_label)

            total_loss = loss + args.weight * contrastive_loss

            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Calculate accuracy for each class
            for i in range(labels.size(0)):
                label = labels[i]
                if predicted[i] == label:
                    class_correct_train[label] += 1
                class_total_train[label] += 1

        # Calculate training accuracy
        train_acc = correct_train / total_train * 100
        class_acc_train = [class_correct_train[i] / class_total_train[i] * 100 if class_total_train[i] > 0 else 0 for i in range(args.output_dim)]

        # Evaluate on test data (same as your original)
        model.eval()
        correct_test = 0
        total_test = 0
        class_correct_test = [0] * args.output_dim
        class_total_test = [0] * args.output_dim

        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()


        # Calculate test accuracy
        test_acc = correct_test / total_test * 100
        # class_acc_test = [class_correct_test[i] / class_total_test[i] * 100 if class_total_test[i] > 0 else 0 for i in range(args.output_dim)]


        # Log metrics to WandB
        wandb.log({'train_loss': running_loss / len(train_loader), 'train_accuracy': train_acc, 'test_accuracy': test_acc})

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_acc:.2f}%")
        print(f"Train Class Accuracy: {class_acc_train}")
    model_path = "/cpfs04/user/hanyujin/causal-dm/latent_class/model/"+f"classifier_latent900_{num_epochs}.pth"
    torch.save(model.state_dict(), model_path)

# Main function
def main():
    args = parse_args()
    folder_path = args.base_path + args.file_name
    data, labels = load_and_process_data(folder_path, args)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    train_dataset = PredictionDataset(X_train, y_train)
    test_dataset = PredictionDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    match = re.search(r'prediction_(\d+)\.npz', args.file_name)
    if match:
        args.number = int(match.group(1))
    else:
        args.number =  -1

    wandb.init(project='latent_space_verification', name=f'contrastive{args.weight}_latent{args.number}_epoch{args.training_epoch}_lr{args.learning_rate}_outputdim{args.output_dim}')

    model = SimpleNN(input_dim=data.shape[1], output_dim=args.output_dim)
    device = torch.device(f"cuda:{args.use_gpu}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    contrastive_criterion = ContrastiveLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train(model, train_loader, test_loader, num_epochs=args.training_epoch, optimizer=optimizer, criterion=criterion, contrastive_criterion=contrastive_criterion, device=device, args=args)

if __name__ == '__main__':
    main()
