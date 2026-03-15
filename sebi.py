import os
HOME = os.getcwd()
print(HOME)

import pandas as pd
import numpy as np
from tqdm import tqdm

def get_frame_rate(data):
    avg_interval = data['timestamp'].diff().mean()
    current_fps = 1 / avg_interval if avg_interval > 0 else None
    return current_fps


def downsample_to_60Hz(data, target_fps=60):
    total_time = data['timestamp'].iloc[-1] - data['timestamp'].iloc[0]
    total_frames = len(data)
    current_fps = total_frames / total_time if total_time > 0 else None

    if current_fps and current_fps > target_fps:
        target_frame_count = int(total_time * target_fps)

        indices = np.linspace(0, total_frames - 1, target_frame_count, dtype=int)
        data_downsampled = data.iloc[indices].reset_index(drop=True)
        return data_downsampled
    else:
        return data


def upsample_to_60Hz(data, target_fps=60):
    target_interval = 1.0 / target_fps

    new_timestamps = np.arange(data['timestamp'].min(), data['timestamp'].max(), target_interval)
    data_interpolated = data.set_index('timestamp').reindex(new_timestamps).interpolate(method='linear').reset_index()
    data_interpolated.rename(columns={'index': 'timestamp'}, inplace=True)

    return data_interpolated

def insole_process(file_path):
    target_fps = 60 # a fixed 60hz sampling frequency for all the datas
    data = pd.read_csv(file_path)
    data = data.dropna()
    left_foot_data = data[data['ele_36'] == 1]  # Left foot data
    right_foot_data = data[data['ele_36'] == 0]  # right foot data

    # 0-17: pressure, 18-20: accelerometer, 30-33: quaternion (orientation)
    cols_to_extract = ['ele_' + str(i) for i in range(18)] + ['ele_18', 'ele_19', 'ele_20', 'ele_30', 'ele_31', 'ele_32', 'ele_33']

    left_foot_data = left_foot_data[['timestamp'] + cols_to_extract]
    right_foot_data = right_foot_data[['timestamp'] + cols_to_extract]

    left_fps = get_frame_rate(left_foot_data)
    right_fps = get_frame_rate(right_foot_data)

    # print('Left foot data', left_foot_data.shape, 'Frame rate: {:.2f} Hz'.format(left_fps))
    # print('Right foot data', right_foot_data.shape, 'Frame rate: {:.2f} Hz'.format(right_fps))

    # use low fps data to match high fps data
    if left_fps < right_fps:
        base_data = left_foot_data  # low fps data
        high_fps_data = right_foot_data  # high fps data
        suffixes = ('_left', '_right')
    else:
        base_data = right_foot_data
        high_fps_data = left_foot_data
        suffixes = ('_right', '_left')

    merged_data = pd.merge_asof(base_data.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True),
                                high_fps_data.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True),
                                on='timestamp',
                                direction='nearest',  # match nearest one
                                suffixes=suffixes,
                                tolerance=0.02) # 10ms = 0.01s

    right_first = any(col.endswith('_right') for col in merged_data.columns[1:])  # if right at front
    if right_first:
        left_columns = [col for col in merged_data.columns if '_left' in col]
        right_columns = [col for col in merged_data.columns if '_right' in col]
        merged_data = merged_data[['timestamp'] + left_columns + right_columns]  # change sequence

    merged_data = merged_data.dropna()
    current_fps = get_frame_rate(merged_data)

    if current_fps < target_fps:
        output_data = upsample_to_60Hz(merged_data, target_fps)
    elif current_fps > target_fps:
        output_data = downsample_to_60Hz(merged_data, target_fps)
    else:
        output_data = merged_data
    return output_data


#  you can change window size and overlap ratio (stride/window_size) here.
window_size = 20  # 0.33 second window (tunable)
stride = 15       # stride between consecutive windows (tunable)

# Transformer model hyperparameters (all tunable)
d_model = 128          # dimensionality of token embeddings
num_heads = 4          # number of attention heads
num_layers = 2         # number of Transformer encoder layers
dim_feedforward = 512  # FFN hidden size inside each encoder layer
dropout = 0.1          # dropout rate inside Transformer

# Training hyperparameters (all tunable)
batch_size = 32
learning_rate = 1e-4
num_epochs = 20

# You can select features here by disabling indexing. (feature set is tunable)
feature_columns = [
    # Left foot data in merged file
    'ele_0_left', 'ele_1_left', 'ele_2_left','ele_3_left', 'ele_4_left', 'ele_5_left','ele_6_left', 'ele_7_left', 'ele_8_left', 'ele_9_left', 'ele_10_left', 'ele_11_left',  # Pressure
    'ele_12_left', 'ele_13_left', 'ele_14_left','ele_15_left', 'ele_16_left', 'ele_17_left',  # Pressure
    'ele_18_left', 'ele_19_left', 'ele_20_left',  # Accelerometer x,y,z
    'ele_30_left', 'ele_31_left', 'ele_32_left', 'ele_33_left',  # Quaternion x,y,z,v

    # Right foot data in merged file
    'ele_0_right', 'ele_1_right', 'ele_2_right','ele_3_right', 'ele_4_right', 'ele_5_right','ele_6_right', 'ele_7_right', 'ele_8_right', 'ele_9_right', 'ele_10_right', 'ele_11_right',  # Pressure
    'ele_12_right', 'ele_13_right', 'ele_14_right','ele_15_right', 'ele_16_right', 'ele_17_right',  # Pressure
    'ele_18_right', 'ele_19_right', 'ele_20_right',  # Accelerometer x,y,z
    'ele_30_right', 'ele_31_right', 'ele_32_right', 'ele_33_right',  # Quaternion x,y,z,v
]

feature_size = len(feature_columns)
label_mapping = {"Normal_walking": 0,
                  "Injury_walking": 1,
                  "Stepping":2,
                  "Swaying":3,
                  "Jumping": 4,
                  }
all_features = []
all_labels = []

# Local folder containing your pasted CSV data
file_path = os.path.join(HOME, "train_dataset")

if not os.path.isdir(file_path):
    print(f"Error: {file_path} is not a directory or does not exist, double check the above file path!")


for folder_name in os.listdir(file_path):
    folder_path = os.path.join(file_path, folder_name)

    for keyword in tqdm(label_mapping, desc="Processing data", unit= folder_name):
        if keyword in folder_name:
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".csv"):
                    csv_file_path = os.path.join(folder_path, file_name)
                    # print("Processing",csv_file_path)
                    features = insole_process(csv_file_path)[feature_columns]

                    num_frames = features.shape[0]
                    for start_idx in range(0, num_frames - window_size + 1, stride):
                      end_idx = start_idx + window_size
                      window_features = features[start_idx:end_idx]

                      all_features.append(window_features)
                      all_labels.append(label_mapping[keyword])

all_features = np.array(all_features)
all_labels = np.array(all_labels)

print("="*50)
print("✅ Data Processing Completed Successfully!")
print(f"📊 Total Dataset Samples: {all_features.shape[0]}")
print(f"📏 Window Size: {window_size} frames ({window_size / 60:.2f} seconds)")
print(f"🔄 Overlap ratio: {stride/window_size:.2f}")
print(f"🧩 Feature Shape {all_features.shape}")
print(f"🔢 Number of Labels: {all_labels.shape[0]}")

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Define the dataset class for gait analysis
class GaitDataset(Dataset):
    def __init__(self, features, labels, Standard_flag=False, Normalize_flag=False):
        """
        Initialize the dataset with feature and label tensors.

        :param features: NumPy array of shape (num_samples, window_size, num_features)
        :param labels: NumPy array of shape (num_samples,)
        """
        num_samples, window_size, num_features = features.shape
        reshaped_features = features.reshape(-1, num_features)  # Flatten the window axis

        if Standard_flag:
            scaler = StandardScaler()

            # Fit the scaler and transform the data (normalize)
            Processed_features = scaler.fit_transform(reshaped_features)

            # Reshape back to the original dimensions (num_samples, window_size, num_features)
            features = Processed_features.reshape((num_samples, window_size, num_features))

        if Normalize_flag:
            scaler = MinMaxScaler()

            # Fit the scaler and transform the data (normalize)
            Processed_features = scaler.fit_transform(reshaped_features)

            # Reshape back to the original dimensions (num_samples, window_size, num_features)
            features = Processed_features.reshape((num_samples, window_size, num_features))

        self.features = torch.tensor(features, dtype=torch.float32)  # Convert features to a PyTorch tensor (float32)
        self.labels = torch.tensor(labels, dtype=torch.long)  # Convert labels to a PyTorch tensor (long, for classification)

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        Retrieve a single sample (features, label) by index.
        :param idx: Index of the sample to retrieve
        :return: Tuple (features, label)
        """
        return self.features[idx], self.labels[idx]

# Convert NumPy arrays into a PyTorch dataset
dataset = GaitDataset(all_features, all_labels,
                      Standard_flag = False,
                      Normalize_flag= False
                      )

train_size = int(0.8 * len(dataset))  # 80% for training
val_size = int(0.1 * len(dataset))    # 10% for validation
test_size = len(dataset) - train_size - val_size  # Remaining 10% for testing

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# ✅ Verify the dataset by printing its details
print(f"✅ Dataset loaded successfully! Total samples: {len(dataset)}")
print(f"   🔹 Training set: {len(train_dataset)} samples")
print(f"   🔹 Validation set: {len(val_dataset)} samples")
print(f"   🔹 Test set: {len(test_dataset)} samples")


import torch

# Function to train the model
def train(model, train_loader, criterion, optimizer):
    """
    Trains the given model using the provided training data.

    Args:
        model (torch.nn.Module): The neural network model.
        train_loader (torch.utils.data.DataLoader): DataLoader containing training data.
        criterion (torch.nn.Module): Loss function (e.g., CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): Optimization algorithm (e.g., Adam, SGD).

    Returns:
        tuple: Average training loss and training accuracy (in percentage).
    """
    model.train()  # Set model to training mode
    total_loss = 0  # Initialize total loss for the epoch
    correct = 0  # Initialize count of correctly classified samples
    total = 0  # Initialize total number of samples

    for inputs, targets in train_loader:  # Iterate over training batches
        inputs, targets = inputs.to(device), targets.to(device)  # Move data to the GPU or CPU
        optimizer.zero_grad()  # Clear previous gradients

        outputs = model(inputs)  # Forward pass: compute predictions
        loss = criterion(outputs, targets)  # Compute loss

        loss.backward()  # Backward pass: compute gradients
        optimizer.step()  # Update model weights

        total_loss += loss.item()  # Accumulate batch loss
        _, predicted = torch.max(outputs, 1)  # Get predicted class indices
        correct += (predicted == targets).sum().item()  # Count correct predictions
        total += targets.size(0)  # Count total samples

    accuracy = 100 * correct / total  # Compute accuracy as percentage
    return total_loss / len(train_loader), accuracy  # Return average loss and accuracy


# Function to validate the model
def validate(model, val_loader, criterion):
    """
    Evaluates the model performance on the validation dataset.

    Args:
        model (torch.nn.Module): The neural network model.
        val_loader (torch.utils.data.DataLoader): DataLoader containing validation data.
        criterion (torch.nn.Module): Loss function (e.g., CrossEntropyLoss).

    Returns:
        tuple: Average validation loss and validation accuracy (in percentage).
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0  # Initialize total validation loss
    correct = 0  # Initialize count of correctly classified samples
    total = 0  # Initialize total number of samples

    with torch.no_grad():  # Disable gradient computation for efficiency
        for inputs, targets in val_loader:  # Iterate over validation batches
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to the GPU or CPU

            outputs = model(inputs)  # Forward pass: compute predictions
            loss = criterion(outputs, targets)  # Compute loss

            total_loss += loss.item()  # Accumulate batch loss
            _, predicted = torch.max(outputs, 1)  # Get predicted class indices
            correct += (predicted == targets).sum().item()  # Count correct predictions
            total += targets.size(0)  # Count total samples

    accuracy = 100 * correct / total  # Compute accuracy as percentage
    return total_loss / len(val_loader), accuracy  # Return average loss and accuracy


import torch
import torch.nn as nn
import math


class TGTModel(nn.Module):
    """
    Temporal Gait Transformer (TGT) implementing the architecture described in the paper:
    - Linear projection of per-timestep features to d_model
    - Fixed sinusoidal positional encoding
    - Learnable [CLS] token
    - Transformer encoder stack
    - LayerNorm on [CLS] followed by linear classification head
    """

    def __init__(
        self,
        num_features: int,
        window_size: int,
        num_classes: int = 5,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.window_size = window_size
        self.d_model = d_model

        # Per-timestep linear projection to 128-D embedding
        self.input_proj = nn.Linear(num_features, d_model)

        # Fixed sinusoidal positional encoding (Window x d_model)
        pe = self._generate_positional_encoding(window_size, d_model)
        self.register_buffer("positional_encoding", pe, persistent=False)

        # Learnable [CLS] token of shape (1, 1, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # we will use (S, B, E) format
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.cls_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    @staticmethod
    def _generate_positional_encoding(window_size: int, d_model: int) -> torch.Tensor:
        """
        Standard sinusoidal positional encoding as described in the paper/text.
        Returns a tensor of shape (window_size, d_model).
        """
        pe = torch.zeros(window_size, d_model)
        position = torch.arange(0, window_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (Window, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, window_size, num_features)
        returns logits: (batch_size, num_classes)
        """
        # Ensure expected shape
        assert (
            x.dim() == 3 and x.size(1) == self.window_size and x.size(2) == self.num_features
        ), f"Expected input of shape (B, {self.window_size}, {self.num_features}), got {x.shape}"

        # Linear projection to (B, W, d_model)
        x = self.input_proj(x)

        # Add sinusoidal positional encoding (broadcast over batch)
        pe = self.positional_encoding.unsqueeze(0)  # (1, W, d_model)
        x = x + pe

        # Prepend [CLS] token: cls expanded to (B, 1, d_model)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, W+1, d_model)

        # Transformer expects (S, B, E)
        x = x.transpose(0, 1)  # (W+1, B, d_model)

        x = self.transformer_encoder(x)  # (W+1, B, d_model)

        # Take [CLS] token (position 0)
        cls_rep = x[0]  # (B, d_model)
        cls_rep = self.cls_norm(cls_rep)
        logits = self.classifier(cls_rep)  # (B, num_classes)
        return logits


# Instantiate Temporal Gait Transformer model
my_ml_model = TGTModel(
    num_features=feature_size,
    window_size=window_size,
    num_classes=5,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
)

import subprocess

subprocess.run(["pip", "install", "torchinfo"], check=True)

from torchinfo import summary

# Print model summary (we pass (batch_size, window_size, feature_size))
print(summary(my_ml_model, input_size=(batch_size, window_size, feature_size)))

import torch.optim as optim
import random

# Local folder to save trained models
save_file_path = os.path.join(HOME, "Saved_model")
os.makedirs(save_file_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set a fixed random seed to ensure reproducibility in model training
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# create DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# CrossEntropyLoss with optimizer, and think is there anything more you can add
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(my_ml_model.parameters(), lr=learning_rate)

model = my_ml_model.to(device)

# Initialize lists to store the losses and accuracies for plotting
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# main training loop
for epoch in range(num_epochs):
    train_loss, train_accuracy = train(model, train_loader, criterion, optimizer)
    val_loss, val_accuracy = validate(model, val_loader, criterion)

    # Store the results for plotting
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch + 1}/{num_epochs}], ")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, ")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

# Save the model after training
model_save_path = os.path.join(save_file_path, "gait_model.pth")
torch.save(model, model_save_path)
print(f"Model saved to {model_save_path}")


import matplotlib.pyplot as plt

# After training is done, plot the loss and accuracy curves
# Plotting Loss Curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)  # Plot the loss curve
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', color='blue')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='red')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plotting Accuracy Curves
plt.subplot(1, 2, 2)  # Plot the accuracy curve
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy', color='blue')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy', color='red')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    accuracy = 100 * correct / total
    return accuracy


import torch

# Select device: Use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the path where the trained model is saved (same as save_file_path above)
model_save_path = os.path.join(HOME, "Saved_model", "gait_model.pth")

# Recreate the model architecture (Make sure it matches the saved model)
model = MLP(input_dim=window_size * feature_size, hidden_dim=8, output_dim=5)

# Load the trained model
# 'weights_only=False' ensures that we are loading the entire model (structure + weights)
model = torch.load(model_save_path, map_location=device, weights_only=False)

# Move the model to the selected device (GPU or CPU)
model.to(device)

# Evaluate the model on the test dataset
test_accuracy = test(model, test_loader)

# Print a detailed summary of the test results
print("=" * 50)
print("✅ Model Testing Completed!")
print(f"📌 Model Path: {model_save_path}")
print(f"📊 Test Dataset Accuracy: {test_accuracy:.2f}%")
print("📌 Device Used:", device)
print("=" * 50)

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(model, test_loader, class_names):
    """
    Generates and plots a confusion matrix for the given model on the test dataset.

    :param model: Trained PyTorch model
    :param test_loader: DataLoader for test dataset
    :param class_names: List of class names for labeling the confusion matrix
    """
    model.eval()  # Set model to evaluation mode

    all_preds = []
    all_test_labels = []

    # Disable gradient calculation for efficiency
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())  # Store predictions
            all_test_labels.extend(labels.cpu().numpy())  # Store true labels

    # Compute the confusion matrix
    cm = confusion_matrix(all_test_labels, all_preds)

    # Normalize confusion matrix (convert to percentage)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot the confusion matrix using Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    # Print classification report
    print("Classification Report:\n", classification_report(all_test_labels, all_preds, target_names=class_names, zero_division=0))

# Define class names (modify based on your dataset)
class_names = ["Normal_walking", "Injury_walking", "Stepping", "Swaying", "Jumping"]

# Call the function to plot the confusion matrix
plot_confusion_matrix(model, test_loader, class_names)

