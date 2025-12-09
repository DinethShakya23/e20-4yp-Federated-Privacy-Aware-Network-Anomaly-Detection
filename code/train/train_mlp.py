import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

# Add the parent directory to path so we can import 'models' and 'utils'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.mlp import NIDSBinaryClassifier
from utils.metrics import calculate_accuracy

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.001
# Get the absolute path of the current script file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate up two levels (train -> code -> root) to find 'data'
PROCESSED_PATH = os.path.join(SCRIPT_DIR, '../../data/unsw-nb15/processed/')

def load_data(path):
    df = pd.read_csv(path)
    # Ensure all feature columns are numeric (floats) before creating tensors.
    # Mixed dtypes (bool + float + int) make the underlying NumPy array "object",
    # which `torch.tensor` cannot convert directly.
    features = df.drop('label', axis=1)
    X_np = features.to_numpy(dtype='float32')
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(df['label'].values, dtype=torch.float32).unsqueeze(1)
    return TensorDataset(X, y)

def train():
    print(f"Training on {DEVICE}")
    
    # 1. Load Data
    train_data = load_data(os.path.join(PROCESSED_PATH, 'train.csv'))
    print(f"Training samples: {len(train_data)}")
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Initialize Model
    # Match input_shape to the actual number of features in the data
    input_dim = train_data.tensors[0].shape[1]
    model = NIDSBinaryClassifier(input_shape=input_dim).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f}")

    # 4. Save the trained model state
    torch.save(model.state_dict(), os.path.join(SCRIPT_DIR, '../models/nids_binary_model.pth'))
    print("Model saved to models/nids_binary_model.pth")

if __name__ == "__main__":
    train()