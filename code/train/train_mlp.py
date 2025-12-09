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
    
    # --- DEBUGGING BLOCK ---
    # Check for non-numeric columns
    non_numeric = df.select_dtypes(include=['object']).columns
    if len(non_numeric) > 0:
        print(f"ERROR: Found non-numeric columns in {path}:")
        print(non_numeric.tolist())
        # Print first few values of the culprit columns
        print(df[non_numeric].head())
        raise ValueError("Data contains strings! Preprocessing might have failed.")
    # -----------------------
    X = torch.tensor(df.drop('label', axis=1).values, dtype=torch.float32) 
    y = torch.tensor(df['label'].values, dtype=torch.float32).unsqueeze(1)
    return TensorDataset(X, y)

def train():
    print(f"Training on {DEVICE}")
    
    # 1. Load Data
    train_data = load_data(os.path.join(PROCESSED_PATH, 'train.csv'))
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Initialize Model
    model = NIDSBinaryClassifier(input_shape=197).to(DEVICE)
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
    torch.save(model.state_dict(), '../models/nids_binary_model.pth')
    print("Model saved to models/nids_binary_model.pth")

if __name__ == "__main__":
    train()