import sys
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.mlp import NIDSBinaryClassifier
from utils.metrics import calculate_accuracy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    # 1. Load Data
    df_test = pd.read_csv('../data/unsw-nb15/processed/test.csv')
    X_test = torch.tensor(df_test.drop('labels', axis=1).values, dtype=torch.float32)
    y_test = torch.tensor(df_test['labels'].values, dtype=torch.float32).unsqueeze(1)
    
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

    # 2. Load Model
    model = NIDSBinaryClassifier(input_shape=197).to(DEVICE)
    model.load_state_dict(torch.load('../models/nids_binary_model.pth'))
    model.eval() # Important: Disables Dropout

    # 3. Evaluation Loop
    total_acc = 0
    batches = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            
            acc = calculate_accuracy(outputs, labels)
            total_acc += acc
            batches += 1

    print(f"Final Test Accuracy: {(total_acc / batches) * 100:.2f}%")

if __name__ == "__main__":
    evaluate()