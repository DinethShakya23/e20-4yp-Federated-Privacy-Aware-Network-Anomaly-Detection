import torch
import torch.nn as nn

class NIDSBinaryClassifier(nn.Module):
    def __init__(self, input_shape=197):
        super(NIDSBinaryClassifier, self).__init__()
        
        # Layer 1: Input -> Hidden
        self.layer1 = nn.Linear(input_shape, 64) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        
        # Layer 2: Hidden -> Output
        self.output = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x