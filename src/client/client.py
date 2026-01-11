import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import copy
from src.client.attacker import Attacker
from src.utils.class_weights import get_class_weights

class Client:
    def __init__(self, client_id, dataset, indices, model, config , lr=0.01 , device='cpu', is_malicious=False):
        self.client_id = client_id
        self.device = device
        self.is_malicious = is_malicious
        self.lr = lr
        self.config = config
        
        # 1. Create Local Data Slice
        self.dataset = Subset(dataset, indices)
        
        # 2. RED TEAM INTEGRATION 😈
        # If this client is malicious, we POISON the data immediately
        if self.is_malicious:
            print(f"⚠️ Client {client_id} is MALICIOUS! Initializing Attacker...")
            self.attacker = Attacker(config)
            # Replace honest dataset with poisoned dataset
            self.dataset = self.attacker.poison_dataset(self.dataset)
        
        # 3. M12: Calculate local class weights from client's data
        if config.client.get('use_class_weights', False):
            # Extract labels from local dataset
            local_labels = torch.tensor([dataset[i][1] for i in indices])
            weight_method = config.client.get('weight_method', 'sqrt')
            self.class_weights = get_class_weights(local_labels, device, method=weight_method)
            print(f"   Client {client_id}: Using {weight_method} class weights")
        else:
            self.class_weights = None
        
        # 4. Local Model Setup
        self.model = copy.deepcopy(model).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)  # M12: Use Adam
        
        # M12: Weighted loss if enabled
        if self.class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def train(self, global_weights, epochs=1, batch_size=32):
        # Load global weights
        self.model.load_state_dict(global_weights)
        self.model.train()
        
        train_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        # M12: Re-initialize optimizer with current model parameters
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # M12: Learning rate scheduler
        use_scheduler = self.config.client.get('use_scheduler', False)
        if use_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=2
            )
        
        # M12: Early stopping
        patience = self.config.client.get('early_stopping_patience', None)
        best_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # Calculate average loss for this epoch
            avg_epoch_loss = epoch_loss / len(train_loader)
            
            # M12: Update scheduler
            if use_scheduler:
                scheduler.step(avg_epoch_loss)
            
            # M12: Early stopping check
            if patience is not None:
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    # print(f"   Client {self.client_id}: Early stopping at epoch {epoch+1}")
                    break
        
        # Final average loss across all epochs and batches
        avg_loss = best_loss if patience is not None else avg_epoch_loss

        # 🆕 PHASE 2 LOGIC: Model Replacement
        final_weights = self.model.state_dict()
        
        # Only apply if this client is Malicious AND has an attacker attached
        if self.is_malicious and hasattr(self, 'attacker'):
             # We pass the original global weights to calculate the delta
            final_weights = self.attacker.scale_update(global_weights, final_weights)

        # Return the (possibly boosted) weights
        return final_weights, len(self.dataset), avg_loss