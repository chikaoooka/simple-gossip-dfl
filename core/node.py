import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn import SimpleCNN

class Node:
    def __init__(self, node_id, model, neighbors, local_data_loader, device='cpu',
                 local_epochs=1, learning_rate=0.01, comm_cost_per_kb=1, seed=42):
        """Initialize a node in the decentralized network."""
        self.node_id = node_id
        self.neighbors = neighbors
        self.local_data_loader = local_data_loader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.comm_cost_per_kb = comm_cost_per_kb
        self.comm_cost_log = []
        self.loss_history = []
        self.accuracy_history = []
        torch.manual_seed(seed)
        
        # Initialize model
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
    
    def train_local_model(self):
        """Train the model on local data."""
        self.model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(self.local_epochs):
            running_loss = 0.0
            for i, data in enumerate(self.local_data_loader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            epoch_loss = running_loss / len(self.local_data_loader)
        
        self.loss_history.append(epoch_loss)
        self.accuracy_history.append(100 * correct / total)
        
        return self.model.get_param_dict()
    
    def receive_model(self, params_dict):
        """Receive model parameters from a neighbor."""
        # Calculate communication cost based on parameter size
        param_size_bytes = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        comm_cost = (param_size_bytes / 1024) * self.comm_cost_per_kb  # Convert to KB
        self.comm_cost_log.append(comm_cost)
        return comm_cost
    
    def aggregate_models(self, neighbors_params_list):
        """Aggregate model parameters from neighbors."""
        #TODO: Implement FedAvg
        if not neighbors_params_list:
            return
        
        # Add self model to the averaging
        all_params = [self.model.get_param_dict()] + neighbors_params_list
        avg_params = {}
        
        # Calculate average for each parameter
        for param_name in all_params[0].keys():
            # Stack all tensors for this parameter
            stacked_params = torch.stack([params[param_name] for params in all_params])
            # Calculate mean across models
            avg_params[param_name] = torch.mean(stacked_params, dim=0)
        
        # Update model with averaged parameters
        self.model.set_param_dict(avg_params)
    
    def evaluate(self, test_loader):
        """Evaluate the model on test data."""
        self.model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = test_loss / len(test_loader)
        
        return avg_loss, accuracy