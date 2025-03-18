import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import time
import copy

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Define CNN model for CIFAR-10
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_param_dict(self):
        """Get the model parameters as a dictionary."""
        return {k: v.clone() for k, v in self.state_dict().items()}

    def set_param_dict(self, params_dict):
        """Set the model parameters from a dictionary."""
        self.load_state_dict(params_dict)

class Node:
    def __init__(self, node_id, neighbors, local_data_loader, device='cpu',
                 local_epochs=1, learning_rate=0.01, comm_cost_per_kb=1):
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
        
        # Initialize model
        self.model = SimpleCNN().to(self.device)
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
        """Aggregate model parameters from neighbors using FedAvg."""
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


class GossipFederatedLearning:
    def __init__(self, num_nodes, connectivity_prob=0.3, comm_prob=0.5, device='cpu'):
        """Initialize the decentralized federated learning system."""
        self.num_nodes = num_nodes
        self.connectivity_prob = connectivity_prob
        self.comm_prob = comm_prob
        self.device = device
        self.nodes = []
        self.network = None
        self.total_comm_cost = 0
        self.global_loss_history = []
        self.global_accuracy_history = []
        self.round_comm_costs = []
    
    def generate_network(self):
        """Generate a random network using Erdos-Renyi model."""
        self.network = nx.erdos_renyi_graph(self.num_nodes, self.connectivity_prob)
        
        # Ensure the graph is connected
        while not nx.is_connected(self.network):
            components = list(nx.connected_components(self.network))
            if len(components) > 1:
                comp1 = random.choice(list(components[0]))
                comp2 = random.choice(list(components[1]))
                self.network.add_edge(comp1, comp2)
        
        return self.network
    
    def partition_cifar10(self, train_dataset, iid=False, batch_size=64, num_workers=2):
        """Partition the CIFAR-10 dataset among nodes."""
        if iid:
            # IID partitioning
            partition_size = len(train_dataset) // self.num_nodes
            partition_lengths = [partition_size] * self.num_nodes
            partition_lengths[-1] += len(train_dataset) - sum(partition_lengths)
            
            train_partitions = torch.utils.data.random_split(train_dataset, partition_lengths)
            
        else:
            # Non-IID partitioning: each node gets 2 primary classes
            # Get all targets
            targets = torch.tensor(train_dataset.targets)
            
            # Partition by class
            class_indices = [torch.where(targets == i)[0] for i in range(10)]
            
            # Assign classes to nodes (each node gets 2 primary classes)
            node_class_map = {i: [(i*2) % 10, (i*2 + 1) % 10] for i in range(self.num_nodes)}
            
            # Create partitions
            node_indices = [[] for _ in range(self.num_nodes)]
            for node_id, primary_classes in node_class_map.items():
                # Take 90% of the primary classes
                for cls in primary_classes:
                    class_size = len(class_indices[cls])
                    node_indices[node_id].extend(class_indices[cls][:int(0.9 * class_size)])
                    class_indices[cls] = class_indices[cls][int(0.9 * class_size):]
            
            # Distribute remaining data
            remaining = []
            for cls_idx in class_indices:
                remaining.extend(cls_idx)
            
            # Shuffle remaining
            random.shuffle(remaining)
            samples_per_node = len(remaining) // self.num_nodes
            
            for i in range(self.num_nodes):
                start_idx = i * samples_per_node
                end_idx = start_idx + samples_per_node if i < self.num_nodes - 1 else len(remaining)
                node_indices[i].extend(remaining[start_idx:end_idx])
            
            # Create subset datasets
            train_partitions = [torch.utils.data.Subset(train_dataset, indices) for indices in node_indices]
        
        # Create dataloaders
        train_loaders = [torch.utils.data.DataLoader(
            partition, batch_size=batch_size, shuffle=True, num_workers=num_workers
        ) for partition in train_partitions]
        
        return train_loaders
    
    def initialize_nodes(self, train_loaders, test_loader):
        """Initialize all nodes with their local datasets."""
        self.generate_network()
        self.test_loader = test_loader
        
        for i in range(self.num_nodes):
            # Get neighbors from network graph
            neighbors = list(self.network.neighbors(i))
            
            # Create node with its dataset
            node = Node(
                node_id=i,
                neighbors=neighbors,
                local_data_loader=train_loaders[i],
                device=self.device
            )
            self.nodes.append(node)
    
    def gossip_round(self):
        """Execute one round of gossip communication."""
        round_comm_cost = 0
        
        # Each node trains its local model
        for node in self.nodes:
            node.train_local_model()
        
        # Nodes exchange models based on comm_prob
        for node in self.nodes:
            neighbors_params = []
            
            for neighbor_id in node.neighbors:
                # Probabilistic communication
                if random.random() < self.comm_prob:
                    neighbor = self.nodes[neighbor_id]
                    neighbors_params.append(neighbor.model.get_param_dict())
                    
                    # Account for communication cost
                    cost = node.receive_model(neighbor.model.get_param_dict())
                    round_comm_cost += cost
            
            # Aggregate models if any were received
            if neighbors_params:
                node.aggregate_models(neighbors_params)
        
        self.total_comm_cost += round_comm_cost
        self.round_comm_costs.append(round_comm_cost)
        
        # Calculate average loss and accuracy across all nodes
        avg_loss = np.mean([node.loss_history[-1] for node in self.nodes])
        avg_accuracy = np.mean([node.accuracy_history[-1] for node in self.nodes])
        
        self.global_loss_history.append(avg_loss)
        self.global_accuracy_history.append(avg_accuracy)
        
        return round_comm_cost, avg_loss, avg_accuracy
    
    def evaluate_global_performance(self):
        """Evaluate the performance of all nodes on the test set."""
        test_losses = []
        test_accuracies = []
        
        for node in self.nodes:
            loss, accuracy = node.evaluate(self.test_loader)
            test_losses.append(loss)
            test_accuracies.append(accuracy)
        
        return np.mean(test_losses), np.mean(test_accuracies), np.std(test_accuracies)
    
    def visualize_network(self):
        """Visualize the network structure."""
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.network, seed=42)
        nx.draw(self.network, pos, with_labels=True, node_color='lightblue', 
                node_size=500, edge_color='gray')
        plt.title("Decentralized Network Structure")
        plt.tight_layout()
        return plt
    
    def plot_convergence(self):
        """Plot the convergence of loss and accuracy over rounds."""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.global_loss_history)
        plt.title("Training Loss Convergence")
        plt.xlabel("Communication Round")
        plt.ylabel("Average Loss")
        
        plt.subplot(1, 3, 2)
        plt.plot(self.global_accuracy_history)
        plt.title("Training Accuracy")
        plt.xlabel("Communication Round")
        plt.ylabel("Average Accuracy (%)")
        
        plt.subplot(1, 3, 3)
        plt.plot(self.round_comm_costs)
        plt.title("Communication Cost per Round")
        plt.xlabel("Communication Round")
        plt.ylabel("Communication Cost (KB)")
        
        plt.tight_layout()
        return plt
    
    def compare_models(self):
        """Compare the performance of different nodes."""
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        for i, node in enumerate(self.nodes):
            plt.plot(node.loss_history, label=f"Node {i}")
        plt.title("Loss Convergence by Node")
        plt.xlabel("Communication Round")
        plt.ylabel("Loss")
        plt.legend()
        
        plt.subplot(1, 2, 2)
        for i, node in enumerate(self.nodes):
            plt.plot(node.accuracy_history, label=f"Node {i}")
        plt.title("Accuracy by Node")
        plt.xlabel("Communication Round")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        
        plt.tight_layout()
        return plt


# Run simulation
def run_simulation(num_nodes=5, num_rounds=20, iid=True, connectivity=0.3, comm_prob=0.5):
    """Run a complete simulation of decentralized federated learning on CIFAR-10."""
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2
    )
    
    # Create federated learning system
    fl_system = GossipFederatedLearning(
        num_nodes=num_nodes,
        connectivity_prob=connectivity,
        comm_prob=comm_prob,
        device=device
    )
    
    # Partition data and initialize nodes
    train_loaders = fl_system.partition_cifar10(trainset, iid=iid)
    fl_system.initialize_nodes(train_loaders, test_loader)
    
    # Visualize network
    fl_system.visualize_network()
    plt.savefig('network_structure.png')
    plt.close()
    
    # Start training
    print(f"Starting training with {num_nodes} nodes, IID={iid}")
    print(f"Network connectivity: {connectivity}, Communication probability: {comm_prob}")
    
    for round_num in range(num_rounds):
        start_time = time.time()
        comm_cost, train_loss, train_acc = fl_system.gossip_round()
        end_time = time.time()
        
        # Evaluate global performance every 5 rounds
        if round_num % 5 == 0 or round_num == num_rounds - 1:
            test_loss, test_acc, test_std = fl_system.evaluate_global_performance()
            print(f"Round {round_num+1}/{num_rounds} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Test Acc: {test_acc:.2f}±{test_std:.2f}%, "
                  f"Comm Cost: {comm_cost:.2f} KB, "
                  f"Time: {end_time - start_time:.2f}s")
        else:
            print(f"Round {round_num+1}/{num_rounds} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Comm Cost: {comm_cost:.2f} KB, "
                  f"Time: {end_time - start_time:.2f}s")
    
    # Plot convergence
    fl_system.plot_convergence()
    plt.savefig('convergence.png')
    plt.close()
    
    # Compare models
    fl_system.compare_models()
    plt.savefig('model_comparison.png')
    plt.close()
    
    # Final evaluation
    test_loss, test_acc, test_std = fl_system.evaluate_global_performance()
    print("\nFinal Results:")
    print(f"Number of nodes: {num_nodes}")
    print(f"Data distribution: {'IID' if iid else 'Non-IID'}")
    print(f"Network connectivity: {connectivity}")
    print(f"Communication probability: {comm_prob}")
    print(f"Test accuracy: {test_acc:.2f}±{test_std:.2f}%")
    print(f"Total communication cost: {fl_system.total_comm_cost:.2f} KB")
    
    return fl_system

# Experiment with different parameters
def run_experiments():
    """Run experiments with different parameters."""
    # Configuration
    num_nodes_list = [5, 10]
    iid_settings = [True, False]
    connectivity_list = [0.3, 0.7]
    comm_prob_list = [0.5, 0.8]
    
    results = []
    
    for num_nodes in num_nodes_list:
        for iid in iid_settings:
            for connectivity in connectivity_list:
                for comm_prob in comm_prob_list:
                    print(f"\n\n===== Running experiment with: =====")
                    print(f"Nodes: {num_nodes}, IID: {iid}, Connectivity: {connectivity}, Comm Prob: {comm_prob}")
                    
                    # Run for fewer rounds to save time during experiments
                    fl_system = run_simulation(
                        num_nodes=num_nodes,
                        num_rounds=10,  # Reduced for faster experimentation
                        iid=iid,
                        connectivity=connectivity,
                        comm_prob=comm_prob
                    )
                    
                    # Record results
                    test_loss, test_acc, test_std = fl_system.evaluate_global_performance()
                    results.append({
                        'num_nodes': num_nodes,
                        'iid': iid,
                        'connectivity': connectivity,
                        'comm_prob': comm_prob,
                        'test_acc': test_acc,
                        'test_std': test_std,
                        'total_comm_cost': fl_system.total_comm_cost
                    })
    
    # Print results summary
    print("\n===== Results Summary =====")
    print("| Nodes | IID | Connectivity | Comm Prob | Test Acc | Comm Cost |")
    print("|-------|-----|--------------|-----------|----------|-----------|")
    for result in results:
        print(f"| {result['num_nodes']:5d} | {'Yes' if result['iid'] else 'No':3s} | "
              f"{result['connectivity']:12.1f} | {result['comm_prob']:9.1f} | "
              f"{result['test_acc']:6.2f}% | {result['total_comm_cost']:9.2f} KB |")
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    # Plot test accuracy vs communication cost
    plt.subplot(2, 2, 1)
    for result in results:
        marker = 'o' if result['iid'] else 'x'
        color = 'blue' if result['num_nodes'] == 5 else 'red'
        plt.scatter(result['total_comm_cost'], result['test_acc'], 
                    marker=marker, color=color, s=100)
    plt.xlabel('Total Communication Cost (KB)')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Accuracy vs Communication Cost')
    
    # Plot test accuracy vs nodes
    plt.subplot(2, 2, 2)
    for iid in iid_settings:
        acc_values = [r['test_acc'] for r in results if r['iid'] == iid]
        nodes_values = [r['num_nodes'] for r in results if r['iid'] == iid]
        plt.scatter(nodes_values, acc_values, label=f"{'IID' if iid else 'Non-IID'}")
    plt.xlabel('Number of Nodes')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Accuracy vs Number of Nodes')
    plt.legend()
    
    # Plot test accuracy vs connectivity
    plt.subplot(2, 2, 3)
    for iid in iid_settings:
        acc_values = [r['test_acc'] for r in results if r['iid'] == iid]
        connectivity_values = [r['connectivity'] for r in results if r['iid'] == iid]
        plt.scatter(connectivity_values, acc_values, label=f"{'IID' if iid else 'Non-IID'}")
    plt.xlabel('Network Connectivity')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Accuracy vs Network Connectivity')
    plt.legend()
    
    # Plot test accuracy vs communication probability
    plt.subplot(2, 2, 4)
    for iid in iid_settings:
        acc_values = [r['test_acc'] for r in results if r['iid'] == iid]
        comm_prob_values = [r['comm_prob'] for r in results if r['iid'] == iid]
        plt.scatter(comm_prob_values, acc_values, label=f"{'IID' if iid else 'Non-IID'}")
    plt.xlabel('Communication Probability')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Accuracy vs Communication Probability')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('experiment_comparison.png')
    plt.close()
    
    return results

if __name__ == "__main__":
    # Run a single simulation
    fl_system = run_simulation(num_nodes=5, num_rounds=15, iid=True)
    
    # Uncomment to run full experiments (takes longer)
    # results = run_experiments()