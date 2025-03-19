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
import os
import datetime
import json

from models.cnn import SimpleCNN
from utils.utils import NumpyEncoder

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class GossipFederatedLearning:
    def __init__(self, num_nodes, connectivity_prob=0.3, comm_prob=0.5, device='cpu', output_dir='results'):
        """Initialize the decentralized federated learning system."""
        self.num_nodes = num_nodes
        self.connectivity_prob = connectivity_prob
        self.comm_prob = comm_prob
        self.device = device
        self.output_dir = output_dir
        self.nodes = []
        self.network = None
        self.total_comm_cost = 0
        self.global_loss_history = []
        self.global_accuracy_history = []
        self.test_loss_history = []
        self.test_accuracy_history = []
        self.test_accuracy_std = []
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
        #TODO: unbalanced partitioning
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
            #NOTE: This method can be improved to ensure no overlap among nodes when num_nodes > 10

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
                model=SimpleCNN(seed=SEED),
                neighbors=neighbors,
                local_data_loader=train_loaders[i],
                device=self.device,
                seed=SEED
            )
            self.nodes.append(node)
    
    def gossip_round(self):
        """Execute one round of gossip communication."""
        #TODO: parallelize this loop
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
    
    def visualize_network(self, save_path):
        """Visualize the network structure."""
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.network, seed=42)
        nx.draw(self.network, pos, with_labels=True, node_color='lightblue', 
                node_size=500, edge_color='gray')
        plt.title("Decentralized Network Structure")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def plot_convergence(self, save_path):
        """Plot the convergence of loss and accuracy over rounds, including test metrics and standard deviation."""
        plt.figure(figsize=(15, 10))
        
        # Training Loss
        plt.subplot(2, 2, 1)
        plt.plot(self.global_loss_history, label="Train Loss", color="blue")
        if hasattr(self, 'test_loss_history'):
            plt.plot(self.test_loss_history, label="Test Loss", color="orange")
            if hasattr(self, 'test_loss_std'):
                plt.fill_between(
                    range(len(self.test_loss_history)),
                    np.array(self.test_loss_history) - np.array(self.test_loss_std),
                    np.array(self.test_loss_history) + np.array(self.test_loss_std),
                    color="orange", alpha=0.2, label="Test Loss Std Dev"
                )
        plt.title("Loss Convergence")
        plt.xlabel("Communication Round")
        plt.ylabel("Loss")
        plt.legend()

        # Training Accuracy
        plt.subplot(2, 2, 2)
        plt.plot(self.global_accuracy_history, label="Train Accuracy", color="blue")
        if hasattr(self, 'test_accuracy_history'):
            plt.plot(self.test_accuracy_history, label="Test Accuracy", color="orange")
            if hasattr(self, 'test_accuracy_std'):
                plt.fill_between(
                    range(len(self.test_accuracy_history)),
                    np.array(self.test_accuracy_history) - np.array(self.test_accuracy_std),
                    np.array(self.test_accuracy_history) + np.array(self.test_accuracy_std),
                    color="orange", alpha=0.2, label="Test Accuracy Std Dev"
                )
        plt.title("Accuracy Convergence")
        plt.xlabel("Communication Round")
        plt.ylabel("Accuracy (%)")
        plt.legend()

        # Communication Cost
        plt.subplot(2, 2, 3)
        plt.plot(self.round_comm_costs, label="Communication Cost", color="green")
        plt.title("Communication Cost per Round")
        plt.xlabel("Communication Round")
        plt.ylabel("Communication Cost (KB)")
        plt.legend()

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def compare_models(self, save_path):
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
        plt.savefig(save_path)
        plt.close()
    
    def save_results(self, exp_dir):
        """Save experiment results to files."""
        results = {
            'global_loss': self.global_loss_history,
            'global_accuracy': self.global_accuracy_history,
            'round_comm_costs': self.round_comm_costs,
            'total_comm_cost': self.total_comm_cost,
            'network_edges': list(self.network.edges()),
            'nodes_data': [
                {
                    'node_id': node.node_id,
                    'neighbors': node.neighbors,
                    'loss_history': node.loss_history,
                    'accuracy_history': node.accuracy_history,
                    'comm_cost_log': node.comm_cost_log
                }
                for node in self.nodes
            ]
        }
        
        with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)


# Create output directory structure
def create_output_dir():
    """Create output directory structure for experiment results."""
    # Create base output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"outputs/federated_learning_results_{timestamp}"
    os.makedirs(base_dir, exist_ok=True)
    
    # Create readme file
    with open(os.path.join(base_dir, "README.md"), "w") as f:
        f.write("# Decentralized Federated Learning Experiment Results\n\n")
        f.write(f"Experiment run on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Directory Structure\n")
        f.write("- Each subdirectory contains results for a specific experimental condition\n")
        f.write("- Directory naming format: nodes{N}_iid{true/false}_conn{connectivity}_comm{probability}\n")
        f.write("- Contains network visualization, convergence plots, model comparisons, and raw data\n\n")
        f.write("## Files\n")
        f.write("- `network_structure.png`: Visualization of the node network\n")
        f.write("- `convergence.png`: Training loss, accuracy, and communication cost plots\n")
        f.write("- `model_comparison.png`: Comparison of performance across nodes\n")
        f.write("- `results.json`: Raw experimental data\n")
        f.write("- `summary.txt`: Summary of experimental results\n\n")
        f.write("## Combined Results\n")
        f.write("- `experiment_comparison.png`: Comparison across all experimental conditions\n")
        f.write("- `results_summary.csv`: Summary metrics for all conditions\n")
    
    return base_dir


# Run simulation
def run_simulation(num_nodes=5, num_rounds=20, iid=True, connectivity=0.3, comm_prob=0.5, base_dir="results"):
    """Run a complete simulation of decentralized federated learning on CIFAR-10."""
    # Create experiment directory
    exp_name = f"nodes{num_nodes}_iid{'true' if iid else 'false'}_conn{connectivity}_comm{comm_prob}"
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Log experiment parameters
    with open(os.path.join(exp_dir, "params.txt"), "w") as f:
        f.write(f"Number of nodes: {num_nodes}\n")
        f.write(f"IID data distribution: {iid}\n")
        f.write(f"Network connectivity: {connectivity}\n")
        f.write(f"Communication probability: {comm_prob}\n")
        f.write(f"Number of rounds: {num_rounds}\n")
    
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
        device=device,
        output_dir=exp_dir
    )
    
    # Partition data and initialize nodes
    train_loaders = fl_system.partition_cifar10(trainset, iid=iid)
    fl_system.initialize_nodes(train_loaders, test_loader)
    
    # Visualize network
    fl_system.visualize_network(os.path.join(exp_dir, 'network_structure.png'))
    
    # Log file for progress
    log_file = os.path.join(exp_dir, "training_log.txt")
    with open(log_file, "w") as f:
        f.write(f"Starting training with {num_nodes} nodes, IID={iid}\n")
        f.write(f"Network connectivity: {connectivity}, Communication probability: {comm_prob}\n\n")
    
    # Start training
    print(f"Starting training with {num_nodes} nodes, IID={iid}")
    print(f"Network connectivity: {connectivity}, Communication probability: {comm_prob}")
    
    for round_num in range(num_rounds):
        # start_time = time.time()
        comm_cost, train_loss, train_acc = fl_system.gossip_round()
        # end_time = time.time()
        
        # Log progress
        log_message = f"Round {round_num+1}/{num_rounds} - " \
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, " \
                      f"Comm Cost: {comm_cost:.2f} KB" \
                    #   f", Time: {end_time - start_time:.2f}s"
        
        # Evaluate global performance
        # if round_num % 5 == 0 or round_num == num_rounds - 1:
        test_loss, test_acc, test_std = fl_system.evaluate_global_performance()
        log_message += f", Test Acc: {test_acc:.2f}±{test_std:.2f}%"
        fl_system.test_loss_history.append(test_loss)
        fl_system.test_accuracy_history.append(test_acc)
        fl_system.test_accuracy_std.append(test_std)
        
        print(log_message)
        with open(log_file, "a") as f:
            f.write(log_message + "\n")
    
    # Plot convergence
    fl_system.plot_convergence(os.path.join(exp_dir, 'convergence.png'))
    
    # Compare models
    fl_system.compare_models(os.path.join(exp_dir, 'model_comparison.png'))
    
    # Save results
    fl_system.save_results(exp_dir)
    
    # Final evaluation
    test_loss, test_acc, test_std = fl_system.evaluate_global_performance()
    
    # Write summary
    with open(os.path.join(exp_dir, "summary.txt"), "w") as f:
        f.write("# Experiment Summary\n\n")
        f.write(f"Number of nodes: {num_nodes}\n")
        f.write(f"Data distribution: {'IID' if iid else 'Non-IID'}\n")
        f.write(f"Network connectivity: {connectivity}\n")
        f.write(f"Communication probability: {comm_prob}\n")
        f.write(f"Number of rounds: {num_rounds}\n\n")
        f.write(f"Final test accuracy: {test_acc:.2f}±{test_std:.2f}%\n")
        f.write(f"Total communication cost: {fl_system.total_comm_cost:.2f} KB\n")
        f.write(f"Average communication cost per round: {fl_system.total_comm_cost/num_rounds:.2f} KB\n")
    
    print("\nFinal Results:")
    print(f"Number of nodes: {num_nodes}")
    print(f"Data distribution: {'IID' if iid else 'Non-IID'}")
    print(f"Network connectivity: {connectivity}")
    print(f"Communication probability: {comm_prob}")
    print(f"Test accuracy: {test_acc:.2f}±{test_std:.2f}%")
    print(f"Total communication cost: {fl_system.total_comm_cost:.2f} KB")
    
    return fl_system, test_acc, test_std, fl_system.total_comm_cost


# Experiment with different parameters
def run_experiments():
    """Run experiments with different parameters."""
    # Create base output directory
    base_dir = create_output_dir()
    
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
                    
                    # Run simulation
                    _, test_acc, test_std, total_comm_cost = run_simulation(
                        num_nodes=num_nodes,
                        num_rounds=100,  # Reduced for faster experimentation
                        iid=iid,
                        connectivity=connectivity,
                        comm_prob=comm_prob,
                        base_dir=base_dir
                    )
                    
                    # Record results
                    results.append({
                        'num_nodes': num_nodes,
                        'iid': iid,
                        'connectivity': connectivity,
                        'comm_prob': comm_prob,
                        'test_acc': test_acc,
                        'test_std': test_std,
                        'total_comm_cost': total_comm_cost
                    })
    
    # Save results summary as CSV
    with open(os.path.join(base_dir, "results_summary.csv"), "w") as f:
        f.write("num_nodes,iid,connectivity,comm_prob,test_acc,test_std,total_comm_cost\n")
        for result in results:
            f.write(f"{result['num_nodes']},{result['iid']},{result['connectivity']},"
                    f"{result['comm_prob']},{result['test_acc']},{result['test_std']},"
                    f"{result['total_comm_cost']}\n")
    
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
    plt.savefig(os.path.join(base_dir, 'experiment_comparison.png'))
    plt.close()
    
    print(f"\nExperiment results saved to: {base_dir}")
    return results, base_dir


if __name__ == "__main__":
    # Run the experiments
    # results, output_dir = run_experiments()
    
    # To run a single simulation instead:
    base_dir = create_output_dir()
    fl_system, test_acc, test_std, total_comm_cost = run_simulation(
        num_nodes=10, 
        num_rounds=50, 
        iid=False, 
        connectivity=0.7, 
        comm_prob=0.8,
        base_dir=base_dir
    )