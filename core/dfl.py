import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random
import os
import json

from core.utils import NumpyEncoder
from core.node import Node
from models.cnn import SimpleCNN


class DFL:
    def __init__(self, num_nodes, connectivity_prob, device, output_dir, seed=42):
        self.num_nodes = num_nodes
        self.connectivity_prob = connectivity_prob
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
        self.seed = seed
        random.seed(seed)

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
                model=SimpleCNN(seed=self.seed),
                neighbors=neighbors,
                local_data_loader=train_loaders[i],
                device=self.device,
                seed=self.seed,
            )
            self.nodes.append(node)

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
        nx.draw(
            self.network,
            pos,
            with_labels=True,
            node_color="lightblue",
            node_size=500,
            edge_color="gray",
        )
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
        if hasattr(self, "test_loss_history"):
            plt.plot(self.test_loss_history, label="Test Loss", color="orange")
            if hasattr(self, "test_loss_std"):
                plt.fill_between(
                    range(len(self.test_loss_history)),
                    np.array(self.test_loss_history) - np.array(self.test_loss_std),
                    np.array(self.test_loss_history) + np.array(self.test_loss_std),
                    color="orange",
                    alpha=0.2,
                    label="Test Loss Std Dev",
                )
        plt.title("Loss Convergence")
        plt.xlabel("Communication Round")
        plt.ylabel("Loss")
        plt.legend()

        # Training Accuracy
        plt.subplot(2, 2, 2)
        plt.plot(self.global_accuracy_history, label="Train Accuracy", color="blue")
        if hasattr(self, "test_accuracy_history"):
            plt.plot(self.test_accuracy_history, label="Test Accuracy", color="orange")
            if hasattr(self, "test_accuracy_std"):
                plt.fill_between(
                    range(len(self.test_accuracy_history)),
                    np.array(self.test_accuracy_history)
                    - np.array(self.test_accuracy_std),
                    np.array(self.test_accuracy_history)
                    + np.array(self.test_accuracy_std),
                    color="orange",
                    alpha=0.2,
                    label="Test Accuracy Std Dev",
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
            "global_loss": self.global_loss_history,
            "global_accuracy": self.global_accuracy_history,
            "round_comm_costs": self.round_comm_costs,
            "total_comm_cost": self.total_comm_cost,
            "network_edges": list(self.network.edges()),
            "nodes_data": [
                {
                    "node_id": node.node_id,
                    "neighbors": node.neighbors,
                    "loss_history": node.loss_history,
                    "accuracy_history": node.accuracy_history,
                    "comm_cost_log": node.comm_cost_log,
                }
                for node in self.nodes
            ],
        }

        with open(os.path.join(exp_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
