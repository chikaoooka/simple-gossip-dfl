import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random
import os
import json

from core.utils import NumpyEncoder
from core.node import Node, ClippedGossipNode
from models.cnn import SimpleCNN


class _DflBase:
    def __init__(
        self,
        num_nodes,
        connectivity_prob,
        device="cpu",
        local_epochs=1,
        output_dir="outputs",
        seed=42,
    ):
        self.num_nodes = num_nodes
        self.connectivity_prob = connectivity_prob
        self.device = device
        self.local_epochs = local_epochs
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
        np.random.seed(seed)

    def get_max_degree(self) -> int:
        """Get the maximum degree of the network."""
        return max([self.network.degree(node) for node in self.network.nodes])

    def generate_network(self) -> nx.Graph:
        """Generate a random network using Erdos-Renyi model."""
        self.network = nx.erdos_renyi_graph(
            self.num_nodes, self.connectivity_prob, seed=self.seed
        )

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
                local_epochs=self.local_epochs,
                seed=self.seed,
            )
            self.nodes.append(node)

    def evaluate_global_performance(self) -> tuple:
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
        pos = nx.spring_layout(self.network, seed=self.seed)
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


class GossipFederatedLearning(_DflBase):
    def __init__(self, comm_prob, **kwargs):
        super().__init__(**kwargs)
        self.comm_prob = comm_prob
        random.seed(self.seed)

    def gossip_round(self):
        """Execute one round of gossip communication."""
        # TODO: parallelize this loop
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


class ClippedGossipFL(_DflBase):
    def __init__(self, comm_prob, **kwargs):
        super().__init__(**kwargs)
        self.comm_prob = comm_prob
        random.seed(self.seed)

    def initialize_nodes(self, train_loaders, test_loader):
        self.generate_network()
        self.test_loader = test_loader

        for i in range(self.num_nodes):
            # Get neighbors from network graph
            neighbors = list(self.network.neighbors(i))
            neighbor_weight, local_weight = self.calculate_weights()
            # print ("neighbor_weight: ", neighbor_weight)
            # print ("local_weight: ", local_weight)

            # Create node with its dataset
            node = ClippedGossipNode(
                neighbor_weight=neighbor_weight,
                local_weight=local_weight,
                clip_threshold=1.0,
                node_id=i,
                model=SimpleCNN(seed=self.seed),
                neighbors=neighbors,
                local_data_loader=train_loaders[i],
                device=self.device,
                local_epochs=self.local_epochs,
                seed=self.seed,
            )
            self.nodes.append(node)

    def calculate_weights(self) -> tuple:
        """Calculate local weights for each node based on its degree."""
        neighbor_weight = 1 / (self.get_max_degree() + 1)
        local_weitght = 1 - neighbor_weight * (self.num_nodes - 1)
        return neighbor_weight, local_weitght

    def gossip_round(self):
        """Execute one round of gossip communication."""
        # TODO: parallelize this loop
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
