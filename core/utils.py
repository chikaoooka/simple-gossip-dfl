import torch
import torchvision
import torchvision.transforms as transforms
import json
import numpy as np
import random
import networkx as nx
import os
import datetime
import matplotlib.pyplot as plt

# Helper class for JSON serialization of numpy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)


def __partition_cifar10(
    train_dataset, num_nodes, iid=False, batch_size=64, num_workers=2, seed=42
):
    """Partition the CIFAR-10 dataset among nodes."""
    # TODO: unbalanced partitioning
    torch.manual_seed(seed)
    random.seed(seed)
    if iid:
        # IID partitioning
        partition_size = len(train_dataset) // num_nodes
        partition_lengths = [partition_size] * num_nodes
        partition_lengths[-1] += len(train_dataset) - sum(partition_lengths)

        train_partitions = torch.utils.data.random_split(
            train_dataset, partition_lengths
        )

    else:
        # Non-IID partitioning: each node gets 2 primary classes
        # Get all targets
        targets = torch.tensor(train_dataset.targets)

        # Partition by class
        class_indices = [torch.where(targets == i)[0] for i in range(10)]

        # Assign classes to nodes (each node gets 2 primary classes)
        node_class_map = {
            i: [(i * 2) % 10, (i * 2 + 1) % 10] for i in range(num_nodes)
        }
        # NOTE: This method can be improved to ensure no overlap among nodes when num_nodes > 10

        # Create partitions
        node_indices = [[] for _ in range(num_nodes)]
        for node_id, primary_classes in node_class_map.items():
            # Take 90% of the primary classes
            for cls in primary_classes:
                class_size = len(class_indices[cls])
                node_indices[node_id].extend(
                    class_indices[cls][: int(0.9 * class_size)]
                )
                class_indices[cls] = class_indices[cls][int(0.9 * class_size) :]

        # Distribute remaining data
        remaining = []
        for cls_idx in class_indices:
            remaining.extend(cls_idx)

        # Shuffle remaining
        random.shuffle(remaining)
        samples_per_node = len(remaining) // num_nodes

        for i in range(num_nodes):
            start_idx = i * samples_per_node
            end_idx = (
                start_idx + samples_per_node
                if i < num_nodes - 1
                else len(remaining)
            )
            node_indices[i].extend(remaining[start_idx:end_idx])

        # Create subset datasets
        train_partitions = [
            torch.utils.data.Subset(train_dataset, indices) for indices in node_indices
        ]

    # Create dataloaders
    train_loaders = [
        torch.utils.data.DataLoader(
            partition, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        for partition in train_partitions
    ]

    return train_loaders

# Prepare data
def prepare_CIFAR10(num_nodes, iid, seed=42):
    """Prepare CIFAR-10 dataset."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2
    )

    train_loaders = __partition_cifar10(train_dataset=trainset, num_nodes=num_nodes, iid=iid, batch_size=64, num_workers=2, seed=seed)

    return train_loaders, test_loader


# Create output directory structure
def create_output_dir():
    """Create output directory structure for experiment results."""
    # Create base output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"outputs/results_{timestamp}"
    os.makedirs(base_dir, exist_ok=True)

    # Create readme file
    # with open(os.path.join(base_dir, "README.md"), "w") as f:
    #     f.write("# Decentralized Federated Learning Experiment Results\n\n")
    #     f.write(
    #         f"Experiment run on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    #     )
    #     f.write("## Directory Structure\n")
    #     f.write(
    #         "- Each subdirectory contains results for a specific experimental condition\n"
    #     )
    #     f.write(
    #         "- Directory naming format: nodes{N}_iid{true/false}_conn{connectivity}_comm{probability}\n"
    #     )
    #     f.write(
    #         "- Contains network visualization, convergence plots, model comparisons, and raw data\n\n"
    #     )
    #     f.write("## Files\n")
    #     f.write("- `network_structure.png`: Visualization of the node network\n")
    #     f.write(
    #         "- `convergence.png`: Training loss, accuracy, and communication cost plots\n"
    #     )
    #     f.write("- `model_comparison.png`: Comparison of performance across nodes\n")
    #     f.write("- `results.json`: Raw experimental data\n")
    #     f.write("- `summary.txt`: Summary of experimental results\n\n")
    #     f.write("## Combined Results\n")
    #     f.write(
    #         "- `experiment_comparison.png`: Comparison across all experimental conditions\n"
    #     )
    #     f.write("- `results_summary.csv`: Summary metrics for all conditions\n")

    return base_dir


def plot_comparison(results, iid_settings, base_dir):
    """Plot comparison of loss and accuracy across nodes."""
    plt.figure(figsize=(10, 10))

    # Plot test accuracy vs communication cost
    plt.subplot(2, 3, 1)
    for result in results:
        marker = "o" if result["iid"] else "x"
        color = "blue" if result["num_nodes"] == 5 else "red"
        plt.scatter(
            result["total_comm_cost"],
            result["test_acc"],
            marker=marker,
            color=color,
            s=100,
        )
    plt.xlabel("Total Communication Cost (KB)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Accuracy vs Communication Cost")

    # Plot test accuracy vs nodes
    plt.subplot(2, 3, 2)
    for iid in iid_settings:
        acc_values = [r["test_acc"] for r in results if r["iid"] == iid]
        nodes_values = [r["num_nodes"] for r in results if r["iid"] == iid]
        plt.scatter(nodes_values, acc_values, label=f"{'IID' if iid else 'Non-IID'}")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Accuracy vs Number of Nodes")
    plt.legend()

    # Plot test accuracy vs local epochs
    plt.subplot(2, 3, 3)
    for iid in iid_settings:
        acc_values = [r["test_acc"] for r in results if r["iid"] == iid]
        epochs_values = [r["local_epochs"] for r in results if r["iid"] == iid]
        plt.scatter(epochs_values, acc_values, label=f"{'IID' if iid else 'Non-IID'}")
    plt.xlabel("Local Epochs")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Accuracy vs Local Epochs")
    plt.legend()

    # Plot test accuracy vs connectivity
    plt.subplot(2, 3, 4)
    for iid in iid_settings:
        acc_values = [r["test_acc"] for r in results if r["iid"] == iid]
        connectivity_values = [r["connectivity"] for r in results if r["iid"] == iid]
        plt.scatter(
            connectivity_values, acc_values, label=f"{'IID' if iid else 'Non-IID'}"
        )
    plt.xlabel("Network Connectivity")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Accuracy vs Network Connectivity")
    plt.legend()

    # Plot test accuracy vs communication probability
    plt.subplot(2, 3, 5)
    for iid in iid_settings:
        acc_values = [r["test_acc"] for r in results if r["iid"] == iid]
        comm_prob_values = [r["comm_prob"] for r in results if r["iid"] == iid]
        plt.scatter(
            comm_prob_values, acc_values, label=f"{'IID' if iid else 'Non-IID'}"
        )
    plt.xlabel("Communication Probability")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Accuracy vs Communication Probability")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "experiment_comparison.png"))
    plt.close()

    print(f"\nExperiment results saved to: {base_dir}")