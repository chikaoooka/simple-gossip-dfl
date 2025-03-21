import torch
import numpy as np
import random
import os

from core.dfl import DFL
from core.utils import create_output_dir, prepare_CIFAR10, plot_comparison


# Set random seed for reproducibility
SEED = 42


class GossipFederatedLearning(DFL):
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


# Run simulation
def run_simulation(
    num_nodes=5,
    num_rounds=20,
    iid=True,
    connectivity=0.3,
    comm_prob=0.5,
    local_epochs=1,
    base_dir="results",
):
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

    # Create federated learning system
    fl_system = GossipFederatedLearning(
        num_nodes=num_nodes,
        connectivity_prob=connectivity,
        comm_prob=comm_prob,
        device=device,
        local_epochs=local_epochs,
        output_dir=exp_dir,
    )

    # Prepare data loaders
    train_loaders, test_loader = prepare_CIFAR10(
        num_nodes=num_nodes, iid=iid, seed=SEED
    )
    fl_system.initialize_nodes(train_loaders, test_loader)

    # Visualize network
    fl_system.visualize_network(os.path.join(exp_dir, "network_structure.png"))

    # Log file for progress
    log_file = os.path.join(exp_dir, "training_log.txt")
    with open(log_file, "w") as f:
        f.write(f"Starting training with {num_nodes} nodes, IID={iid}\n")
        f.write(
            f"Network connectivity: {connectivity}, Communication probability: {comm_prob}\n\n"
        )

    # Start training
    print(f"Starting training with {num_nodes} nodes, IID={iid}")
    print(
        f"Network connectivity: {connectivity}, Communication probability: {comm_prob}"
    )

    for round_num in range(num_rounds):
        # start_time = time.time()
        comm_cost, train_loss, train_acc = fl_system.gossip_round()
        # end_time = time.time()

        # Log progress
        log_message = (
            f"Round {round_num+1}/{num_rounds} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Comm Cost: {comm_cost:.2f} KB"
        )  #   f", Time: {end_time - start_time:.2f}s"

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

    # Plot and save results
    fl_system.plot_convergence(os.path.join(exp_dir, "convergence.png"))
    fl_system.compare_models(os.path.join(exp_dir, "model_comparison.png"))
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
        f.write(
            f"Average communication cost per round: {fl_system.total_comm_cost/num_rounds:.2f} KB\n"
        )

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
    num_nodes_list = [10]
    iid_settings = [True, False]
    connectivity_list = [0.3, 0.7]
    comm_prob_list = [0.8]
    local_epoch_list = [1, 3, 5]

    results = []

    for num_nodes in num_nodes_list:
        for iid in iid_settings:
            for connectivity in connectivity_list:
                for comm_prob in comm_prob_list:
                    for local_epochs in local_epoch_list:
                        print(f"\n\n===== Running experiment with: =====")
                        print(
                            f"Nodes: {num_nodes}, IID: {iid}, Connectivity: {connectivity}, Comm Prob: {comm_prob}, Local Epochs: {local_epochs}"
                        )

                        # Run simulation
                        _, test_acc, test_std, total_comm_cost = run_simulation(
                            num_nodes=num_nodes,
                            num_rounds=100,  # Reduced for faster experimentation
                            iid=iid,
                            connectivity=connectivity,
                            comm_prob=comm_prob,
                            local_epochs=local_epochs,
                            base_dir=base_dir,
                        )

                        # Record results
                        results.append(
                            {
                                "num_nodes": num_nodes,
                                "iid": iid,
                                "connectivity": connectivity,
                                "comm_prob": comm_prob,
                                "local_epochs": local_epochs,
                                "test_acc": test_acc,
                                "test_std": test_std,
                                "total_comm_cost": total_comm_cost,
                            }
                        )

    # Save results summary as CSV
    with open(os.path.join(base_dir, "results_summary.csv"), "w") as f:
        f.write(
            "num_nodes,iid,connectivity,comm_prob,test_acc,test_std,total_comm_cost\n"
        )
        for result in results:
            f.write(
                f"{result['num_nodes']},{result['iid']},{result['connectivity']},"
                f"{result['comm_prob']},{result['test_acc']},{result['test_std']},"
                f"{result['total_comm_cost']}\n"
            )

    # Print results summary
    print("\n===== Results Summary =====")
    print("| Nodes | IID | Connectivity | Comm Prob | Test Acc | Comm Cost |")
    print("|-------|-----|--------------|-----------|----------|-----------|")
    for result in results:
        print(
            f"| {result['num_nodes']:5d} | {'Yes' if result['iid'] else 'No':3s} | "
            f"{result['connectivity']:12.1f} | {result['comm_prob']:9.1f} | "
            f"{result['test_acc']:6.2f}% | {result['total_comm_cost']:9.2f} KB |"
        )

    # Plot comparison
    plot_comparison(results, iid_settings, base_dir)

    return results, base_dir


if __name__ == "__main__":
    # Run the experiments
    # results, output_dir = run_experiments()

    # To run a single simulation instead:
    base_dir = create_output_dir()
    fl_system, test_acc, test_std, total_comm_cost = run_simulation(
        num_nodes=3,
        num_rounds=3,
        iid=True,
        connectivity=0.7,
        comm_prob=0.8,
        base_dir=base_dir,
    )
