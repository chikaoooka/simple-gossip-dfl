import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn import SimpleCNN


class Node:
    def __init__(
        self,
        node_id,
        model,
        neighbors,
        local_data_loader,
        device="cpu",
        local_epochs=1,
        learning_rate=0.01,
        comm_cost_per_kb=1,
        seed=42,
    ):
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
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=self.learning_rate, momentum=0.9
        )

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
        param_size_bytes = sum(
            p.nelement() * p.element_size() for p in self.model.parameters()
        )
        comm_cost = (param_size_bytes / 1024) * self.comm_cost_per_kb  # Convert to KB
        self.comm_cost_log.append(comm_cost)
        return comm_cost

    def aggregate_models(self, neighbors_params_list):
        """Aggregate model parameters from neighbors."""
        # TODO: Implement FedAvg
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


# ---------------------------------------------------------------------------- #
#    Implementation of ClippedGossip from He L, Karimireddy S et al. (2022)    #
# ---------------------------------------------------------------------------- #


def clip(v, tau):
    v_norm = torch.norm(v)
    scale = min(1, tau / v_norm)
    if torch.isnan(v_norm):
        return 0
    return v * scale


def bucketing(inputs):
    import numpy as np

    s = 2
    indices = list(range(len(inputs)))
    np.random.shuffle(indices)
    T = int(np.ceil(len(inputs) / s))

    reshuflled_inputs = []
    for t in range(T):
        indices_slice = indices[t * s : (t + 1) * s]
        g_bar = sum(inputs[i] for i in indices_slice) / len(indices_slice)
        reshuflled_inputs.append(g_bar)
    return reshuflled_inputs


class ClippedGossipNode(Node):
    # TODO: Implement other mixing matrices
    def __init__(self, local_weight, neighbor_weight, clip_threshold=1, **kwargs):
        super().__init__(**kwargs)
        self.clip_threshold = clip_threshold
        self.local_weight = local_weight
        self.neighbor_weight = neighbor_weight

    def aggregate_models(self, neighbors_params_list):
        """Aggregate model parameters from neighbors with clipping."""
        if not neighbors_params_list:
            return

        if self.clip_threshold is None:
            distances = [
                (n - self.model.get_param_dict()) for n in neighbors_params_list
            ]
            if len(distances) >= 2:
                self.clip_threshold = sorted(distances)[-2]
            else:
                self.clip_threshold = distances[-1]

        aggregated_params = {}
        for param_name in neighbors_params_list[0].keys():
            stacked_params = []
            local_param = self.model.get_param_dict()[param_name]
            for i in range(len(neighbors_params_list)):
                neighbor_param = neighbors_params_list[i][param_name]
                stacked_params.append(
                    self.neighbor_weight
                    * (
                        clip(neighbor_param - local_param, self.clip_threshold)
                        + local_param
                    )
                )
            stacked_params.append(self.local_weight * local_param)
            stacked_params = torch.stack(stacked_params)
            aggregated_params[param_name] = torch.sum(stacked_params, dim=0)

        self.model.set_param_dict(aggregated_params)
