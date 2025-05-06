"""Hybrid QNN Parallel model."""
from typing import List, Callable

import torch
import torch.nn as nn
import pennylane as qml
from pennylane.qnn import TorchLayer


dev: qml.device = None


def get_quantum_circuit(wires: int) -> Callable:
    """Wrap a PennyLane device into a torch-compatible QNode."""
    device = qml.device("default.qubit", wires=wires) if dev is None else dev

    @qml.qnode(device, interface='torch')
    def quantum_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=list(range(wires)))
        qml.BasicEntanglerLayers(weights, wires=list(range(wires)))
        return [qml.expval(qml.PauliZ(wires=w)) for w in range(wires)]

    return quantum_circuit


class HQNN_Parallel(nn.Module):
    """HQNN_Parallel model combining classical CNN and quantum layers."""
    def __init__(self, in_channels: int, num_classes: int,
                 conv_channels: List[int], kernel_sizes: List[int],
                 pool_sizes: List[int], fc_sizes: List[int],
                 num_qubits: int, num_qlayers: int) -> None:
        super().__init__()

        # Classical convolutional and linear layers
        self.conv1 = nn.Conv2d(in_channels, conv_channels[0],
                               kernel_sizes[0], padding=kernel_sizes[0] // 2)
        self.bn1 = nn.BatchNorm2d(conv_channels[0])
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=pool_sizes[0])

        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1],
                               kernel_sizes[1], padding=kernel_sizes[1] // 2)
        self.bn2 = nn.BatchNorm2d(conv_channels[1])
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=pool_sizes[1])

        self.flatten = nn.Flatten(start_dim=1)

        self.fc1 = nn.LazyLinear(fc_sizes[0])  # To not calculate in_features
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(fc_sizes[0], fc_sizes[1])
        self.relu4 = nn.ReLU()

        # Quantum layers
        self.num_qubits = num_qubits
        weight_shapes = {"weights": (num_qlayers, num_qubits)}
        circuit = get_quantum_circuit(num_qubits)

        assert fc_sizes[1] % num_qubits == 0
        self.qnetlayers = nn.ModuleList([
            TorchLayer(circuit, weight_shapes)
            for _ in range(fc_sizes[1] // num_qubits)
        ])

        # Classical linear layer
        self.fc3 = nn.Linear(fc_sizes[1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.flatten(x)

        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))

        x_chunks = torch.split(x, self.num_qubits, dim=1)
        q_outs = [
            layer(chunk)
            for layer, chunk in zip(self.qnetlayers, x_chunks)
        ]
        x = torch.cat(q_outs, dim=1)

        x = self.fc3(x)

        return x
