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

    @qml.qnode(device, interface='torch', diff_method="parameter-shift")
    def quantum_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, rotation='X', wires=list(range(wires)))
        qml.StronglyEntanglingLayers(weights, wires=list(range(wires)))
        return [qml.expval(qml.PauliY(wires=w)) for w in range(wires)]

    return quantum_circuit


class HQNN_Parallel(nn.Module):
    """HQNN_Parallel model combining classical CNN and quantum layers."""
    def __init__(self, in_channels: int, num_classes: int,
                 conv_channels: List[int], conv_kernels: List[int],
                 conv_strides: List[int], conv_paddings: List[int],
                 pool_sizes: List[int], num_qubits: int,
                 num_qlayers: int, num_qrepetitions: int) -> None:
        super().__init__()

        self.num_qubits = num_qubits
        self.num_qlayers = num_qlayers
        self.num_q_features = num_qlayers * num_qubits

        # Classical convolutional and linear layers
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=conv_channels[0],
                kernel_size=conv_kernels[0],
                stride=conv_strides[0],
                padding=conv_paddings[0]
            ),
            nn.BatchNorm2d(num_features=conv_channels[0]),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=pool_sizes[0])

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv_channels[0],
                out_channels=conv_channels[1],
                kernel_size=conv_kernels[1],
                stride=conv_strides[1],
                padding=conv_paddings[1]
            ),
            nn.BatchNorm2d(num_features=conv_channels[1]),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=pool_sizes[1])

        self.flatten = nn.Flatten(start_dim=1)

        self.fc_block1 = nn.Sequential(
            nn.LazyLinear(out_features=self.num_q_features),
            nn.BatchNorm1d(num_features=self.num_q_features),
            nn.ReLU()
        )

        # Quantum layers
        weight_shapes = {"weights": qml.StronglyEntanglingLayers.shape(
            n_layers=num_qrepetitions, n_wires=num_qubits)}
        circuit = get_quantum_circuit(num_qubits)

        self.qlayers = nn.ModuleList([
            TorchLayer(circuit, weight_shapes)
            for _ in range(num_qlayers)
        ])

        # Classical linear layer
        self.fc_block2 = nn.Sequential(
            nn.LazyLinear(out_features=num_classes),
            nn.BatchNorm1d(num_features=num_classes),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.pool1(x)

        x = self.conv_block2(x)
        x = self.pool2(x)

        x = self.flatten(x)

        x = self.fc_block1(x)

        x_c = x.chunk(self.num_qlayers, dim=1)
        q = [layer(c) for layer, c in zip(self.qlayers, x_c)]
        x = torch.cat(q, dim=1)

        x = self.fc_block2(x)

        return x
