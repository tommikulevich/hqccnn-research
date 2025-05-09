"""Hybrid QNN Parallel model."""
from typing import List

import torch
import torch.nn as nn
import pennylane as qml
from pennylane.qnn import TorchLayer


class QLayer(nn.Module):
    def __init__(self, num_qubits: int, num_qreps: int,
                 device_name: str = "default.qubit",
                 diff_method: str = "best"):
        super().__init__()

        self.num_qubits = num_qubits
        self.num_qreps = num_qreps

        self.qdevice = qml.device(device_name, wires=num_qubits)
        self.qnode = qml.QNode(self._circuit,
                               device=self.qdevice,
                               interface="torch",
                               diff_method=diff_method)

        weight_shapes = {
            "weights": qml.StronglyEntanglingLayers.shape(
                n_layers=num_qreps, n_wires=num_qubits
            )
        }

        self.qlayer = TorchLayer(self.qnode, weight_shapes)

    def _circuit(self, inputs, weights):
        qml.AngleEmbedding(inputs, rotation="X",
                           wires=list(range(self.num_qubits)))
        qml.StronglyEntanglingLayers(weights,
                                     wires=list(range(self.num_qubits)))
        return [qml.expval(qml.PauliY(wires=i))
                for i in range(self.num_qubits)]

    def forward(self, x):
        return self.qlayer(x)


class HQNN_Parallel(nn.Module):
    """HQNN_Parallel model combining classical CNN and quantum layers."""
    def __init__(self, in_channels: int, num_classes: int,
                 conv_channels: List[int], conv_kernels: List[int],
                 conv_strides: List[int], conv_paddings: List[int],
                 pool_sizes: List[int], num_qubits: int,
                 num_qlayers: int, num_qreps: int,
                 qdevice: str, qdiff_method: str) -> None:
        super().__init__()

        self.num_qubits = num_qubits
        self.num_qlayers = num_qlayers
        self.num_q_features = num_qlayers * num_qubits

        # Classical convolutional and linear layers
        n_conv = len(conv_channels)
        assert len(conv_kernels) == n_conv, \
            "conv_kernels must have the same length as conv_channels"
        assert len(conv_strides) == n_conv, \
            "conv_strides must have the same length as conv_channels"
        assert len(conv_paddings) == n_conv, \
            "conv_paddings must have the same length as conv_channels"
        assert len(pool_sizes) == n_conv, \
            "pool_sizes must have the same length as conv_channels"

        self.conv_blocks = nn.ModuleList()
        prev_ch = in_channels
        for i in range(n_conv):
            out_ch = conv_channels[i]
            k = conv_kernels[i]
            s = conv_strides[i]
            p = conv_paddings[i]
            pool_k = pool_sizes[i]

            block = nn.Sequential(
                nn.Conv2d(prev_ch, out_ch, kernel_size=k,
                          stride=s, padding=p),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=pool_k),
            )
            self.conv_blocks.append(block)
            prev_ch = out_ch

        self.flatten = nn.Flatten(start_dim=1)

        self.fc_block1 = nn.Sequential(
            nn.LazyLinear(out_features=self.num_q_features),
            nn.BatchNorm1d(num_features=self.num_q_features),
            nn.ReLU()
        )

        # Quantum layers
        self.qlayers = nn.ModuleList([
            QLayer(num_qubits=num_qubits, num_qreps=num_qreps,
                   device_name=qdevice, diff_method=qdiff_method)
            for _ in range(num_qlayers)
        ])

        # Classical linear layer
        self.fc_block2 = nn.Sequential(
            nn.LazyLinear(out_features=num_classes),
            nn.BatchNorm1d(num_features=num_classes),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        x = self.flatten(x)

        x = self.fc_block1(x)

        x_c = x.chunk(self.num_qlayers, dim=1)
        q = [layer(c) for layer, c in zip(self.qlayers, x_c)]
        x = torch.cat(q, dim=1)

        x = self.fc_block2(x)

        return x
