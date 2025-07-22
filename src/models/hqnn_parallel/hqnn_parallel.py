"""Hybrid QNN Parallel model."""
import math
from enum import Enum
from typing import List, Tuple

import torch
import torch.nn as nn
import pennylane as qml
from pennylane.qnn import TorchLayer

try:
    from utils.seeds import get_seed
except ImportError:
    print("Warning: problem with importing utils.seeds. \
        Set seed to 42 in RandomLayer (if it is used)")

    def get_seed() -> int:
        return 42


# TODO: make it more universal (every type has many different params)
class AnsatzType(str, Enum):
    SEL = "sel"     # StronglyEntanglingLayers
    BEL = "bel"     # BasicEntanglerLayers
    RL = "rl"       # RandomLayers


class QLayer(nn.Module):
    def __init__(self, ansatz_type: AnsatzType,
                 num_qubits: int, num_qreps: int,
                 device_name: str = "default.qubit",
                 diff_method: str = "best") -> None:
        super().__init__()

        # Some way to apply string instead of enum
        if isinstance(ansatz_type, str):
            try:
                ansatz_type = AnsatzType[ansatz_type.upper()]
            except KeyError:
                raise ValueError(f"Unknown ansatz type: {ansatz_type}")

        self.ansatz_type = ansatz_type
        self.num_qubits = num_qubits
        self.num_qreps = num_qreps

        self.qdevice = qml.device(device_name, wires=num_qubits)
        self.qnode = qml.QNode(self._circuit,
                               device=self.qdevice,
                               interface="torch",
                               diff_method=diff_method)

        if ansatz_type is AnsatzType.SEL:
            weight_shapes = {
                "weights": qml.StronglyEntanglingLayers.shape(
                    n_layers=num_qreps, n_wires=num_qubits
                )
            }
        elif ansatz_type is AnsatzType.BEL:
            weight_shapes = {
                "weights": qml.BasicEntanglerLayers.shape(
                    n_layers=num_qreps, n_wires=num_qubits
                )
            }
        elif ansatz_type is AnsatzType.RL:
            weight_shapes = {
                "weights": qml.RandomLayers.shape(
                    n_layers=num_qreps, n_rotations=num_qubits
                )
            }
        else:
            raise RuntimeError(f"Unknown ansatz type: {ansatz_type}")

        self.qlayer = TorchLayer(self.qnode, weight_shapes)

    def _circuit(self, inputs: torch.Tensor, weights: torch.Tensor) \
            -> List[torch.Tensor]:
        # Encoding
        qml.AngleEmbedding(inputs, rotation="X",
                           wires=list(range(self.num_qubits)))

        # Ansatz
        if self.ansatz_type is AnsatzType.SEL:
            qml.StronglyEntanglingLayers(weights,
                                         wires=list(range(self.num_qubits)),
                                         imprimitive=qml.CNOT)
        elif self.ansatz_type is AnsatzType.BEL:
            qml.BasicEntanglerLayers(weights,
                                     wires=list(range(self.num_qubits)),
                                     rotation=qml.RZ)
        elif self.ansatz_type is AnsatzType.RL:
            qml.RandomLayers(weights,
                             wires=list(range(self.num_qubits)),
                             imprimitive=qml.CNOT,
                             seed=get_seed())
        else:
            raise RuntimeError(f"Undefined AnsatzType: {self.ansatz_type}")

        # Measure
        return [qml.expval(qml.PauliY(wires=i))
                for i in range(self.num_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qlayer(x)


class QLayersParallel(nn.Module):
    """Parallel application of multiple QLayer modules."""
    def __init__(self, ansatz_type: AnsatzType,
                 num_layers: int, num_qubits: int, num_qreps: int,
                 device_name: str, diff_method: str) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            QLayer(ansatz_type=ansatz_type,
                   num_qubits=num_qubits, num_qreps=num_qreps,
                   device_name=device_name, diff_method=diff_method)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Chunk input, apply each QLayer, and concat outputs
        chunks = x.chunk(self.num_layers, dim=1)
        outs = [layer(c) for layer, c in zip(self.layers, chunks)]
        return torch.cat(outs, dim=1)


class TanhScale(nn.Module):
    """tanh(x) * scale"""
    def __init__(self, scale: float = 1) -> None:
        super().__init__()
        self.scale = scale
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tanh(x) * self.scale


class HQNN_Parallel(nn.Module):
    """HQNN_Parallel model combining classical CNN and quantum layers."""
    def __init__(self, in_channels: int, num_classes: int,
                 input_size: Tuple[int, int, int],
                 conv_channels: List[int], conv_kernels: List[int],
                 conv_strides: List[int], conv_paddings: List[int],
                 pool_sizes: List[int], ansatz_type: AnsatzType,
                 num_qubits: int, num_qlayers: int, num_qreps: int,
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
        _, h, w = input_size
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

            # After conv
            h = (h + 2 * p - k) // s + 1
            w = (w + 2 * p - k) // s + 1

            # After pool
            h = (h - pool_k) // pool_k + 1
            w = (w - pool_k) // pool_k + 1

        self.flatten = nn.Flatten(start_dim=1)
        flat_dim = prev_ch * h * w

        self.fc_block1 = nn.Sequential(
            nn.Linear(in_features=flat_dim, out_features=self.num_q_features),
            nn.BatchNorm1d(num_features=self.num_q_features),
            nn.ReLU()
        )

        # Quantum layers
        self.tanh_scale = TanhScale(scale=math.pi/2)
        self.qlayers = QLayersParallel(
            ansatz_type=ansatz_type,
            num_layers=num_qlayers,
            num_qubits=num_qubits,
            num_qreps=num_qreps,
            device_name=qdevice,
            diff_method=qdiff_method,
        )

        # Classical linear layer
        self.fc_block2 = nn.Linear(in_features=self.num_q_features,
                                   out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        x = self.flatten(x)
        x = self.fc_block1(x)

        x = self.tanh_scale(x)
        x = self.qlayers(x)

        x = self.fc_block2(x)

        return x
