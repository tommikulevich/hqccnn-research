"""Hybrid QNN Quanvolutional model."""
import math
from enum import Enum
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class Quanv(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int,
                 stride: int, padding: int,
                 ansatz_type: AnsatzType,
                 num_qubits: int, num_qreps: int,
                 split_channels: bool = False,
                 device_name: str = "default.qubit",
                 diff_method: str = "best") -> None:
        super().__init__()

        # Some way to apply string instead of enum
        if isinstance(ansatz_type, str):
            try:
                ansatz_type = AnsatzType[ansatz_type.upper()]
            except KeyError:
                raise ValueError(f"Unknown ansatz type: {ansatz_type}")

        self.in_channels = in_channels
        self.k = kernel_size
        self.s = stride
        self.p = padding
        self.split_channels = split_channels

        self.ansatz_type = ansatz_type
        self.num_qubits = num_qubits
        self.num_qreps = num_qreps

        self.n_inputs = self.k * self.k if split_channels \
            else in_channels * self.k * self.k
        assert num_qubits == self.n_inputs, (
            "num_qubits must equal "
            + ('kernel_size*kernel_size' if split_channels
                else 'in_channels*kernel_size*kernel_size')
        )

        self.qdevice = qml.device(device_name, wires=self.num_qubits)
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
        BATCH_DIM = 0
        CHANNEL_DIM = 1
        HEIGHT_DIM = 2
        WIDTH_DIM = 3

        b = x.size(BATCH_DIM)
        c = x.size(CHANNEL_DIM)
        assert c == self.in_channels, "input channels mismatch"

        if self.p > 0:
            x = F.pad(x, (self.p,)*4)  # (left, right, top, bottom)

        patches = (
            x
            .unfold(HEIGHT_DIM, self.k, self.s)
            .unfold(WIDTH_DIM,  self.k, self.s)
        )

        ho = patches.size(HEIGHT_DIM)
        wo = patches.size(WIDTH_DIM)

        flat = patches.contiguous().view(-1, self.n_inputs)
        qout = self.qlayer(flat)

        OUT_B, OUT_H, OUT_W, OUT_Q, OUT_K = 0, 1, 2, 3, 4

        if self.split_channels:
            qout = qout.view(b, ho, wo, c, self.k * self.k)
            qout = qout.permute(OUT_B, OUT_Q, OUT_K, OUT_H, OUT_W)
            return qout.reshape(b, c * self.k * self.k, ho, wo)
        else:
            qout = qout.view(b, ho, wo, self.n_inputs)
            return qout.permute(OUT_B, OUT_Q, OUT_H, OUT_W)


class TanhScale(nn.Module):
    """tanh(x) * scale"""
    def __init__(self, scale: float = 1) -> None:
        super().__init__()
        self.scale = scale
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tanh(x) * self.scale


class HQNN_Quanv(nn.Module):
    """HQNN_Quanv model combining quanvolutional layers and classical."""
    def __init__(self, in_channels: int, num_classes: int,
                 input_size: Tuple[int, int, int],
                 quanv_kernels: List[int], quanv_strides: List[int],
                 quanv_paddings: List[int], pool_sizes: List[int],
                 ansatz_type: AnsatzType,
                 num_qreps: int, split_channels: bool,
                 qdevice: str, qdiff_method: str) -> None:
        super().__init__()

        n_quanv = len(quanv_kernels)
        assert len(quanv_strides) == n_quanv, \
            "quanv_strides must have the same length as quanv_kernels"
        assert len(quanv_paddings) == n_quanv, \
            "quanv_paddings must have the same length as quanv_kernels"
        assert len(pool_sizes) == n_quanv, \
            "pool_sizes must have the same length as quanv_kernels"

        self.tanh_scale = TanhScale(scale=math.pi/2)
        self.quanv_blocks = nn.ModuleList()
        _, h, w = input_size
        prev_ch = in_channels
        for i in range(n_quanv):
            k = quanv_kernels[i]
            s = quanv_strides[i]
            p = quanv_paddings[i]
            pool_k = pool_sizes[i]
            out_ch = prev_ch * k * k
            num_q = k * k if split_channels else out_ch

            block = nn.Sequential(
                Quanv(in_channels=prev_ch, num_qubits=num_q, kernel_size=k,
                      stride=s, padding=p, ansatz_type=ansatz_type,
                      num_qreps=num_qreps, split_channels=split_channels,
                      device_name=qdevice, diff_method=qdiff_method),
                nn.BatchNorm2d(num_features=out_ch),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=pool_k),
            )
            self.quanv_blocks.append(block)
            prev_ch = out_ch

            # After quanv
            h = (h + 2 * p - k) // s + 1
            w = (w + 2 * p - k) // s + 1

            # After pool
            h = (h - pool_k) // pool_k + 1
            w = (w - pool_k) // pool_k + 1

        self.flatten = nn.Flatten(start_dim=1)
        flat_dim = prev_ch * h * w

        self.fc = nn.Linear(in_features=flat_dim,
                            out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tanh_scale(x)
        for quanv_block in self.quanv_blocks:
            x = quanv_block(x)

        x = self.flatten(x)
        x = self.fc(x)

        return x
