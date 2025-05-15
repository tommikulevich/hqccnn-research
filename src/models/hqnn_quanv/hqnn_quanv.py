"""Hybrid QNN Quanvolutional model."""
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane.qnn import TorchLayer


class Quanv(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int,
                 stride: int, padding: int,
                 num_qubits: int, num_qreps: int,
                 qdevice: str = "default.qubit",
                 qdiff_method: str = "best") -> None:
        super().__init__()

        self.in_channels = in_channels
        self.k = kernel_size
        self.s = stride
        self.p = padding

        self.num_qubits = num_qubits
        self.num_qreps = num_qreps

        self.n_inputs = in_channels * self.k * self.k
        assert self.num_qubits == self.n_inputs, \
            "num_qubits must equal in_channels*kernel_size*kernel_size"

        self.qdevice = qml.device(qdevice, wires=self.num_qubits)
        self.qnode = qml.QNode(self._circuit,
                               device=self.qdevice,
                               interface="torch",
                               qdiff_method=qdiff_method)

        weight_shapes = {
            "weights": qml.StronglyEntanglingLayers.shape(
                n_layers=self.num_qreps, n_wires=self.num_qubits
            )
        }

        self.qlayer = TorchLayer(self.qnode, weight_shapes)

    def _circuit(self, inputs: torch.Tensor, weights: torch.Tensor) \
            -> List[torch.Tensor]:
        qml.AngleEmbedding(inputs, rotation="Y",
                           wires=list(range(self.num_qubits)))
        qml.StronglyEntanglingLayers(weights,
                                     wires=list(range(self.num_qubits)))
        return [qml.expval(qml.PauliZ(wires=i))
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
        flat = flat.tanh()
        flat = flat * (math.pi / 2)  # [-pi/2; pi/2]

        qout = self.qlayer(flat)                       # [B*Ho*Wo, qubits]
        qm = qout.view(b, ho, wo, self.num_qubits)     # [B, Ho, Wo, qubits]

        OUT_B = 0
        OUT_Q = 3
        OUT_H = 1
        OUT_W = 2

        return qm.permute(OUT_B, OUT_Q, OUT_H, OUT_W)  # [B, qubits, Ho, Wo]


class HQNN_Quanv(nn.Module):
    """HQNN_Quanv model combining quanvolutional layers and classical."""
    def __init__(self, in_channels: int, num_classes: int,
                 input_size: Tuple[int, int, int],
                 quanv_kernels: List[int], quanv_strides: List[int],
                 quanv_paddings: List[int], pool_sizes: List[int],
                 num_qreps: int, qdevice: str, qdiff_method: str) -> None:
        super().__init__()

        n_quanv = len(quanv_kernels)
        assert len(quanv_strides) == n_quanv, \
            "quanv_strides must have the same length as quanv_kernels"
        assert len(quanv_paddings) == n_quanv, \
            "quanv_paddings must have the same length as quanv_kernels"
        assert len(pool_sizes) == n_quanv, \
            "pool_sizes must have the same length as quanv_kernels"

        self.quanv_blocks = nn.ModuleList()
        _, h, w = input_size
        prev_ch = in_channels
        for i in range(n_quanv):
            k = quanv_kernels[i]
            s = quanv_strides[i]
            p = quanv_paddings[i]
            pool_k = pool_sizes[i]
            out_ch = prev_ch * k * k

            block = nn.Sequential(
                Quanv(in_channels=prev_ch, num_qubits=out_ch, kernel_size=k,
                      stride=s, padding=p,
                      num_qreps=num_qreps, qdevice=qdevice,
                      qdiff_method=qdiff_method),
                nn.BatchNorm2d(out_ch),
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
        for quanv_block in self.quanv_blocks:
            x = quanv_block(x)

        x = self.flatten(x)
        x = self.fc(x)

        return x
