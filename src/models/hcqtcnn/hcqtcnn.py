"""Hybrid Classical-Quantum Transferring CNN with ResNet34 backbone."""
import math
from typing import Tuple, List

import torch
import torch.nn as nn
import torchvision.models as models
import pennylane as qml
from pennylane.qnn import TorchLayer


class QLayer(nn.Module):
    def __init__(self, num_qubits: int, num_qreps: int,
                 qdevice: str = "default.qubit",
                 qdiff_method: str = "best"):
        super().__init__()

        self.num_qubits = num_qubits
        self.num_qreps = num_qreps

        self.qdevice = qml.device(qdevice, wires=num_qubits)
        self.qnode = qml.QNode(self._circuit,
                               device=self.qdevice,
                               interface="torch",
                               diff_method=qdiff_method)

        weight_shapes = {
            "weights": qml.StronglyEntanglingLayers.shape(
                n_layers=num_qreps, n_wires=num_qubits
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
        return self.qlayer(x)


class SigmoidScale(nn.Module):
    """sigmoid(x) * scale"""
    def __init__(self, scale: float = 1) -> None:
        super().__init__()
        self.scale = scale
        self.sigm = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigm(x) * self.scale


class HCQTCNN(nn.Module):
    """HCQTCNN_ResNet combining pre-trained ResNet34 and quantum layer."""
    def __init__(self, in_channels: int, num_classes: int,
                 input_size: Tuple[int, int, int],
                 num_qubits: int, num_qreps: int,
                 qdevice: str, qdiff_method) -> None:
        super().__init__()

        assert in_channels == input_size[0], \
            "in_channels must have the same length as input_size[0]"

        # Pretrained ResNet34 feature extractor (remove last layer)
        resnet = models.resnet34(weights='IMAGENET1K_V1')
        self.resnet_features = nn.Sequential(
            *list(resnet.children())[:-1],
        )

        # Flatten output of ResNet34
        self.flatten = nn.Flatten(start_dim=1)

        # Linear to reduce ResNet feature dim to num_qubits inputs
        resnet_out_dim = resnet.fc.in_features
        self.fc_reduce = nn.Sequential(
            nn.Linear(in_features=resnet_out_dim,
                      out_features=num_qubits),
            nn.BatchNorm1d(num_features=num_qubits),
            nn.ReLU()
        )

        # Quantum variational layer
        self.sigm_scale = SigmoidScale(scale=math.pi)
        self.qlayer = QLayer(num_qubits=num_qubits, num_qreps=num_qreps,
                             qdevice=qdevice, qdiff_method=qdiff_method)

        # Classical classifier: from QLayer outputs to num_classes
        self.fc_classifier = nn.Linear(in_features=num_qubits,
                                       out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet_features(x)
        x = self.flatten(x)
        x = self.fc_reduce(x)

        x = self.sigm_scale(x)
        x = self.qlayer(x)

        x = self.fc_classifier(x)

        return x
