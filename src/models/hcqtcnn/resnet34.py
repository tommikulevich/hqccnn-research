"""Classical ResNet34."""
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models


class HCQTCNN_Classic_ResNet(nn.Module):
    """Classical pre-trained ResNet34."""
    def __init__(self, in_channels: int, num_classes: int,
                 input_size: Tuple[int, int, int]) -> None:
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

        # Classical classifier
        self.fc_classifier = nn.Linear(in_features=resnet.fc.in_features,
                                       out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet_features(x)
        x = self.flatten(x)
        x = self.fc_classifier(x)

        return x
