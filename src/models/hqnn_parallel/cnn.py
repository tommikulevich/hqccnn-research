"""Classic CNN model."""
from typing import List, Tuple

import torch
import torch.nn as nn


class HQNN_Parallel_Classic_CNN(nn.Module):
    """Classical CNN model without quantum layers."""
    def __init__(self, in_channels: int, num_classes: int,
                 input_size: Tuple[int, int, int],
                 conv_channels: List[int], conv_kernels: List[int],
                 conv_strides: List[int], conv_paddings: List[int],
                 pool_sizes: List[int], num_pseudo_qubits: int,
                 num_pseudo_qlayers: int) -> None:
        super().__init__()

        self.num_q_features = num_pseudo_qlayers * num_pseudo_qubits

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

        self.fc_block1_5 = nn.Linear(in_features=self.num_q_features,
                                     out_features=self.num_q_features)

        # Classical linear layer
        self.fc_block2 = nn.Linear(in_features=self.num_q_features,
                                   out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        x = self.flatten(x)
        x = self.fc_block1(x)
        x = self.fc_block1_5(x)
        x = self.fc_block2(x)

        return x
