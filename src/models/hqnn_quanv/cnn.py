"""Classic CNN model."""
from typing import List, Tuple

import torch
import torch.nn as nn


class HQNN_Quanv_Classic_CNN(nn.Module):
    """Classical CNN model without quantum layers."""
    def __init__(self, in_channels: int, num_classes: int,
                 input_size: Tuple[int, int, int],
                 conv_kernels: List[int], conv_strides: List[int],
                 conv_paddings: List[int], pool_sizes: List[int]) -> None:
        super().__init__()

        n_quanv = len(conv_kernels)
        assert len(conv_strides) == n_quanv, \
            "conv_strides must have the same length as conv_kernels"
        assert len(conv_paddings) == n_quanv, \
            "conv_paddings must have the same length as conv_kernels"
        assert len(pool_sizes) == n_quanv, \
            "pool_sizes must have the same length as conv_kernels"

        self.conv_blocks = nn.ModuleList()
        _, h, w = input_size
        prev_ch = in_channels
        for i in range(n_quanv):
            k = conv_kernels[i]
            s = conv_strides[i]
            p = conv_paddings[i]
            pool_k = pool_sizes[i]
            out_ch = prev_ch * k * k

            block = nn.Sequential(
                nn.Conv2d(in_channels=prev_ch, out_channels=out_ch,
                          kernel_size=k, stride=s, padding=p),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=pool_k),
            )
            self.conv_blocks.append(block)
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
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        x = self.flatten(x)
        x = self.fc(x)

        return x
