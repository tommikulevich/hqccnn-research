"""Utilities for running model inference and save per-layer activations."""
import os
import glob
import csv
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms

from config.schema import Config
from config.registry import MODEL_REGISTRY


def plot_activation(npy_path: str, max_channels: int = 32,
                    figsize: tuple = (10, 10),
                    cmap: str = 'viridis', add_colorbar: bool = True,
                    save_path: str = None, show: bool = False) -> None:
    """Load an activation map from a .npy file and plot up to max_channels."""
    arr = np.load(npy_path)

    if arr.ndim == 4 and arr.shape[0] == 1:
        # 4D → strip batch
        arr = arr[0]

    first_im = None

    if arr.ndim == 3:
        # (C, H, W) → grid of feature maps
        C, _, _ = arr.shape
        n = min(C, max_channels)
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes_list = list(np.array(axes).ravel())

        for i in range(n):
            ax = axes_list[i]
            im = ax.imshow(arr[i], cmap=cmap, interpolation='nearest',
                           aspect='auto')
            ax.axis('off')
            ax.set_title(f'ch {i}', fontsize=8)

            if first_im is None:
                first_im = im
        for ax in axes_list[n:]:
            ax.axis('off')
    elif arr.ndim == 2:
        # (H, W) → single image
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(arr, cmap=cmap, interpolation='nearest',
                       aspect='auto')
        ax.axis('off')
        ax.set_title(Path(npy_path).stem)

        first_im = im
    elif arr.ndim == 1:
        # (D,) → pixel line plot
        stripe = np.tile(arr[:, None], (1, 10))
        fig, ax = plt.subplots(figsize=(1, figsize[1]))
        im = ax.imshow(stripe, cmap=cmap, interpolation='nearest',
                       aspect='auto')
        ax.axis('off')
        ax.set_title(Path(npy_path).stem)

        first_im = im
    else:
        raise ValueError(f"Unsupported array ndim={arr.ndim}")

    plt.tight_layout()

    if add_colorbar and first_im is not None:
        cbar = fig.colorbar(first_im,
                            ax=fig.axes,
                            orientation='vertical',
                            fraction=0.02, pad=0.02)
        cbar.ax.tick_params(labelsize=8)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"[plot_activation] Saved plot to {save_path}")

    if show:
        plt.show()

    plt.close(fig)


def run_inference(config: Config, checkpoint_path: str, input_path: str,
                  output_dir: str) -> None:
    """Perform inference on images or directory using a trained model."""
    device = config.training.device

    # Instantiate model
    ModelClass = MODEL_REGISTRY.get(config.model.name)
    if ModelClass is None:
        raise ValueError(f"Unknown model: {config.model.name}")

    # Determine input channels from config
    input_shape = config.data.params.get('input_shape')
    if not input_shape or not isinstance(input_shape, (list, tuple)):
        raise ValueError("'input_shape' must be specified as a list in \
            data.params for inference mode")
    in_channels = input_shape[0]
    model = ModelClass(
        in_channels=in_channels,
        num_classes=config.data.params['num_classes'],
        **config.model.params,
    )

    # Load checkpoint
    chk = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = chk.get('model_state', chk)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # Build test transform
    transforms_cfg = config.data.params.get('transforms', {})
    tf_list = transforms_cfg.get('test')

    def _build_transform(transform_list):
        if not transform_list:
            return transforms.ToTensor()

        tfs = []
        for entry in transform_list:
            name = entry.get('name')
            params = entry.get('params', {})
            if not hasattr(transforms, name):
                raise ValueError(f"Unknown transform: {name}")
            cls = getattr(transforms, name)
            tfs.append(cls(**params))

        return transforms.Compose(tfs)

    test_transform = _build_transform(tf_list)

    # Collect image paths
    paths = []
    if os.path.isdir(input_path):
        patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
        for pat in patterns:
            paths.extend(glob.glob(os.path.join(input_path, pat)))
    elif os.path.isfile(input_path):
        paths = [input_path]
    else:
        raise ValueError(f"Invalid input path: {input_path}")

    if not paths:
        raise ValueError(f"No images found in {input_path}")

    # Prepare output directories
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(output_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    activ_dir = out_dir / 'activations_npy'
    activ_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / 'activations_plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    meta_file = out_dir / 'metadata.txt'
    with open(meta_file, 'w') as mf:
        mf.write(f"Timestamp: {ts}\n")
        mf.write(f"Config:\n{config}\n")
        mf.write(f"Model:\n{model}\n")
        mf.write(f"Checkpoint: {checkpoint_path}\n")
        mf.write(f"Input_path: {input_path}\n")
        mf.write(f"Device: {device}\n")
        mf.write("Images:\n")
        for p in paths:
            mf.write(f"- {p}\n")

    # Register hooks for activation maps
    activations = {}

    def _make_pre_hook(name):
        def hook(module, inputs):
            activations[f"{name}__in"] = inputs[0].detach().cpu()
        return hook

    def _make_post_hook(name):
        def hook(module, inputs, output):
            activations[f"{name}__out"] = output.detach().cpu()
        return hook

    hooks = []
    for name, module in model.named_modules():
        if not name:
            continue
        hooks.append(module.register_forward_pre_hook(_make_pre_hook(name)))
        hooks.append(module.register_forward_hook(_make_post_hook(name)))

    ordered_keys = []
    for name, module in model.named_modules():
        if not name:
            continue
        ordered_keys.append(f"{name}__in")
        ordered_keys.append(f"{name}__out")

    layer_to_idx = {k: i+1 for i, k in enumerate(ordered_keys)}

    # Open CSV for predictions
    csv_path = out_dir / 'predictions.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['filename', 'prediction',
                                                     'probabilities'])
        writer.writeheader()
        for img_path in paths:
            activations.clear()

            img = Image.open(img_path)
            tensor = test_transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(tensor)
                probs = F.softmax(output, dim=1)[0].cpu().numpy().tolist()
                pred = int(np.argmax(probs))

            # Save activation maps and plot
            for key in ordered_keys:
                if key not in activations:
                    continue
                arr = activations[key].squeeze(0).numpy()
                idx = layer_to_idx[key]

                stem = Path(img_path).stem
                act_file = f"{stem}-idx{idx:02d}-{key}.npy"
                np.save(activ_dir / act_file, arr)

                plot_file = f"{stem}-idx{idx:02d}-{key}.png"
                plot_activation(activ_dir / act_file,
                                save_path=plots_dir / plot_file)

            # Write prediction
            writer.writerow({
                'filename': Path(img_path).name,
                'prediction': pred,
                'probabilities': ';'.join(map(str, probs))
            })

    print(f"Inference completed. Results saved to {out_dir}")

    # Remove hooks
    for h in hooks:
        h.remove()
