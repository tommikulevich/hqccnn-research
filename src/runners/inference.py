"""Script to train HQNN_Parallel models."""
import os
import glob
import csv
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from config.config import Config
from config.registry import MODEL_REGISTRY


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
    chk = torch.load(checkpoint_path, map_location=device)
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
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    activ_dir = out_dir / 'activations'
    activ_dir.mkdir(parents=True, exist_ok=True)

    # Register hooks for activation maps
    activations = {}
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(
                module.register_forward_hook(
                    lambda mod, inp, out, name=name:
                        activations.setdefault(name, out.detach().cpu())
                )
            )

    # Open CSV for predictions
    csv_path = out_dir / 'predictions.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['filename', 'prediction',
                                                     'probabilities'])
        writer.writeheader()
        for img_path in paths:
            activations.clear()

            img = Image.open(img_path).convert('RGB')
            tensor = test_transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(tensor)
                probs = F.softmax(output, dim=1)[0].cpu().numpy().tolist()
                pred = int(np.argmax(probs))

            # Save activation maps
            for layer, act in activations.items():
                arr = act.squeeze(0).numpy()
                np.save(activ_dir / f"{Path(img_path).stem}_{layer}.npy", arr)

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
