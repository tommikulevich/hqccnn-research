"""Script to train HQNN_Parallel models."""
import argparse

from runners.inference import run_inference
from runners.train import run_train
from config.schema import Config
from config.loader import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Environment for hybrid model training')
    parser.add_argument('-c', '--config', type=str,
                        default="configs/default.yaml",
                        help='Path to YAML config')
    parser.add_argument('-r', '--resume', type=str, default=None,
                        help='[Train] Path to checkpoint to resume')
    parser.add_argument('--dry-run', action='store_true',
                        help='[Train] Perform a dry run')
    parser.add_argument('--infer', action='store_true',
                        help='Run inference only (skips training)')
    parser.add_argument('--checkpoint', type=str,
                        help='[Inference] Path to model checkpoint')
    parser.add_argument('--input', type=str,
                        help='[Inference] Path to image file or directory.')
    parser.add_argument('--output', type=str, default='outputs/inference',
                        help='[Inference] Directory to save inference')
    args = parser.parse_args()

    # Load config
    config: Config = load_config(args.config)

    # Inference mode: skip training and perform inference
    if args.infer:
        if not args.checkpoint or not args.input:
            parser.error("--infer requires --checkpoint and --input")
        run_inference(config, args.checkpoint, args.input, args.output)
        return

    # Train mode
    run_train(config, args.resume, args.dry_run)


if __name__ == '__main__':
    main()
