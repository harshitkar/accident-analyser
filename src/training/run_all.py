"""Run both trainers inside the training package.

Usage:
    python -m src.training.run_all
"""

import sys
import subprocess
import os


def run_module(module_name: str):
    cmd = [sys.executable, '-m', module_name]
    print(f"Running: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"{module_name} failed with code {res.returncode}")


def main():
    cwd = os.getcwd()
    print(f"Starting training from {cwd}")

    # Run classifier trainer within package
    run_module('src.training.classifier_trainer')

    # Run detector trainer within package
    run_module('src.training.detector_trainer')

    print('All training steps finished')


if __name__ == '__main__':
    main()
