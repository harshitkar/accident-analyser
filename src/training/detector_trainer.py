import os
import shutil
import time
from pathlib import Path

from ultralytics import YOLO

DATA_DIR = 'assets/dataset'
OUT_YAML = 'assets/dataset_yolo.yaml'
TARGET_WEIGHTS = 'models/yolo/yolov8n.pt'


def write_dataset_yaml(data_dir: str, out_path: str, names: dict):
    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'valid')
    test_path = os.path.join(data_dir, 'test')

    lines = [f"train: {train_path}", f"val: {val_path}"]
    if os.path.exists(test_path):
        lines.append(f"test: {test_path}")
    lines.append('names:')
    for k in sorted(names.keys()):
        lines.append(f"  {k}: '{names[k]}'")

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))


def find_weights(project_dir: str = 'runs') -> str:
    runs = list(Path(project_dir).glob('train*')) if os.path.exists(project_dir) else []
    if not runs:
        runs = list(Path(project_dir).glob('*'))
    if not runs:
        return ''
    latest = max(runs, key=lambda p: p.stat().st_mtime)
    wdir = latest / 'weights'
    if not wdir.exists():
        return ''
    b = wdir / 'best.pt'
    l = wdir / 'last.pt'
    if b.exists():
        return str(b)
    if l.exists():
        return str(l)
    pts = list(wdir.glob('*.pt'))
    return str(pts[0]) if pts else ''


def copy_to_target(weights_path: str, target: str):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copy2(weights_path, target)


def main():
    names = {0: 'accident', 1: 'normal'}
    write_dataset_yaml(DATA_DIR, OUT_YAML, names)

    # start training from official yolov8n seed weights
    model = YOLO('yolov8n.pt')
    model.train(data=OUT_YAML, epochs=50, imgsz=640, batch=16, project='runs', name=f'yolov8n_train_{int(time.time())}')

    weights = find_weights('runs')
    if weights:
        copy_to_target(weights, TARGET_WEIGHTS)
        print(f'Copied weights to {TARGET_WEIGHTS}')
    else:
        print('No weights found; check training output under runs/')


if __name__ == '__main__':
    main()
