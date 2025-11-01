# accident-analyser

## Project Overview
This project detects vehicle accidents and classifies damage severity using deep learning models (YOLO for detection, detection_model for classification). It processes video or image data and outputs accident status and damage class.

## Setup Instructions

### 1. Clone the repository
```sh
git clone https://github.com/harshitkar/accident-analyser
cd accident-analyser
```

### 2. Create and activate a Conda environment
```sh
conda create --name vehicle_damage_env python=3.9.19
conda activate vehicle_damage_env
```

### 3. Install requirements
```sh
pip install --upgrade pip
pip install -r requirements.txt
```

## Requirements
- Python 3.9.19 (recommended via Conda)
- See `requirements.txt` for all Python dependencies.

## Running the Project
```sh
python main.py
```
