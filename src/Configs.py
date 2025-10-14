from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class DetectionConfig:
    model_path: str
    threshold_conf: float
    valid_classes: List[int]
    nms_threshold: float = 0.4

@dataclass
class DisplayConfig:
    window_name: str = "Window"
    overlay_alpha: float = 0.3
    overlay_text: str = "Accident Detected!"
    WINDOW_X: int = 1280
    WINDOW_Y: int = 720

@dataclass
class DetectCondition:
    accident: str = "accident"
    normal: str = "no accident"

@dataclass
class ClassifierConfig:
    model_type: str = 'EfficientNetB0'  
    classifier_path: str = 'models/cnn/EfficientNetB0.h5'