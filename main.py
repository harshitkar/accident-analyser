from src.Configs import *
from src.Detection import *
from src.Processors import *


def main():
    classifier_config = ClassifierConfig(
        model_type="detection_model",
        classifier_path="models/cnn/detection_model.h5"
    )

    detection_config = DetectionConfig(
        model_path='models/yolo/yolov8n.pt',
        threshold_conf=0.4,
        valid_classes=[0, 1],
        nms_threshold=0.4
    )

    display_config = DisplayConfig(window_name="Vehicle Detection")

    detection_model = VehicleDetection(detection_config, classifier_config)

    video_processor = VideoProcessor(
        video_path="assets/input/Testing Video.ts",
        detection_model=detection_model,
        display_config=display_config,
        frame_skip=1
    )
    video_processor.process_video()
    

if __name__ == "__main__":
    main()