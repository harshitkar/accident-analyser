import os
import time
from .Configs import *
from .Detection import *

class VideoProcessor:
    def __init__(self, video_path: str, detection_model: VehicleDetection, display_config: DisplayConfig, frame_skip=1):
        self.video_path = video_path
        self.detection_model = detection_model
        self.display_config = display_config
        self.frame_skip = frame_skip
        self.frame_count = 0
        self.label_condition = DetectCondition()
        self.preprocess_times = []
        self.postprocess_times = []



    def draw_bbox(self, frame, bbox, cls, track_id, conf):
        x1, y1, width, height = bbox
        x2, y2 = x1 + width, y1 + height
        frame_height, frame_width = frame.shape[:2]
        
        # Ensure bounding box is within image dimensions
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame_width, x2), min(frame_height, y2)
        
        condition = self.detection_model.CarConditionStatus(int(cls))

        # Default color based on condition
        condition_colors = {
            self.label_condition.accident: (0, 0, 255),  # Red
            self.label_condition.normal: (0, 255, 0),   # Green
        }
        color = condition_colors.get(condition, (255, 255, 255))  # Default to white

        # Damage classification
        damage_class, damage_conf = "No Damage", 0.0
        final_time = 0
        if condition == self.label_condition.accident:
            cropped_image = frame[y1:y2, x1:x2]
            if cropped_image.size > 0:
                start_time = time.time()
                damage_class, damage_conf = self.detection_model.classify_damage(cropped_image)
                end_time = time.time()

                final_time = (end_time - start_time) * 1000
                print(f"Execution time classify demage car {final_time:.2f}ms")

            damage_colors = {
                1: (0, 255, 255),  # normal -> yellow
                0: (0, 165, 255),  # moderate -> orange
                2: (0, 0, 255),    # severe -> red
            }
            
            if damage_class in damage_colors:
                color = damage_colors[damage_class]
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Damage labels and text
        damage_labels = {0: 'moderate', 1: 'normal', 2: 'severe'}
        damage_text = damage_labels.get(damage_class, "No Damage")

        # Text annotations
        text_lines = [
            f'Status: {condition} ({conf:.2f})',
            f'Classify Damage: {damage_text} ({damage_conf:.2f})',
            f'ID: {track_id}'
        ]
        text_y_offset = 10
        for i, text in enumerate(text_lines):
            cv2.putText(frame, text, (x1, y1 - text_y_offset - (i * 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            return

        cv2.namedWindow(self.display_config.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.display_config.window_name, self.display_config.WINDOW_X, self.display_config.WINDOW_Y)

        start_time = time.time()
        paused = False
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                if self.frame_count % self.frame_skip != 0:
                    self.frame_count += 1
                    continue

                preprocess_start = time.time()
                preprocess_end = time.time()

                # Detection
                boxes, confidences, classes = self.detection_model.process_frame(frame)
                indices = self.detection_model.apply_nms(boxes, confidences)
                detections = [(boxes[i], confidences[i], classes[i]) for i in indices]

                accident_detected = False

                # Classify
                for i, (bbox, conf, cls) in enumerate(detections):
                    condition = self.detection_model.CarConditionStatus(int(cls))
                    if condition == self.label_condition.accident:
                        accident_detected = True
                    self.draw_bbox(frame, bbox, cls, i, conf)

                self.preprocess_times.append(preprocess_end - preprocess_start)

                # Display FPS
                time_diff = time.time() - start_time
                fps = 1.0 / time_diff if time_diff > 0.001 else 0.0
                start_time = time.time()

                cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                if accident_detected:
                    overlay = frame.copy()
                    alpha = self.display_config.overlay_alpha
                    cv2.rectangle(overlay, (10, 50), (300, 90), (0, 0, 255), -1)
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                    cv2.putText(frame, self.display_config.overlay_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow(self.display_config.window_name, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            if key == ord('s'):
                paused = not paused 
                if paused:
                    print("Video paused. Press 's' to resume.")
                else:
                    print("Video resumed.")

            self.frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
