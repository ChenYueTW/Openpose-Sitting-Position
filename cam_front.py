import cv2
import numpy as np
import helper

confidence = 0.1

class FrontSystem:
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.set(3, 640)
        self.cap.set(4, 480)

        if not self.cap.isOpened():
            print(f"Camera {camera_id} not found!")

        self.total_frames = 0
        self.valid_count = 0

    def reset(self):
        self.total_frames = 0
        self.valid_count = 0
        print("Front Counters Reset!")

    def process(self, image, keypoints_list):
        if not self.cap.isOpened():
            return np.zeros((480, 640, 3), dtype=np.uint8)

        ret, frame = self.cap.read()
        if not ret:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        self.total_frames += 1

        keypoints_list, rendered_img = helper.get_keypoints(op_wrapper, op_lib, frame)
        image = rendered_img if rendered_img is not None else frame

        is_detected = False

        if keypoints_list is not None and len(keypoints_list) > 0:
            keypoints = keypoints_list[0]
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[2] 
            
            if left_shoulder[2] > confidence and right_shoulder[2] > confidence:
                x1, y1 = int(left_shoulder[0]), int(left_shoulder[1])
                x2, y2 = int(right_shoulder[0]), int(right_shoulder[1])

                if not ((x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0)):
                    is_detected = True

                    delta_y = abs(y1 - y2)
                    delta_x = abs(x1 - x2)

                    if delta_x == 0: slope_percent = 0
                    else: slope_percent = (delta_y / delta_x) * 100

                    if slope_percent < 2.0: 
                        color = (0, 255, 0); status = "Good"
                    elif slope_percent < 5.0: 
                        color = (0, 255, 255); status = "Tilt"
                    else: 
                        color = (0, 0, 255); status = "Bad"

                    cv2.line(image, (x1, y1), (x2, y2), color, 2)
                    
                    text = f"Slope: {slope_percent:.1f}% ({status})"
                    mid_x, mid_y = int((x1 + x2) / 2), int((y1 + y2) / 2) - 20
                    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(image, (mid_x - 10, mid_y - h - 10), (mid_x + w + 10, mid_y + 10), (0,0,0), -1)
                    cv2.putText(image, text, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if is_detected:
            self.valid_count += 1
        
        accuracy = (self.valid_count / self.total_frames * 100) if self.total_frames > 0 else 0

        cv2.putText(image, "Cam 1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Acc: {accuracy:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return image