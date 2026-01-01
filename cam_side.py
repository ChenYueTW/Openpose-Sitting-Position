import cv2
import numpy as np
import math
import helper

confidence = 0.2

def calculate_neck_angle(ear_pt, shoulder_pt):
    ex, ey = ear_pt
    sx, sy = shoulder_pt

    dx = abs(ex - sx)
    dy = abs(sx - sy)
    
    if dx == 0: dx = 0.0001
        radian = math.atan2(dy, dx) 
        return math.degrees(radian)

class SideSystem:
    def __init__(self, camera_id=1):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.set(3, 640)
        self.cap.set(4, 480)

        if not self.cap.isOpened():
            print(f"Camera {camera_id} not found!")

        self.total_frames = 0
        self.valid_count = 0
        
        self.K = np.array([[640, 0, 480], [0, 640, 240], [0, 0, 1]], dtype=np.float32)
        self.D = np.array([-0.06, -0.2, 0.0, 0.0], dtype=np.float32)

        self.TARGET_ANGLE = 55

    def reset(self):
        self.total_frames = 0
        self.valid_count = 0
        print("Side Counters Reset!")

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

    def process(self, image, keypoints_list):
        if not self.cap.isOpened():
            return np.zeros((480, 640, 3), dtype=np.uint8)

        ret, raw_frame = self.cap.read()
        if not ret:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        frame = cv2.undistort(raw_frame, self.K, self.D)
        self.total_frames += 1
        
        keypoints_list, rendered_img = helper.get_keypoints(op_wrapper, op_lib, frame)
        image = rendered_img if rendered_img is not None else frame
        
        is_detected = False

        if keypoints_list is not None and len(keypoints_list) > 0:
            keypoints = keypoints_list[0]
            
            target_ear = keypoints[17]
            target_shoulder = keypoints[2]
            
            score = target_ear[2] + target_shoulder[2]

            if score > confidence:
                ex, ey = int(target_ear[0]), int(target_ear[1])
                sx, sy = int(target_shoulder[0]), int(target_shoulder[1])

                if not ((ex == 0 and ey == 0) or (sx == 0 and sy == 0)):
                    is_detected = True

                    direction = -1 if ex < sx else 1

                    current_angle = calculate_neck_angle((ex, ey), (sx, sy))

                    if current_angle >= self.TARGET_ANGLE:
                        status, color = "Good", (0, 255, 0)
                    elif current_angle >= (self.TARGET_ANGLE - 10): # 45~55
                        status, color = "Warning", (0, 255, 255)
                    else:
                        status, color = "Bad Pos", (0, 0, 255)

                    ref_len = 150
                    ref_rad = math.radians(self.TARGET_ANGLE)
                    ref_dx = int(ref_len * math.cos(ref_rad)) * direction
                    ref_dy = int(ref_len * math.sin(ref_rad)) 
                    rx, ry = sx + ref_dx, sy - ref_dy
                    
                    cv2.line(image, (sx, sy), (rx, ry), (150, 150, 150), 2)
                    cv2.putText(image, f"{self.TARGET_ANGLE} deg", (rx, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

                    cv2.line(image, (sx, sy), (ex, ey), color, 4)
                    cv2.circle(image, (ex, ey), 8, (255, 0, 0), -1)
                    cv2.circle(image, (sx, sy), 8, (0, 0, 255), -1)

                    cv2.putText(image, f"Angle: {int(current_angle)}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(image, status, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if is_detected:
            self.valid_count += 1
            
        accuracy = (self.valid_count / self.total_frames * 100) if self.total_frames > 0 else 0

        cv2.putText(image, f"Cam 2: Side (> {self.TARGET_ANGLE})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Acc: {accuracy:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return image