import cv2
import math
import json
import numpy as np
import mediapipe as mp
from random import random
from ultralytics import YOLO
from datetime import datetime

model = YOLO("yolov8n.pt")
ankle_xy_l = None
ankle_xy_r = None
yolo_model = None
face_mesh = None
mp_face_mesh = None
UP_THRESHOLD = -10 
DOWN_THRESHOLD = 10
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
prev_distance = None
prev_frame = None
skip_frames = 0  
prev_pitch = None
pitch_alpha = 0.25  
PITCH_JUMP_THRESH = 30  
initial_detections = []
determined_foot = None
detection_count = 0
prev_side_frame = None
chosen_side_frame = None
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) 

pose_side = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) 

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),            
    (0.0, -330.0, -65.0),      
    (-225.0, 170.0, -135.0),    
    (225.0, 170.0, -135.0),    
    (-150.0, -150.0, -125.0),  
    (150.0, -150.0, -125.0)     
])

def init_head_pose_detector(yolo_model_path='yolov8n.pt', up_threshold=-10, down_threshold=10):  
    global yolo_model, face_mesh, mp_face_mesh, UP_THRESHOLD, DOWN_THRESHOLD
    
    try:
        yolo_model = YOLO(yolo_model_path)
        
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.15,
            min_tracking_confidence=0.15
        )
        
        UP_THRESHOLD = up_threshold
        DOWN_THRESHOLD = down_threshold
        
        print(f"Head pose detector initialized successfully!")
        print(f"Thresholds: UP < {up_threshold}°, DOWN > {down_threshold}°")
        return True
    except Exception as e:
        print(f"Error initializing head pose detector: {e}")
        return False

def detect_head_pose(frame):
    global yolo_model, face_mesh, UP_THRESHOLD, DOWN_THRESHOLD, MODEL_POINTS
    global prev_pitch, pitch_alpha, PITCH_JUMP_THRESH

    if yolo_model is None or face_mesh is None:
        if not init_head_pose_detector():
            return frame.copy(), "DETECTOR_ERROR", None, None, None, None

    try:
        output_image = frame.copy()
        results = yolo_model(frame, verbose=False)
        bbox = None

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    if int(box.cls[0]) == 0:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        if confidence > 0.15:
                            bbox = (int(x1), int(y1), int(x2), int(y2))
                            break
                if bbox:
                    break

        if bbox is None:
            return output_image, "NO_PERSON", None, None, None, None

        x1, y1, x2, y2 = bbox
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output_image, 'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        height = y2 - y1
        upper_third_height = int(height * 0.3)
        upper_crop = frame[y1:y1 + upper_third_height, x1:x2]

        if upper_crop.size == 0:
            return output_image, "CROP_FAILED", None, None, None, bbox

        rgb_image = cv2.cvtColor(upper_crop, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return output_image, "NO_FACE", None, None, None, bbox

        face_landmarks = results.multi_face_landmarks[0]
        h, w = upper_crop.shape[:2]
        landmarks_2d = []
        key_landmarks = [1, 152, 33, 263, 61, 291]  # nose, chin, left eye, right eye, left mouth, right mouth

        for idx in key_landmarks:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks_2d.append([x, y])

        landmarks_2d = np.array(landmarks_2d, dtype=np.float64)

        focal_length = w
        cam_matrix = np.array([[focal_length, 0, w / 2],
                               [0, focal_length, h / 2],
                               [0, 0, 1]], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))

        success, rot_vec, trans_vec = cv2.solvePnP(
            MODEL_POINTS, landmarks_2d, cam_matrix, dist_coeffs
        )

        if not success:
            return output_image, "POSE_FAILED", None, None, None, bbox

        rot_matrix, _ = cv2.Rodrigues(rot_vec)
        sy = math.sqrt(rot_matrix[0, 0] ** 2 + rot_matrix[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(rot_matrix[2, 1], rot_matrix[2, 2])
            y = math.atan2(-rot_matrix[2, 0], sy)
            z = math.atan2(rot_matrix[1, 0], rot_matrix[0, 0])
        else:
            x = math.atan2(-rot_matrix[1, 2], rot_matrix[1, 1])
            y = math.atan2(-rot_matrix[2, 0], sy)
            z = 0

        pitch = math.degrees(x)  # Head down/up
        yaw = math.degrees(y)    # Head left/right
        roll = math.degrees(z)   # Head tilt

        smoothed_pitch = pitch
        if prev_pitch is not None:
            if abs(pitch - prev_pitch) > PITCH_JUMP_THRESH:
                smoothed_pitch = prev_pitch  
            else:
                smoothed_pitch = prev_pitch * (1 - pitch_alpha) + pitch * pitch_alpha
        prev_pitch = smoothed_pitch  

        if smoothed_pitch < UP_THRESHOLD:
            direction = 'DOWN'
        elif smoothed_pitch > DOWN_THRESHOLD:
            direction = 'UP'
        else:
            direction = 'FORWARD'

        offset_x, offset_y = x1, y1
        for point in landmarks_2d:
            x = int(point[0] + offset_x)
            y = int(point[1] + offset_y)
            cv2.circle(output_image, (x, y), 3, (255, 0, 0), -1)

        color = (0, 255, 0) if direction == 'FORWARD' else (0, 0, 255) if direction == 'DOWN' else (255, 0, 0)
        cv2.putText(output_image, f'Direction: {direction}', (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(output_image, f'Pitch: {smoothed_pitch:.1f}°', (x1, y2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(output_image, f'Yaw: {yaw:.1f}°', (x1, y2 + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        crop_y2 = y1 + upper_third_height
        cv2.rectangle(output_image, (x1, y1), (x2, crop_y2), (255, 255, 0), 2)
        cv2.putText(output_image, 'Head Region', (x1, crop_y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        return output_image, direction, smoothed_pitch, yaw, roll, bbox

    except Exception as e:
        print(f"Error in head pose detection: {e}")
        return frame.copy(), "ERROR", None, None, None, None

def detect_ball(frame):
    yolo_result = model.predict(frame, verbose=False)[0]
    
    center = (0, 0)
    ball_diameter = 0
    best_box = None
    max_diameter = 0  

    for box, cls_id, score in zip(yolo_result.boxes.xyxy, yolo_result.boxes.cls, yolo_result.boxes.conf):
        cls_id = int(cls_id)
        x1, y1, x2, y2 = map(int, box)
        if score < 0.1:
            continue

        if cls_id == 32: 
            width = x2 - x1
            height = y2 - y1
            diameter = (width + height) / 2

            if diameter > max_diameter:
                max_diameter = diameter
                ball_diameter = diameter
                best_box = [x1, y1, x2, y2]
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

    if best_box:
        x1, y1, x2, y2 = best_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame, center, ball_diameter, best_box

def detect_person(side_frame):
    global person_center
    yolo_result = model.predict(side_frame, verbose=False)[0]
    
    person_center = (0, 0)
    person_diameter = 0
    best_box = None
    max_diameter = 0  

    for box, cls_id, score in zip(yolo_result.boxes.xyxy, yolo_result.boxes.cls, yolo_result.boxes.conf):
        cls_id = int(cls_id)
        x1, y1, x2, y2 = map(int, box)
        if score < 0.4:
            continue

        if cls_id == 0: 
            width = x2 - x1
            height = y2 - y1
            diameter = (width + height) / 2

            if diameter > max_diameter:
                max_diameter = diameter
                person_diameter = diameter
                person_best_box = [x1, y1, x2, y2]
                person_center = ((x1 + x2) // 2, (y1 + y2) // 2)

    if person_best_box:  
        x1, y1, x2, y2 = person_best_box  
        cv2.rectangle(side_frame, (x1, y1), (x2, y2), (0, 0, 225), 2)

    return side_frame, person_center, person_diameter, person_best_box 

def detect_receiving_foot(best_box, ball_center):
    global initial_detections, determined_foot, detection_count
    
    if ball_center is None or best_box is None or ball_center == (0, 0):
        return determined_foot  
    
    if determined_foot is None:
        x1, y1, x2, y2 = best_box
        person_center_x = (x1 + x2) // 2
        
        if ball_center[0] > person_center_x:
            current_detection = "right"
        else:
            current_detection = "left"
        
        initial_detections.append(current_detection)
        detection_count += 1
        
        print(f"Detection {detection_count}: {current_detection}")
        
        if len(initial_detections) >= 10:
            right_count = initial_detections.count("right")
            left_count = initial_detections.count("left")
            
            if right_count > left_count:
                determined_foot = "right"
                print(f"FINAL DECISION: RIGHT foot ({right_count}/10 votes)")
            else:
                determined_foot = "left"
                print(f"FINAL DECISION: LEFT foot ({left_count}/10 votes)")
        
        return current_detection
    
    else:
        return determined_foot

def check_receiving_position(side_frame, right_point, left_point, ball_center, best_box): 
    if ball_center == (0, 0) or right_point is None or left_point is None or best_box is None:
        return False  
    
    foot = detect_receiving_foot(best_box, ball_center)

    
    if foot == "right" and right_point is not None:
        right_dist = math.sqrt((right_point[0] - ball_center[0]) ** 2 + (right_point[1] - ball_center[1]) ** 2)
        print(f"Distance from point to ball right: {right_dist}")
        
        cv2.line(side_frame, right_point, ball_center, (0, 255, 255), 2) 
        cv2.putText(side_frame, f"R-Dist: {int(right_dist)}", 
                    (right_point[0], right_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if right_dist < 45:
            print("detected (right foot)")
            return True
            
    elif foot == "left" and left_point is not None:         
        left_dist = math.sqrt((left_point[0] - ball_center[0]) ** 2 + (left_point[1] - ball_center[1]) ** 2)
        print(f"Distance from point to ball left: {left_dist}")
        
        cv2.line(side_frame, left_point, ball_center, (0, 255, 255), 2)  
        cv2.putText(side_frame, f"L-Dist: {int(left_dist)}", 
                    (left_point[0], left_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if left_dist < 45:
            print("detected (left foot)")
            return True
            
    return False

def draw_ankel_arrow(frame, ankle, finger, hip=None, knee=None, color=(255,0,0), thickness=2):#completed

    cv2.arrowedLine(frame, ankle, finger, color, thickness, tipLength=0.3)
    
    dist = int(((ankle[0] - finger[0])**2 + (ankle[1] - finger[1])**2)**0.5)
    
    angle_text = ""
    if hip is not None and knee is not None:
        angle = calculate_knee_angle(hip, knee, ankle)
        if angle is not None:
            angle_text = f"Angle: {angle} deg"
    
    mid_point = ((ankle[0] + finger[0]) // 2, (ankle[1] + finger[1]) // 2)
    
    cv2.putText(frame, f"Dist: {dist}", (mid_point[0], mid_point[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    if angle_text:
        cv2.putText(frame, angle_text, (mid_point[0], mid_point[1]+25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    return angle, dist

def process_pose_and_draw_arrows(frame, pose_instance):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_instance.process(image_rgb)
    if results.pose_landmarks:
        h, w = frame.shape[:2]
        landmarks = results.pose_landmarks.landmark

        left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                         int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
        right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))

        left_hip = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w),
                    int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h))
        right_hip = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w),
                     int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * h))

        left_knee = (int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * w),
                     int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * h))
        right_knee = (int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x * w),
                      int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y * h))

        left_ankle = (int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * w),
                      int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * h))
        right_ankle = (int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * w),
                       int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h))

        left_heel = (int(landmarks[mp_pose.PoseLandmark.LEFT_HEEL].x * w),
                     int(landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y * h))
        right_heel = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].x * w),
                      int(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y * h))

        left_foot_index = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
        right_foot_index = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
        finger_xy_l = (int(left_foot_index.x * w), int(left_foot_index.y * h))
        finger_xy_r = (int(right_foot_index.x * w), int(right_foot_index.y * h))

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

        return (right_ankle, left_ankle,
                right_heel, left_heel,
                finger_xy_l, finger_xy_r,
                left_hip, right_hip,
                left_shoulder, right_shoulder,
                left_knee, right_knee)
    else:
        return (None,) * 12 

def calculate_knee_angle(hip, knee, ankle):
    v1 = (hip[0] - knee[0], hip[1] - knee[1])   
    v2 = (ankle[0] - knee[0], ankle[1] - knee[1]) 

    v1_len = math.sqrt(v1[0]**2 + v1[1]**2)
    v2_len = math.sqrt(v2[0]**2 + v2[1]**2)

    if v1_len == 0 or v2_len == 0:
        return None 

    cos_theta = (v1[0]*v2[0] + v1[1]*v2[1]) / (v1_len * v2_len)
    cos_theta = max(min(cos_theta, 1), -1)  
    angle = math.degrees(math.acos(cos_theta))
    return int(angle)

def draw_knee_angle(frame, hip, knee, ankle, color=(255, 255, 0)):
    angle = calculate_knee_angle(hip, knee, ankle)
    if angle is not None:
        cv2.circle(frame, knee, 5, color, -1) 
        cv2.putText(frame, f"{angle}", (knee[0] + 10, knee[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return angle

def calculate_horizontal_pelvis_angle(left_hip, right_hip):
    dx = right_hip[0] - left_hip[0]
    dy = right_hip[1] - left_hip[1]

    angle = math.degrees(math.atan2(dy, dx))
    return int(abs(angle))

def calculate_vertical_torso_angle(left_hip, right_hip, left_shoulder, right_shoulder):
    mid_hip = ((left_hip[0] + right_hip[0]) // 2,
               (left_hip[1] + right_hip[1]) // 2)

    mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) // 2,
                    (left_shoulder[1] + right_shoulder[1]) // 2)

    dx = mid_shoulder[0] - mid_hip[0]
    dy = mid_shoulder[1] - mid_hip[1]

    angle = math.degrees(math.atan2(dy, dx)) 
    return int(abs(angle))

def calculate_side_torso_angle(hip, shoulder):
    if hip is None or shoulder is None:
        return None

    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]

    angle = math.degrees(math.atan2(dx, dy))  

    return int(abs(angle))

def calculate_ankle_ball_dist_and_draw(frame, right_ankle, left_ankle, ball_center, ball_diameter, foot):
    if right_ankle is None or left_ankle is None or ball_center is None or ball_diameter is None:
        return None
    
    if ball_diameter <= 0: 
        cv2.putText(frame, "Ball not detected", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return None

    if foot == "right" and left_ankle is not None:
        ankle = left_ankle
    elif foot == "left" and right_ankle is not None:
        ankle = right_ankle
    else:
        return None   

    ankle_ball_dist_px = math.sqrt(
        (ankle[0] - ball_center[0])**2 +
        (ankle[1] - ball_center[1])**2
    )

    real_ball_diameter = 22  
    scale_factor = ball_diameter / real_ball_diameter 

    if scale_factor == 0: 
        return None

    ankle_ball_dist_cm = ankle_ball_dist_px / scale_factor

    cv2.line(frame, ankle, ball_center, (0, 255, 255), 3)
    mid_point = ((ankle[0] + ball_center[0]) // 2,
                 (ankle[1] + ball_center[1]) // 2)
    cv2.putText(frame, f"{ankle_ball_dist_cm:.1f} cm", mid_point,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return ankle_ball_dist_cm

def get_normalized_heel_positions(box, right_heel, left_heel, foot):
    if box is None:
        return None, None
    x1, y1, x2, y2 = box
    box_height = y2 - y1
    box_width = x2 - x1

    if foot == "right" and right_heel is not None:
        heel = right_heel
        ref_side = "right"
    elif foot == "left" and left_heel is not None:
        heel = left_heel
        ref_side = "left"
    else:
        return None, None

    ankle_x, ankle_y = heel

    if box_width != 0:
        if foot == "right":
            ankle_x_percentage = ((ankle_x - x1) / box_width) * 100
        else:
            ankle_x_percentage = (1 - ((ankle_x - x1) / box_width)) * 100
    else:
        ankle_x_percentage = None

    if box_height != 0:
        ankle_y_percentage = ((ankle_y - y1) / box_height) * 100
    else:
        ankle_y_percentage = None

    return round(ankle_y_percentage, 1), round(ankle_x_percentage, 1)

def export_analysis_to_json(receiving_foot, head_direction, front_pelvis_angle, front_torso_angle,
                            ankle_arrows, normalized_receiving_ankle_position, side_supporting_knee_angle,
                            side_torso_angle_, heel_ball_y, heel_ball_x, right_ankle_values, left_ankle_values,
                            ankle_ball_dist_cm):  # 👈 أضف هذا الحقل
    analysis_data = {
        "receiving_foot": receiving_foot,
        "heel_ball_percentage_y": heel_ball_y,
        "heel_ball_percentage_x": heel_ball_x,
        "right_ankle_angle": right_ankle_values[0] if right_ankle_values else None,
        "right_ankle_distance": right_ankle_values[1] if right_ankle_values else None,
        "left_ankle_angle": left_ankle_values[0] if left_ankle_values else None,
        "left_ankle_distance": left_ankle_values[1] if left_ankle_values else None,
        "front_pelvis_angle": front_pelvis_angle,
        "front_torso_angle": front_torso_angle,
        "head_direction": head_direction,
        "ankle_arrows": ankle_arrows,
        "normalized_receiving_ankle_position": normalized_receiving_ankle_position, 
        "side_supporting_knee_angle": side_supporting_knee_angle,
        "side_torso_angle": side_torso_angle_,
        "timestamp": datetime.now().isoformat(),
        "ankle_ball_dist_cm": ankle_ball_dist_cm  # 👈 أضف هذا السطر
    }
    return analysis_data

def calculate_ankle_ground_dis(right_ankle,left_ankle,foot, box):#completed
    x1, y1, x2, y2 = box
    box_height = y2 - y1
    box_width = x2 - x1
    if foot=="right":
        
        ankle_x, ankle_y = right_ankle
    elif foot=="left":
        
        ankle_x, ankle_y = left_ankle
    

    if box_height != 0:
        ankle_y_percentage = ((ankle_y - y1) / box_height) * 100
    else:
        ankle_y_percentage = None  

    if box_width != 0:
        ankle_x_percentage = ((ankle_x - x1) / box_width) * 100
    else:
        ankle_x_percentage = None

        
    return ankle_y_percentage, ankle_x_percentage
    
init_head_pose_detector(up_threshold=-10, down_threshold=10)

def reset_detection_state():
    global initial_detections, determined_foot, detection_count
    global prev_distance, prev_frame, skip_frames
    global prev_pitch, prev_side_frame, chosen_side_frame
    initial_detections = []
    determined_foot = None
    detection_count = 0
    prev_distance = None
    prev_frame = None
    prev_side_frame = None
    chosen_side_frame = None
    skip_frames = 0
    prev_pitch = None


def analyze_videos(front_path, side_path):
    reset_detection_state()
    global model, pose, pose_side, prev_distance, prev_frame, skip_frames, prev_pitch, initial_detections, determined_foot, detection_count, prev_side_frame, chosen_side_frame

    cap = cv2.VideoCapture(front_path)
    cap2 = cv2.VideoCapture(side_path)
    
    results = []  
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  
        ret2, side_frame = cap2.read()
        if not ret2:
            break 
        frame_for_headpose = frame.copy()
        processed_frame, direction, pitch, yaw, roll, bbox = detect_head_pose(frame_for_headpose)  
         
        frame, center, ball_diameter, box = detect_ball(frame)
        side_frame, center_side, side_ball_diameter, side_box = detect_ball(side_frame)
        side_frame, person_center, person_diameter, best_box = detect_person(side_frame)
           
        right_ankle, left_ankle, right_heel, left_heel, finger_xy_l, finger_xy_r, left_hip, right_hip, left_shoulder, right_shoulder, left_knee, right_knee = process_pose_and_draw_arrows(frame, pose)
        
        side_right_ankle, side_left_ankle, side_right_heel, side_left_heel, side_finger_xy_l, side_finger_xy_r, side_left_hip, side_right_hip, side_left_shoulder, side_right_shoulder, side_left_knee, side_right_knee = process_pose_and_draw_arrows(side_frame, pose_side)
        foot = detect_receiving_foot(best_box, center_side)
        real_distance=calculate_ankle_ball_dist_and_draw(chosen_frame, right_ankle, left_ankle, center, ball_diameter, foot)

        if skip_frames > 0:
            skip_frames -= 1
            if ankle_xy_r is None or center == (0,0):
                continue
            
        receiving = check_receiving_position(side_frame, side_right_ankle, side_left_ankle, center_side, best_box)

        
        
        if receiving:
            if foot == "right":
                distance = math.sqrt(
                        (side_right_ankle[0] - center_side[0])**2 +
                        (side_right_ankle[1] - center_side[1])**2
                    )
            elif foot == "left":
                distance = math.sqrt(
                        (side_left_ankle[0] - center_side[0])**2 +
                        (side_left_ankle[1] - center_side[1])**2
                    ) 

            if prev_distance is None:
                prev_distance = distance
                prev_frame = frame.copy()
                prev_side_frame = side_frame.copy() 
            else:
                if distance < prev_distance:
                    chosen_frame = frame.copy()
                    chosen_side_frame = side_frame.copy() 
                else:
                    chosen_frame = prev_frame.copy()
                    chosen_side_frame = prev_side_frame.copy()             
                    
                r_a_a, r_a_d = draw_ankel_arrow(chosen_frame, right_ankle, finger_xy_r, hip=right_hip, knee=right_knee, color=(255,0,255))
                l_a_a, l_a_d = draw_ankel_arrow(chosen_frame, left_ankle, finger_xy_l, hip=left_hip, knee=left_knee, color=(255,0,0))
                
                pelvis_angle = calculate_horizontal_pelvis_angle(left_hip, right_hip)
                torso_angle = calculate_vertical_torso_angle(left_hip, right_hip, left_shoulder, right_shoulder)
                
                if right_hip and right_knee and right_ankle:
                    draw_knee_angle(chosen_frame, right_hip, right_knee, right_ankle, color=(0,255,255))
                if left_hip and left_knee and left_ankle:
                    draw_knee_angle(chosen_frame, left_hip, left_knee, left_ankle, color=(255,0,255))
                
                # Side process
                if side_right_hip and side_right_knee and side_right_ankle:
                    side_right_angle = draw_knee_angle(chosen_side_frame, side_right_hip, side_right_knee, side_right_ankle, color=(0,255,255))
                if side_left_hip and side_left_knee and side_left_ankle:
                    side_left_angle = draw_knee_angle(chosen_side_frame, side_left_hip, side_left_knee, side_left_ankle, color=(255,0,255))
                
                if side_left_hip and side_right_hip:
                    side_pelvis_angle = calculate_horizontal_pelvis_angle(side_left_hip, side_right_hip)
                
                if side_left_hip and side_right_hip and side_left_shoulder and side_right_shoulder:
                    side_torso_angle_ = calculate_vertical_torso_angle(side_left_hip, side_right_hip, side_left_shoulder, side_right_shoulder)
                
                processed_frame, direction, pitch, yaw, roll, bbox = detect_head_pose(frame_for_headpose) 
                
                # Calculate heel/ball positions
                if box is not None and right_heel is not None:
                    y_perc, x_perc = calculate_ankle_ground_dis(right_heel, left_heel, foot, box)
                    x_perc = abs(100 - x_perc)
                    print(f"Ankle Y position: {y_perc:.1f} % | X position: {x_perc:.1f} %")
                    cv2.putText(chosen_frame, f"Ankle Y: {y_perc:.1f}% | X: {x_perc:.1f}%", (50,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                else:
                    y_perc = None
                    x_perc = None
                    cv2.putText(chosen_frame, "Ball not detected - cannot calculate position", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                if box is not None and foot is not None:
                    y_perc_norm, x_perc_norm = get_normalized_heel_positions(box, right_heel, left_heel, foot)
                    normalized_receiving_ankle_position = {
                        "Y_position_percentage": y_perc_norm,
                        "X_position_percentage": x_perc_norm
                    }
                else:
                    normalized_receiving_ankle_position = None
                    
                ankle_arrows = {}  
                if right_ankle is not None and finger_xy_r is not None and right_hip is not None and right_knee is not None:
                    arrow_dist_r = int(math.sqrt(
                        (right_ankle[0] - finger_xy_r[0])**2 +
                        (right_ankle[1] - finger_xy_r[1])**2
                    ))
                    arrow_dir_r = (finger_xy_r[0] - right_ankle[0], finger_xy_r[1] - right_ankle[1])
                    arrow_angle_r = calculate_knee_angle(right_hip, right_knee, right_ankle)
                    ankle_arrows['right'] = {
                        'distance': arrow_dist_r,
                        'direction_vector': arrow_dir_r,
                        'knee_angle': arrow_angle_r
                    }
                else:
                    ankle_arrows['right'] = None

                if left_ankle is not None and finger_xy_l is not None and left_hip is not None and left_knee is not None:
                    arrow_dist_l = int(math.sqrt(
                        (left_ankle[0] - finger_xy_l[0])**2 +
                        (left_ankle[1] - finger_xy_l[1])**2
                    ))
                    arrow_dir_l = (finger_xy_l[0] - left_ankle[0], finger_xy_l[1] - left_ankle[1])
                    arrow_angle_l = calculate_knee_angle(left_hip, left_knee, left_ankle)
                    ankle_arrows['left'] = {
                        'distance': arrow_dist_l,
                        'direction_vector': arrow_dir_l,
                        'knee_angle': arrow_angle_l
                    }
                else:
                    ankle_arrows['left'] = None

                if foot == "right":
                    supporting_knee_angle = side_left_angle if 'side_left_angle' in locals() else None
                else:
                    supporting_knee_angle = side_right_angle if 'side_right_angle' in locals() else None

                calculate_ankle_ball_dist_and_draw(chosen_frame, right_ankle, left_ankle, center, ball_diameter, foot)
                              
                chosen_frame = cv2.resize(chosen_frame, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_CUBIC)
                chosen_side_frame = cv2.resize(chosen_side_frame, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_CUBIC)
                
                analysis_dict = export_analysis_to_json(
    foot, direction, pelvis_angle, torso_angle,
    ankle_arrows, normalized_receiving_ankle_position,
    supporting_knee_angle, side_torso_angle_,
    y_perc, x_perc, (r_a_a, r_a_d), (l_a_a, l_a_d),
    real_distance  
)
                
                results.append({
                    "analysis": analysis_dict,
                    "front_frame": chosen_frame,
                    "side_frame": chosen_side_frame
                })
                
                prev_distance = None
                prev_frame = None
                prev_side_frame = None
                skip_frames = 50
    cap.release()
    cap2.release()
    
    if not results:
        raise ValueError("No receiving actions detected in the videos.")
    
    return results
