import cv2
import torch
from ultralytics import YOLO
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.spatial.distance import cosine
import pickle  

# the person ReID model
reid_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
reid_model.fc = torch.nn.Identity()  
reid_model.eval()

# initializing DeepSORT Tracker
deepsort = DeepSort(max_age=30, n_init=3, nn_budget=100)
model = YOLO("yolov8s.pt") 

cap = cv2.VideoCapture('PROJECT\miniproject3\853889-hd_1920_1080_25fps.mp4')  

# Load stored features from the file
with open("PROJECT\miniproject3features.pkl", "rb") as f:
    stored_features = pickle.load(f)
    print("Loaded stored features from features.pkl")

#dictionary to store track IDs and features
tracked_persons = {}


frame_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
  
    results = model(frame)  
    detections = results[0]
    
    raw_detections = []
    if detections.boxes is not None:
        for box in detections.boxes:
            if box.xyxy.shape[1] == 4:  
                xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
                confidence = box.conf.item()
                class_id = int(box.cls.item())

                width = xmax - xmin
                height = ymax - ymin
                bbox_xywh = [xmin, ymin, width, height]
                
                raw_detections.append((bbox_xywh, confidence, "person"))

    # update tracker with raw detections
    trackers = deepsort.update_tracks(raw_detections, frame=frame)

    for track in trackers:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        track_id = track.track_id
        bbox = track.to_ltrb()  # Convert to [left, top, right, bottom] format
        x1, y1, x2, y2 = map(int, bbox)

        # Check similarity every 10th frame
        if True or frame_counter % 10 == 1:
            # Extract features from the ReID model for the detected person
            cropped_person = frame[y1:y2, x1:x2]
            try:
                cropped_person = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)
                cropped_person = cv2.resize(cropped_person, (224, 224))
            except:
                break
            cropped_person_tensor = torch.tensor(np.transpose(cropped_person, (2, 0, 1)), dtype=torch.float32) / 255.0
            cropped_person_tensor = cropped_person_tensor.unsqueeze(0)  

            with torch.no_grad():
                features = reid_model(cropped_person_tensor).cpu().numpy()

            # normalizing features
            features = features.flatten() / np.linalg.norm(features)
            stored_features_normalized = stored_features.flatten() / np.linalg.norm(stored_features)

            # compare with stored features using cosine similarity
            similarity = cosine(features, stored_features_normalized)

            # store track ID and similarity 
            tracked_persons[track_id] = {"features": features, "similarity": similarity}

     

        if track_id in tracked_persons:
            similarity = tracked_persons[track_id]["similarity"]

             
            similarity = max(0, min(similarity, 1))  

            similarity_normalized = (similarity - 0.10) / (0.35 - 0.10)  
            
            #  red (low similarity value) and blue (high similarity value)
            red_value = int((1 - similarity_normalized) * 255)  
            blue_value = int(similarity_normalized * 255)      

            
            color = (blue_value, 0, red_value)  #bgr

            # drawing rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {track_id} - {similarity:.4f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

   
    frame_counter += 1

    cv2.imshow("Person Tracking & Re-ID", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
