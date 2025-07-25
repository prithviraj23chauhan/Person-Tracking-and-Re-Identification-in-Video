import cv2
import torch
from ultralytics import YOLO
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
from deep_sort_realtime.deepsort_tracker import DeepSort
import random
import pickle  # For saving to file

# Initialize the person ReID model
reid_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
reid_model.fc = torch.nn.Identity()  # Remove classification head
reid_model.eval()

# Initialize DeepSORT Tracker
deepsort = DeepSort(max_age=30, n_init=3, nn_budget=100)
model = YOLO("yolov8s.pt") 

# Dictionary to store the features of identified people

# Open the video feed
cap = cv2.VideoCapture('PROJECT\\miniproject3\\853889-hd_1920_1080_25fps.mp4')  # Replace with your video path

# Define a variable to store the track ID of the person to re-identify
target_track_id = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform detection using YOLOv8 or any other detection model (assuming you have detections)
    results = model(frame)  # Replace with YOLOv8 or any other model you're using
    detections = results[0]
    
    raw_detections = []
    if detections.boxes is not None:
        for box in detections.boxes:
            if box.xyxy.shape[1] == 4:  # Check for proper shape
                xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
                confidence = box.conf.item()
                class_id = int(box.cls.item())

                width = xmax - xmin
                height = ymax - ymin
                bbox_xywh = [xmin, ymin, width, height]
                
                raw_detections.append((bbox_xywh, confidence, "person"))

    # Update tracker with raw detections
    trackers = deepsort.update_tracks(raw_detections, frame=frame)

    for track in trackers:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        track_id = track.track_id
        

        bbox = track.to_ltrb()  # Convert to [left, top, right, bottom] format
        x1, y1, x2, y2 = map(int, bbox)

        # Store track ID of the person to track
        
        
        # If it's the identified person, draw bounding box around them
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if  target_track_id is not None and int(target_track_id) == int(track_id):

            # Now store the features of this person
            # Extract features from the ReID model for the selected person
            cropped_person = frame[y1:y2, x1:x2]
            cropped_person = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)
            cropped_person = cv2.resize(cropped_person, (224, 224))
            cropped_person_tensor = torch.tensor(np.transpose(cropped_person, (2, 0, 1)), dtype=torch.float32) / 255.0
            cropped_person_tensor = cropped_person_tensor.unsqueeze(0)  # Add batch dimension

            with torch.no_grad():
                features = reid_model(cropped_person_tensor).cpu().numpy()

            # Store the features for the given track ID
            stored_features = features
            print(f" \n\n\n #### Stored features of track id :{target_track_id}  ####\n\n\n")
            target_track_id = None

            

    # Display the frame with bounding boxes around the identified person
    cv2.imshow("Tracking", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('g'):
        target_track_id = input("enter id: ")


# Release resources
cap.release()
cv2.destroyAllWindows()

# Save the features to a file
with open("PROJECT\miniproject3features.pkl", "wb") as f:
    pickle.dump(stored_features, f)
    print("Stored features saved to features.pkl ")
    print(target_track_id)
