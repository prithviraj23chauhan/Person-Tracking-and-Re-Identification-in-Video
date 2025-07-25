import cv2
import torch
from ultralytics import YOLO
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
from deep_sort_realtime.deepsort_tracker import DeepSort
import pickle 

# Initialize the person ReID model
reid_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
reid_model.fc = torch.nn.Identity()  # Remove classification head
reid_model.eval()

# Initialize DeepSORT Tracker
deepsort = DeepSort(max_age=30, n_init=3, nn_budget=100)
model = YOLO("yolov8s.pt") 


# Open the video feed
cap = cv2.VideoCapture('PROJECT\\miniproject3\\853889-hd_1920_1080_25fps.mp4') 

# track ID of the person to re-identify
target_track_id = None

# dictionary to store tracking paths
tracking_paths = {}
path_enabled = False
stored_features = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # YOLOv8 
    results = model(frame)  
    detections = results[0] # person
    
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

    # update tracker with new detections
    trackers = deepsort.update_tracks(raw_detections, frame=frame)

    for track in trackers:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        track_id = track.track_id
        bbox = track.to_ltrb()  # Convert to [left, top, right, bottom] format
        x1, y1, x2, y2 = map(int, bbox)

        
        
        
        #  draw bounding box 
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # update tracking path
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        if track_id not in tracking_paths:
            tracking_paths[track_id] = []
        tracking_paths[track_id].append((center_x, center_y))

        # draw the tracking path
        if path_enabled and track_id in tracking_paths:
            path = tracking_paths[track_id]
            for i in range(1, len(path)):
                if path[i - 1] is None or path[i] is None:
                    continue
                cv2.line(frame, path[i - 1], path[i],(255, 0, 0),2, )

        if target_track_id is not None and int(target_track_id) == int(track_id):
            
            # extract features from the ReID model for the selected person
            cropped_person = frame[y1:y2, x1:x2]
            cropped_person = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)
            cropped_person = cv2.resize(cropped_person, (224, 224))
            cropped_person_tensor = torch.tensor(np.transpose(cropped_person, (2, 0, 1)), dtype=torch.float32) / 255.0
            cropped_person_tensor = cropped_person_tensor.unsqueeze(0) 

            with torch.no_grad():
                features = reid_model(cropped_person_tensor).cpu().numpy()

            # store the features for the given track ID
            stored_features = features
            print(f" \n\n\n #### Stored features of track id :{target_track_id}  ####\n\n\n")
            target_track_id = None

    # display the frame with bounding boxes around the identified person
    cv2.imshow("Tracking", frame)

    # Exit on 'q' 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('i'):
        target_track_id = input("enter id: ")

    if cv2.waitKey(1) & 0xFF == ord('t'):
        path_enabled = not path_enabled
        print(f"Path drawing {'enabled' if path_enabled else 'disabled'}")



cap.release()
cv2.destroyAllWindows()

# save the features to a file
with open("PROJECT\\miniproject3features.pkl", "wb") as f:
    if stored_features is not None:
        pickle.dump(stored_features, f)
    print("Stored features saved to features.pkl ")
    print(target_track_id)
