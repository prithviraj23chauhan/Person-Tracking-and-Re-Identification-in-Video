import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from reid_utils import extract_features, save_features_db
from utils import config

def run_tracking(video_path):
    model = YOLO(config["model_path"])
    deepsort = DeepSort(
        max_age=config["deepsort"]["max_age"],
        n_init=config["deepsort"]["n_init"],
        nn_budget=config["deepsort"]["nn_budget"]
    )
    cap = cv2.VideoCapture(video_path)

    selected_id = None
    features_db = {}
    current_bboxes = []

    def on_mouse(event, x, y, flags, param):
        nonlocal selected_id
        if event == cv2.EVENT_LBUTTONDOWN:
            for track_id, bbox in current_bboxes:
                x1, y1, x2, y2 = map(int, bbox)
                if x1 <= x <= x2 and y1 <= y <= y2:
                    selected_id = track_id
                    print(f"Selected ID: {selected_id}")
                    break

    cv2.namedWindow("Tracking")
    cv2.setMouseCallback("Tracking", on_mouse)
    print("Press 's' to save features for selected ID, 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=config["logging"]["verbose"])
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
        trackers = deepsort.update_tracks(raw_detections, frame=frame)
        current_bboxes = []
        for track in trackers:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            track_id = track.track_id
            bbox = track.to_ltrb()
            current_bboxes.append((track_id, bbox))
            color = (0, 255, 0) if track_id == selected_id else (255, 0, 0)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.putText(frame, f"ID: {track_id}", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow("Tracking", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and selected_id is not None:
            for track_id, bbox in current_bboxes:
                if track_id == selected_id:
                    features = extract_features(frame, bbox)
                    if features is not None:
                        features_db[track_id] = features
                        save_features_db(features_db)
                        print(f"Features for ID {track_id} saved.")
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            return
    cap.release()
    cv2.destroyAllWindows()


# import cv2
# import torch
# from ultralytics import YOLO
# import numpy as np
# from torchvision.models import resnet18, ResNet18_Weights
# from deep_sort_realtime.deepsort_tracker import DeepSort
# import pickle 

# # Initialize the person ReID model
# reid_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
# reid_model.fc = torch.nn.Identity()  # Remove classification head
# reid_model.eval()

# # Initialize DeepSORT Tracker
# deepsort = DeepSort(max_age=30, n_init=3, nn_budget=100)
# model = YOLO("yolov8s.pt") 


# # Open the video feed
# cap = cv2.VideoCapture('PROJECT\\miniproject3\\853889-hd_1920_1080_25fps.mp4') 

# # track ID of the person to re-identify
# target_track_id = None

# # dictionary to store tracking paths
# tracking_paths = {}
# path_enabled = False
# stored_features = None

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # YOLOv8 
#     results = model(frame)  
#     detections = results[0] # person
    
#     raw_detections = []
#     if detections.boxes is not None:
#         for box in detections.boxes:
#             if box.xyxy.shape[1] == 4:  
#                 xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
#                 confidence = box.conf.item()
#                 class_id = int(box.cls.item())

#                 width = xmax - xmin
#                 height = ymax - ymin
#                 bbox_xywh = [xmin, ymin, width, height]
                
#                 raw_detections.append((bbox_xywh, confidence, "person"))

#     # update tracker with new detections
#     trackers = deepsort.update_tracks(raw_detections, frame=frame)

#     for track in trackers:
#         if not track.is_confirmed() or track.time_since_update > 1:
#             continue

#         track_id = track.track_id
#         bbox = track.to_ltrb()  # Convert to [left, top, right, bottom] format
#         x1, y1, x2, y2 = map(int, bbox)

        
        
        
#         #  draw bounding box 
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
#         # update tracking path
#         center_x = int((x1 + x2) / 2)
#         center_y = int((y1 + y2) / 2)
#         if track_id not in tracking_paths:
#             tracking_paths[track_id] = []
#         tracking_paths[track_id].append((center_x, center_y))

#         # draw the tracking path
#         if path_enabled and track_id in tracking_paths:
#             path = tracking_paths[track_id]
#             for i in range(1, len(path)):
#                 if path[i - 1] is None or path[i] is None:
#                     continue
#                 cv2.line(frame, path[i - 1], path[i],(255, 0, 0),2, )

#         if target_track_id is not None and int(target_track_id) == int(track_id):
            
#             # extract features from the ReID model for the selected person
#             cropped_person = frame[y1:y2, x1:x2]
#             cropped_person = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)
#             cropped_person = cv2.resize(cropped_person, (224, 224))
#             cropped_person_tensor = torch.tensor(np.transpose(cropped_person, (2, 0, 1)), dtype=torch.float32) / 255.0
#             cropped_person_tensor = cropped_person_tensor.unsqueeze(0) 

#             with torch.no_grad():
#                 features = reid_model(cropped_person_tensor).cpu().numpy()

#             # store the features for the given track ID
#             stored_features = features
#             print(f" \n\n\n #### Stored features of track id :{target_track_id}  ####\n\n\n")
#             target_track_id = None

#     # display the frame with bounding boxes around the identified person
#     cv2.imshow("Tracking", frame)

#     # Exit on 'q' 
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

#     if cv2.waitKey(1) & 0xFF == ord('i'):
#         target_track_id = input("enter id: ")

#     if cv2.waitKey(1) & 0xFF == ord('t'):
#         path_enabled = not path_enabled
#         print(f"Path drawing {'enabled' if path_enabled else 'disabled'}")



# cap.release()
# cv2.destroyAllWindows()

# # save the features to a file
# with open("PROJECT\\miniproject3features.pkl", "wb") as f:
#     if stored_features is not None:
#         pickle.dump(stored_features, f)
#     print("Stored features saved to features.pkl ")
#     print(target_track_id)
