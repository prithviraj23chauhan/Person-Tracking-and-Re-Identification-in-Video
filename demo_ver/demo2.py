import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize the YOLOv8 model
model = YOLO("yolov8s.pt")  # Make sure the model file is available

# Initialize the DeepSORT tracker
deepsort = DeepSort(max_age=30, n_init=3, nn_budget=100)

# Open the video feed
cap = cv2.VideoCapture('PROJECT\\miniproject3\\853889-hd_1920_1080_25fps.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection using YOLOv8
    results = model(frame)
    detections = results[0]

    # Prepare raw_detections
    raw_detections = []
    if detections.boxes is not None:
        for box in detections.boxes:
            # Ensure the box has valid coordinates
            if box.xyxy.shape[1] == 4:  # Check for proper shape
                # Extract bounding box coordinates, confidence, and class ID
                xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
                confidence = box.conf.item()
                class_id = int(box.cls.item())

                # Convert to [left, top, width, height] format
                width = xmax - xmin
                height = ymax - ymin
                bbox_xywh = [xmin, ymin, width, height]

                # Append detection as (bbox, confidence, class)
                raw_detections.append((bbox_xywh, confidence, "person"))  # Replace "person" if needed


    # Update tracker with raw_detections
    trackers = deepsort.update_tracks(raw_detections, frame=frame)

    # Draw tracked objects
    for track in trackers:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        # Get track ID and bounding box
        track_id = track.track_id
        bbox = track.to_ltrb()  # Convert to [left, top, right, bottom] format
        x1, y1, x2, y2 = map(int, bbox)

        # Draw bounding box and track ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Tracking", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
