import cv2
import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
config = load_config()
    
def draw_bbox(frame, bbox, color, track_id, similarity=None):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID: {track_id}"
    if similarity is not None:
        label += f" - {similarity:.4f}"
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def draw_legend(frame):
    instructions = [
        "Click box: Select person",
        "'q': Quit"
    ]
    for i, text in enumerate(instructions):
        cv2.putText(frame, text, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)


def get_similarity_color(similarity):
    """
    Maps similarity to a red (low) to blue (high) gradient.
    similarity is expected in [0, 1].
    """
    # Clamp similarity to [0, 1]
    similarity = max(0, min(similarity, 1))

    # Normalize to your chosen range (e.g., 0.10 to 0.35)
    min_sim = config["similarity"]["min"]
    max_sim = config["similarity"]["max"]   
    similarity_normalized = (similarity - min_sim) / (max_sim - min_sim)
    similarity_normalized = max(0, min(similarity_normalized, 1))  # Clamp to [0, 1]

    red_value = int((1 - similarity_normalized) * 255)
    blue_value = int(similarity_normalized * 255)
    green_value = 0  # You can add green for purple/cyan gradients if you want

    return (blue_value, green_value, red_value)