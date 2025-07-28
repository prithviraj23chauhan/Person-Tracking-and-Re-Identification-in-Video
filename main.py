from tracking import run_tracking
from reidentify import run_reidentification
import pickle
from utils import config

def main():
    print("Person Tracking & Re-Identification")
    print("1. Track and extract features (Video 1)")
    print("2. Re-identify in another video (Video 2)")

    # Set your default video paths here
    tracking_video_path = config["tracking_video_path"]
    reid_video_path = config["reid_video_path"]
    while True:
        choice = input("Enter your choice (1/2): ").strip()
        if choice == '1':
            print(f"Using default tracking video: {tracking_video_path}")
            run_tracking(tracking_video_path)
        elif choice == '2':
            print(f"Using default re-identification video: {reid_video_path}")
            run_reidentification(reid_video_path)
        else:
            features_file = config.get("features_db_path", "D:/vs code/PROJECT/miniproject3/features/features.pkl")            
            print("Invalid choice.")
            with open(features_file, "wb") as f:
                pickle.dump(features_file, f)
            print("All stored features have been cleared.")
            break
if __name__ == "__main__":
    main()