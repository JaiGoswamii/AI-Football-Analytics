import cv2
import os

def extract_frames(video_path, output_dir, frame_rate=5):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_rate == 0:
            cv2.imwrite(os.path.join(output_dir, f"frame_{frame_idx}.jpg"), frame)
        frame_idx += 1
    cap.release()

if __name__ == "__main__":
    # Change <YOUR_USERNAME> to your actual username on macOS:
    video_path = "/Users/macbook/Desktop/Football_Analysis_CV/08fd33_4.mp4"
    output_dir = "/Users/macbook/Desktop/Football_Analysis_CV/frames_output"

    extract_frames(video_path, output_dir, frame_rate=5)
