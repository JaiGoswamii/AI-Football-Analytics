from ultralytics import YOLO

def main():
    # Path to your data.yaml (adjust if it's located elsewhere)
    data_config = "dataset/data.yaml"
    
    # 1. Load the pretrained YOLOv8-nano model weights
    model = YOLO("yolov8n.pt")
    
    # 2. Train the model
    model.train(
        data=data_config,   # path to data.yaml
        epochs=50,          # number of training epochs
        imgsz=640,          # image size
        batch=8,            # batch size (reduce if you get OOM errors)
        name="yolov8n_football",  # folder name inside 'runs/'
        project="runs"      # parent folder for all training outputs
    )

if __name__ == "__main__":
    main()
