from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8n.pt")
    model.train(data="D:/yolo/ultralytics-main/ultralytics-main/ultralytics/remote_sensing/cfg/data_remote.yaml", epochs=30)
    result = model.val()

