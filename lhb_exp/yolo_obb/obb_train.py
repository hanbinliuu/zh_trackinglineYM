from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO(r'D:\pycharmProject\zh_trackingposition\lhb_exp\yolo_obb\yolov8n-obb.pt')
    results = model.train(data='D:\pycharmProject\zh_trackingposition\lhb_exp\yolo_obb\dota8-obb.yaml',
                          epochs=800, imgsz=640, device='0')

