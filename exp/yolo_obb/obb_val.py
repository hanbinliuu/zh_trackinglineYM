from ultralytics import YOLO
import cv2


model = YOLO(r"C:\Users\lhb\Desktop\zh_trackingposition\lhb_exp\yolo_obb\best.pt")
img = cv2.imread(r"C:\Users\lhb\Desktop\zhpic\Image_20240309152910223.bmp")
results = model.predict(img, device='0', save=True)
