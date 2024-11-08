import os

import cv2
from ultralytics.data.converter import convert_dota_to_yolo_obb
from ultralytics.utils import TQDM

#convert_dota_to_yolo_obb(r'C:\Users\ym\Desktop\离职code\data\电容\labelTxt')

class_mapping = {
    "line": 0,

}
dota_root_path= (r"C:\Users\lhb\Desktop\labelTxt")
def check_file(path):
    """
    检查文件是否存在，没有就创建
    Args:
        path：需要检测的文件

    """

    if not os.path.exists(path):
        print(path)
        with open(path, mode='w', encoding='utf-8') as ff:
            print("文件创建成功！")
def convert_label(image_name, image_width, image_height, orig_label_dir, save_dir):
    """Converts a single image's DOTA annotation to YOLO OBB format and saves it to a specified directory."""
    orig_label_path = orig_label_dir  +"\\"+ f"{image_name}.txt"
    save_path = save_dir  +"\\"+ f"{image_name}.txt"
    check_file(save_path)
    with open(orig_label_path, "r") as f,open(save_path, "w") as g:
        lines = f.readlines()
        for line in lines:

            parts = line.strip().split()
            if len(parts) < 9:
                continue
            class_name = parts[8]
            class_idx = class_mapping[class_name]
            coords = [float(p) for p in parts[:8]]
            normalized_coords = [
                coords[i] / image_width if i % 2 == 0 else coords[i] / image_height for i in range(8)
            ]
            formatted_coords = ["{:.6g}".format(coord) for coord in normalized_coords]
            g.write(f"{class_idx} {' '.join(formatted_coords)}\n")


test_root = r"C:\Users\lhb\Desktop\labelTxt\images\train"
save_dir = r"C:\Users\lhb\Desktop\labelTxt\labels\train"
label_root = r"C:\Users\lhb\Desktop\labelTxt\labels\train_original"
for file in sorted(os.listdir(test_root)):
    if file.endswith('json'):
        continue
    test_image = os.path.join(test_root, file)
    label =file.split(".")[0]+".txt"
    print(label)
    convert_label(file.split(".")[0], 1920, 1200, label_root, save_dir)