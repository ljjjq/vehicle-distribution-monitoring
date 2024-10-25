from ultralytics import YOLO
from PIL import Image

model = YOLO("./runs/detect/train/weights/best.pt") # 训练好的模型路径
img = Image.open("./remote_data/test/images/038.jpg") # 要预测的图像路径
results = model.predict(source=img, show=True, save=True, save_txt=True)  # 展示并保存绘制的图像和619