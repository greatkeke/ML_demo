import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

image = Image.open('dog_bike_car.png')
image = image.resize((800, 600))
image_tensor = F.to_tensor(image)
with torch.no_grad():
    prediction = model([image_tensor])

labels = prediction[0]['labels']
scores = prediction[0]['scores']
boxes = prediction[0]['boxes']
# 筛选分数大于0.7的预测
threshold = 0.7
labels = labels[scores > threshold]
boxes = boxes[scores > threshold].cpu().numpy()
scores = scores[scores > threshold].cpu().numpy()

COCO_CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife',
    'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
# 使用PIL在图像上绘制边界框
draw = ImageDraw.Draw(image)
for box, label, score in zip(boxes, labels, scores):
    class_name = COCO_CLASSES[label.item()]
    draw.rectangle(box, outline='yellow', width=3)
    draw.text((box[0], box[1]), text=f'{class_name} {score:.2f}', fill='blue')
# 绘制最终结果图
plt.imshow(image)
plt.axis('off')
plt.show()