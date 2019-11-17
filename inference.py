import cv2
import glob
import numpy as np
import os
import os.path as osp
from utils import get_anchors, get_classes, preprocess_image
from model import yolo_body

input_shape = (416, 416)
anchors = get_anchors('voc_anchors_416.txt')
classes = get_classes('voc_classes.txt')
num_classes = len(classes)
model, prediction_model = yolo_body(anchors=anchors, score_threshold=0.1)
model.load_weights('checkpoints/pascal_21_9.4463_12.8289_0.8334_0.8535.h5', by_name=True)
batch_size = 1
image_paths = glob.glob('datasets/VOC2007/JPEGImages/*.jpg')
num_images = len(image_paths)
colors = [np.random.randint(0, 256, 3).tolist() for i in range(num_classes)]


def show_image(image, name, contours=None):
    image = image.astype(np.uint8)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    if contours is not None:
        if isinstance(contours, list):
            cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
        else:
            cv2.drawContours(image, [contours], -1, (0, 0, 255), 2)
    cv2.imshow(name, image)


for i in range(0, num_images, batch_size):
    if i + batch_size > num_images:
        batch_image_paths = image_paths[i:]
    else:
        batch_image_paths = image_paths[i:i + batch_size]
    batch_images_data = []
    batch_image_shapes = []
    for image_path in batch_image_paths:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_shape = image.shape[:2]
        image_shape = np.array(image_shape)
        image_data = preprocess_image(image)
        batch_images_data.append(image_data)
        batch_image_shapes.append(image_shape)

    batch_images_data = np.array(batch_images_data)
    batch_image_shapes = np.array(batch_image_shapes)
    batch_detections = prediction_model.predict([batch_images_data, batch_image_shapes])
    for i, detections in enumerate(batch_detections):
        image_path = batch_image_paths[i]
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        detections = detections[detections[:, 4] > 0.0]
        for detection in detections:
            ymin = max(int(round(detection[0])), 0)
            xmin = max(int(round(detection[1])), 0)
            ymax = min(int(round(detection[2])), h - 1)
            xmax = min(int(round(detection[3])), w - 1)
            score = '{:.4f}'.format(detection[4])
            class_id = int(detection[5])
            color = colors[class_id - 1]
            class_name = classes[class_id]
            label = '-'.join([class_name, score])
            ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
            cv2.rectangle(image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
            cv2.putText(image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', image)
        key = cv2.waitKey(0)
        if int(key) == 121:
            image_fname = osp.split(image_path)[-1]
            cv2.imwrite('test/{}'.format(image_fname), image)
