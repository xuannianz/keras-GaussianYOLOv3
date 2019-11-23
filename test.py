import cv2
import glob
import numpy as np
import os
import os.path as osp
from utils import get_anchors, get_classes, preprocess_image
from model import yolo_body
import time
import copy

os.environ["CUDA_VISIBLE"]="0"
input_shape = (416, 416)
anchors = get_anchors('voc_anchors_416.txt')
classes = get_classes('voc_classes.txt')
num_classes = len(classes)
model, prediction_model = yolo_body(anchors=anchors, score_threshold=0.1)
# model.summary()
model.load_weights('/home/beast/Desktop/keras-GaussianYOLOv3/pascal_21_9.4463_12.8289_0.8334_0.8535.h5', by_name=True)
# batch_size = 1
# image_paths = glob.glob('/media/beast/akhilesh/daniel_panoptc/*.jpg')
# num_images = len(image_paths)
colors = [np.random.randint(0, 256, 3).tolist() for i in range(num_classes)]


cap = cv2.VideoCapture("/home/beast/Desktop/keras-GaussianYOLOv3/VID_030.mp4")
out = cv2.VideoWriter('test.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (416,416))
count=0
while 1:
    start = time.time()
    
    #Read Franes
    ret, frame = cap.read()

    
    if ret != 1:
        break
    #Preprocessing of input1

    input_image = cv2.resize(frame, (416, 416))
    input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
    
    image = copy.deepcopy(input_image)
    
    image_shape = input_image.shape
    input_image =input_image
    
    image_shape = input_image.shape[:2]

    image_data = preprocess_image(input_image)
    image_data = np.expand_dims(image_data,axis=0)
    batch_images_data= np.array(image_data)
    batch_image_shapes=np.array([image_shape])
    
    
    batch_detections = prediction_model.predict([batch_images_data, batch_image_shapes])
    
    for i, detections in enumerate(batch_detections):
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
        # elapsed=time.time()-start
        # cv2.putText(image,1/elapsed,(10,10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        out.write(image)
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.imshow('image', image)
        # key = cv2.waitKey(1)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elapsed=time.time()-start
    
    # Update number of frames and print FPS
    print("FPS : ", 1/elapsed, end = '\r')
cap.release()
out.release()
cv2.destroyAllWindows()
# echo 'export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}' >> ~/.bashrc
# echo 'export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
# sudo apt-get install -y --no-install-recommends libnvinfer6=6.0.1+cuda10.1 \
#     libnvinfer-dev=6.0.1-1+cuda10.1