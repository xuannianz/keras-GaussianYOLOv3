import numpy as np
import cv2


def get_anchors(anchors_path):
    """
    loads the anchors from a txt file
    """
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    # (N, 2), wh
    return np.array(anchors).reshape(-1, 2)


def get_classes(classes_path):
    """
    loads the classes form a txt file
    """
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def preprocess_image(image, image_size=416):
    image_height, image_width = image.shape[:2]
    if image_height > image_width:
        scale = image_size / image_height
        resized_height = image_size
        resized_width = int(image_width * scale)
    else:
        scale = image_size / image_width
        resized_height = int(image_height * scale)
        resized_width = image_size
    image = cv2.resize(image, (resized_width, resized_height))
    new_image = np.ones((image_size, image_size, 3), dtype=np.float32) * 128.
    offset_h = (image_size - resized_height) // 2
    offset_w = (image_size - resized_width) // 2
    new_image[offset_h:offset_h + resized_height, offset_w:offset_w + resized_width] = image.astype(np.float32)
    new_image /= 255.
    return new_image
