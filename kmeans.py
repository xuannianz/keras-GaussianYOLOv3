import numpy as np


class YOLOKmeans:
    def __init__(self, cluster_number, annotation_paths, anchors_path, input_size=(416, 416)):
        self.cluster_number = cluster_number
        self.annotation_paths = annotation_paths
        self.anchors_path = anchors_path
        # (h, w)
        self.input_size = input_size

    def iou(self, boxes, clusters):
        n = boxes.shape[0]
        k = self.cluster_number
        # (n, )
        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        # init k clusters
        clusters = boxes[np.random.choice(box_number, k, replace=False)]
        while True:
            distances = 1 - self.iou(boxes, clusters)
            current_nearest = np.argmin(distances, axis=1)
            # clusters won't change
            if (last_nearest == current_nearest).all():
                break
            for cluster in range(k):
                clusters[cluster] = dist(boxes[current_nearest == cluster], axis=0)
            print('clusters={}'.format(clusters))
            last_nearest = current_nearest

        return clusters

    def result2txt(self, data):
        f = open(self.anchors_path, 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2boxes(self):
        dataset = []
        for annotation_path in self.annotation_paths:
            f = open(annotation_path, 'r')
            for line in f:
                infos = line.split(" ")
                length = len(infos)
                for i in range(1, length):
                    width = int(infos[i].split(",")[2]) - int(infos[i].split(",")[0])
                    height = int(infos[i].split(",")[3]) - int(infos[i].split(",")[1])
                    dataset.append([width, height])
            f.close()
        result = np.array(dataset)
        return result

    def resize_bbox(self, bbox, image_shape):
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        scale = min(self.input_size[0] / image_shape[0], self.input_size[1] / image_shape[1])
        return round(bbox_width * scale), round(bbox_height * scale)

    def txt2clusters(self):
        all_boxes = self.txt2boxes()
        # (num_clusters, 2) wh
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        # self.result2txt(result)
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(self.avg_iou(all_boxes, result) * 100))


if __name__ == "__main__":
    num_clusters = 9
    annotation_paths = [
        "/home/adam/.keras/datasets/VOCdevkit/trainval/train.txt",
        "/home/adam/.keras/datasets/VOCdevkit/test/test.txt"
    ]
    kmeans = YOLOKmeans(num_clusters, annotation_paths, 'voc_anchors_416.txt')
    kmeans.txt2clusters()
