import tensorflow as tf
import keras.backend as K
import keras.engine as KE

from util_graphs import correct_boxes_and_scores_graph


class DetectionLayer(KE.Layer):
    def __init__(self,
                 anchors,
                 num_classes=20,
                 max_boxes_per_class_per_image=20,
                 score_threshold=.2,
                 iou_threshold=.5,
                 max_boxes_per_image=400,
                 **kwargs
                 ):
        super(DetectionLayer, self).__init__(**kwargs)
        self.anchors = anchors
        self.iou_threshold = iou_threshold
        self.max_boxes_per_class_per_image = max_boxes_per_class_per_image
        self.max_boxes_per_image = max_boxes_per_image
        self.num_classes = num_classes
        self.score_threshold = score_threshold

    def call(self, inputs, **kwargs):
        yolo_outputs = inputs[:-1]
        batch_image_shape = inputs[-1]
        num_output_layers = len(yolo_outputs)
        num_anchors_per_layer = len(self.anchors) // num_output_layers
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        # tensor, (2, )
        input_shape = K.shape(yolo_outputs[0])[1:3] * 32
        grid_shapes = [K.shape(yolo_outputs[l])[1:3] for l in range(num_output_layers)]
        boxes_all_layers = []
        scores_all_layers = []
        for l in range(num_output_layers):
            yolo_output = yolo_outputs[l]
            grid_shape = grid_shapes[l]
            raw_y_pred = K.reshape(yolo_output,
                                   [-1, grid_shape[0], grid_shape[1], num_anchors_per_layer, self.num_classes + 9])
            boxes_this_layer, scores_this_layer = correct_boxes_and_scores_graph(raw_y_pred,
                                                                                 self.anchors[anchor_mask[l]],
                                                                                 self.num_classes,
                                                                                 input_shape,
                                                                                 batch_image_shape,
                                                                                 )
            boxes_all_layers.append(boxes_this_layer)
            scores_all_layers.append(scores_this_layer)

        # (b, total_num_anchors_all_layers, 4)
        boxes = K.concatenate(boxes_all_layers, axis=1)
        # (b, total_num_anchors_all_layers, num_classes)
        scores = K.concatenate(scores_all_layers, axis=1)
        mask = scores >= self.score_threshold
        max_boxes_per_class_per_image_tensor = K.constant(self.max_boxes_per_class_per_image, dtype='int32')
        max_boxes_per_image_tensor = K.constant(self.max_boxes_per_image, dtype='int32')

        def evaluate_batch_item(batch_item_boxes, batch_item_scores, batch_item_mask):
            boxes_per_class = []
            scores_per_class = []
            class_ids_per_class = []
            for c in range(self.num_classes):
                class_boxes = tf.boolean_mask(batch_item_boxes, batch_item_mask[:, c])
                # (num_keep_this_class_boxes, )
                class_scores = tf.boolean_mask(batch_item_scores[:, c], batch_item_mask[:, c])
                nms_keep_indices = tf.image.non_max_suppression(class_boxes,
                                                                class_scores,
                                                                max_boxes_per_class_per_image_tensor,
                                                                iou_threshold=self.iou_threshold)
                class_boxes = K.gather(class_boxes, nms_keep_indices)
                class_scores = K.gather(class_scores, nms_keep_indices)
                # (num_keep_this_class_boxes, )
                class_class_ids = K.ones_like(class_scores, 'float32') * c
                boxes_per_class.append(class_boxes)
                scores_per_class.append(class_scores)
                class_ids_per_class.append(class_class_ids)
            batch_item_boxes = K.concatenate(boxes_per_class, axis=0)
            batch_item_scores = K.concatenate(scores_per_class, axis=0)
            batch_item_scores = K.expand_dims(batch_item_scores, axis=-1)
            batch_item_class_ids = K.concatenate(class_ids_per_class, axis=0)
            batch_item_class_ids = K.expand_dims(batch_item_class_ids, axis=-1)
            # (num_keep_all_class_boxes, 6)
            batch_item_predictions = K.concatenate([batch_item_boxes,
                                                    batch_item_scores,
                                                    batch_item_class_ids], axis=-1)
            batch_item_num_predictions = tf.shape(batch_item_boxes)[0]
            batch_item_num_predictions = tf.Print(batch_item_num_predictions, [batch_item_num_predictions], '\nbatch_item_num_predictions', summarize=1000)
            batch_item_num_pad = tf.maximum(max_boxes_per_image_tensor - batch_item_num_predictions, 0)
            padded_batch_item_predictions = tf.pad(tensor=batch_item_predictions,
                                                   paddings=[
                                                       [0, batch_item_num_pad],
                                                       [0, 0]],
                                                   mode='CONSTANT',
                                                   constant_values=0.0)
            return padded_batch_item_predictions

        predictions = tf.map_fn(lambda x: evaluate_batch_item(x[0], x[1], x[2]),
                                elems=(boxes, scores, mask),
                                dtype=tf.float32)

        predictions = tf.reshape(predictions, (-1, self.max_boxes_per_image, 6))
        return predictions

    def compute_output_shape(self, input_shape):
        return None, self.max_boxes_per_image, 6
