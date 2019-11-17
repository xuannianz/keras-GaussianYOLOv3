import keras.backend as K
import tensorflow as tf
import numpy as np

from util_graphs import box_iou_graph, y_pred_graph


def nll_loss(x, mu, sigma, sigma_const=0.3):
    pi = tf.constant(np.pi)
    Z = (2 * pi * (sigma + sigma_const) ** 2) ** 0.5
    probability_density = tf.exp(-0.5 * (x - mu) ** 2 / ((sigma + sigma_const) ** 2)) / Z
    nll = -tf.log(probability_density + 1e-7)
    return nll


def yolo_loss(args, anchors, num_anchors_per_layer, num_classes, ignore_thresh=.5, print_loss=True):
    """
    Return yolo_loss tensor

    Args:
        args (list): args[:num_output_layers] the output of yolo_body or tiny_yolo_body
            args[num_output_layers:] raw_y_true
        anchors (np.array): shape=(N, 2), wh
        num_anchors_per_layer (int):
        num_classes (int):
        ignore_thresh (float): the iou threshold whether to ignore object confidence loss
        print_loss:

    Returns:
        loss: tensor, shape=(1,)

    """
    num_output_layers = len(anchors) // num_anchors_per_layer
    yolo_outputs = args[:num_output_layers]
    raw_y_trues = args[num_output_layers:]
    anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(raw_y_trues[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(raw_y_trues[0])) for l in range(num_output_layers)]
    loss = 0
    batch_size = K.shape(yolo_outputs[0])[0]
    batch_size_f = K.cast(batch_size, K.dtype(yolo_outputs[0]))

    for l in range(num_output_layers):
        grid_shape = grid_shapes[l]
        yolo_output = yolo_outputs[l]
        raw_y_pred = K.reshape(yolo_output, [-1, grid_shape[0], grid_shape[1], num_anchors_per_layer, num_classes + 9])
        raw_y_true = raw_y_trues[l]
        anchor_mask = anchor_masks[l]
        # (batch_size, grid_height, grid_width, num_anchors_this_layer, 1)
        object_mask = raw_y_true[..., 4:5]
        # (batch_size, grid_height, grid_width, num_anchors_this_layer, num_classes)
        y_true_class_probs = raw_y_true[..., 5:]
        grid, y_pred_box, y_pred_delta_xy, y_pred_log_wh, y_pred_sigma, y_pred_confidence, y_pred_class_probs = \
            y_pred_graph(raw_y_pred, anchors[anchor_mask], input_shape)
        y_true_delta_xy = raw_y_true[..., :2] * grid_shapes[l][::-1] - grid
        y_true_log_wh = K.log(raw_y_true[..., 2:4] * input_shape[::-1] / anchors[anchor_mask])
        y_true_log_wh = K.switch(object_mask, y_true_log_wh, K.zeros_like(y_true_log_wh))
        box_loss_scale = 2 - raw_y_true[..., 2:3] * raw_y_true[..., 3:4]
        ignore_mask = tf.TensorArray(K.dtype(raw_y_trues[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask_):
            # (num_gt_boxes, 4)
            gt_box = tf.boolean_mask(raw_y_true[b, ..., 0:4], object_mask_bool[b, ..., 0])
            # (grid_height, grid_width, num_anchors_this_layer, num_gt_boxes)
            iou = box_iou_graph(y_pred_box[b], gt_box)
            # (grid_height, grid_width, num_anchors_this_layer)
            best_iou = K.max(iou, axis=-1)
            ignore_mask_ = ignore_mask_.write(b, K.cast(best_iou < ignore_thresh, K.dtype(gt_box)))
            return b + 1, ignore_mask_

        _, ignore_mask = tf.while_loop(lambda b, *largs: b < batch_size, loop_body, [0, ignore_mask])
        # (batch_size, grid_height, grid_width, num_anchors_this_layer)
        ignore_mask = ignore_mask.stack()
        # (batch_size, grid_height, grid_width, num_anchors_this_layer, 1)
        ignore_mask = K.expand_dims(ignore_mask, -1)

        y_true = tf.concat([y_true_delta_xy, y_true_log_wh], axis=-1)
        y_pred_mu = tf.concat([y_pred_delta_xy, y_pred_log_wh], axis=-1)
        x_loss = nll_loss(y_true[..., 0:1], y_pred_mu[..., 0:1], y_pred_sigma[..., 0:1])
        x_loss = object_mask * box_loss_scale * x_loss
        y_loss = nll_loss(y_true[..., 1:2], y_pred_mu[..., 1:2], y_pred_sigma[..., 1:2])
        y_loss = object_mask * box_loss_scale * y_loss
        w_loss = nll_loss(y_true[..., 2:3], y_pred_mu[..., 2:3], y_pred_sigma[..., 2:3])
        w_loss = object_mask * box_loss_scale * w_loss
        h_loss = nll_loss(y_true[..., 3:4], y_pred_mu[..., 3:4], y_pred_sigma[..., 3:4])
        h_loss = object_mask * box_loss_scale * h_loss
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, y_pred_confidence) + \
                          (1 - object_mask) * K.binary_crossentropy(object_mask, y_pred_confidence) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(y_true_class_probs, y_pred_class_probs)
        x_loss = K.sum(x_loss) / batch_size_f
        y_loss = K.sum(y_loss) / batch_size_f
        w_loss = K.sum(w_loss) / batch_size_f
        h_loss = K.sum(h_loss) / batch_size_f
        confidence_loss = K.sum(confidence_loss) / batch_size_f
        class_loss = K.sum(class_loss) / batch_size_f
        loss += x_loss + y_loss + w_loss + h_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, x_loss, y_loss, w_loss, h_loss, confidence_loss, class_loss, K.sum(ignore_mask)],
                            message='\nloss: ')
    return loss
