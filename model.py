from keras.layers import Add, BatchNormalization, Concatenate, Conv2D, Input
from keras.layers import Lambda, LeakyReLU, UpSampling2D, ZeroPadding2D
from keras.regularizers import l2
from keras.models import Model
from functools import reduce

from loss import yolo_loss
from layers import DetectionLayer


def compose(*funcs):
    """
    Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def darknet_conv2d(*args, **kwargs):
    """
    Wrapper to set Darknet parameters for Convolution2D.
    """
    darknet_conv_kwargs = dict({'kernel_regularizer': l2(5e-4)})
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def darknet_conv2d_bn_leaky(*args, **kwargs):
    """
    Darknet Convolution2D followed by BatchNormalization and LeakyReLU.
    """
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        darknet_conv2d(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters, num_blocks):
    """
    A series of resblocks starting with a downsampling Convolution2D
    """
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = darknet_conv2d_bn_leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
            darknet_conv2d_bn_leaky(num_filters // 2, (1, 1)),
            darknet_conv2d_bn_leaky(num_filters, (3, 3)))(x)
        x = Add()([x, y])
    return x


def darknet_body(x):
    """
    Darknet body having 52 Convolution2D layers
    1 + (1 + 1 * 2) + (1 + 2 * 2) + (1 + 8 * 2) + (1 + 8 * 2) + (1 + 4 * 2) = 1 + 3 + 5 + 17 + 17 + 9 = 52
    """
    # (416, 416)
    x = darknet_conv2d_bn_leaky(32, (3, 3))(x)
    # (208, 208)
    x = resblock_body(x, 64, 1)
    # (104, 104)
    x = resblock_body(x, 128, 2)
    # (52, 52)
    x = resblock_body(x, 256, 8)
    # (26, 26)
    x = resblock_body(x, 512, 8)
    # (13, 13)
    x = resblock_body(x, 1024, 4)
    return x


def make_last_layers(x, num_filters, out_filters):
    """
    6 conv2d_bn_leaky layers followed by a conv2d layer
    """
    x = compose(darknet_conv2d_bn_leaky(num_filters, (1, 1)),
                darknet_conv2d_bn_leaky(num_filters * 2, (3, 3)),
                darknet_conv2d_bn_leaky(num_filters, (1, 1)),
                darknet_conv2d_bn_leaky(num_filters * 2, (3, 3)),
                darknet_conv2d_bn_leaky(num_filters, (1, 1)))(x)
    y = compose(darknet_conv2d_bn_leaky(num_filters * 2, (3, 3)),
                darknet_conv2d(out_filters, (1, 1)))(x)
    return x, y


def yolo_body(anchors, num_classes=20, score_threshold=0.01):
    """
    Create YOLO_V3 model CNN body in Keras.

    Args:
        anchors:
        num_classes:
        score_threshold:

    Returns:

    """
    num_anchors = len(anchors)
    num_anchors_per_layer = num_anchors // 3
    image_input = Input(shape=(None, None, 3), name='image_input')
    fm_13_input = Input(shape=(None, None, num_anchors_per_layer, num_classes + 5), name='fm_13_input')
    fm_26_input = Input(shape=(None, None, num_anchors_per_layer, num_classes + 5), name='fm_26_input')
    fm_52_input = Input(shape=(None, None, num_anchors_per_layer, num_classes + 5), name='fm_52_input')
    image_shape_input = Input(shape=(2,), name='image_shape_input')
    darknet = Model([image_input], darknet_body(image_input))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors_per_layer * (num_classes + 9))
    x = compose(darknet_conv2d_bn_leaky(256, (1, 1)), UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors_per_layer * (num_classes + 9))
    x = compose(darknet_conv2d_bn_leaky(128, (1, 1)), UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors_per_layer * (num_classes + 9))

    loss = Lambda(yolo_loss,
                  output_shape=(1,),
                  name='yolo_loss',
                  arguments={'anchors': anchors,
                             'num_anchors_per_layer': num_anchors_per_layer,
                             'num_classes': num_classes,
                             'ignore_thresh': 0.5})(
        [y1, y2, y3, fm_13_input, fm_26_input, fm_52_input])
    training_model = Model([image_input, fm_13_input, fm_26_input, fm_52_input], loss, name='yolo')
    detections = DetectionLayer(anchors, num_classes=num_classes, score_threshold=score_threshold, name='yolo_detection')(
        [y1, y2, y3, image_shape_input])
    prediction_model = Model([image_input, image_shape_input], detections, name='yolo')
    return training_model, prediction_model
