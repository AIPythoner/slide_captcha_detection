import time
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import matplotlib

import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

matplotlib.use('TkAgg')

PATH_TO_CKPT = '../bank_pisition/pb/frozen_inference_graph.pb'
PATH_TO_LABELS = r'./labelmap.pbtxt'

NUM_CLASSES = 1

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    image = image.convert('RGB')
    im_width, im_height = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


PATH_TO_TEST_IMAGES_DIR = r'E:\deepLearingData\character_position\image'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, i) for i in os.listdir(PATH_TO_TEST_IMAGES_DIR)]

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor_ = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes_ = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores_ = detection_graph.get_tensor_by_name('detection_scores:0')
        classes_ = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections_ = detection_graph.get_tensor_by_name('num_detections:0')
        for image_path in TEST_IMAGE_PATHS:
            image = Image.open(image_path)
            im_width, im_height = image.size
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)

            # Actual detection.
            s = time.time()
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes_, scores_, classes_, num_detections_],
                feed_dict={image_tensor_: image_np_expanded})
            for i in range(len(classes)):
                y_min = int(boxes[0][i][0] * im_height)
                x_min = int(boxes[0][i][1] * im_width)
                y_max = int(boxes[0][i][2] * im_height)
                x_max = int(boxes[0][i][3] * im_width)
            print(x_min, y_min, x_max, y_max, '本次预测耗时：%s ms' % round((time.time() - s) * 1000))

            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            # Visualization of the results of a detection.
            plt.figure(figsize=(12, 8))
            plt.imshow(image_np)
            plt.show()
