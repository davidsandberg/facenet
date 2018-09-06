import argparse
import json
import sys
from typing import Any, List, cast

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


class FaceDetector:
    def __init__(self, model_path: str, gpu_memory_fraction: float=0.25,
                 visible_device_list: str='0') -> None:
        """
        Arguments:
            model_path: a string, path to a pb file.
            gpu_memory_fraction: a float number.
            visible_device_list: a string.
        """
        with tf.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(graph_def, name='import')

        self.input_image = graph.get_tensor_by_name('import/image_tensor:0')
        self.output_ops = [
            graph.get_tensor_by_name('import/boxes:0'),
            graph.get_tensor_by_name('import/scores:0'),
            graph.get_tensor_by_name('import/num_boxes:0'),
        ]

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_memory_fraction,
            visible_device_list=visible_device_list
        )
        config_proto = tf.ConfigProto(
            gpu_options=gpu_options,
            log_device_placement=False)
        self.sess = tf.Session(graph=graph, config=config_proto)

    def __call__(self, image: np.ndarray, ratio: float=1.0,
                 score_threshold: float=0.5) -> List[List[int]]:
        """Detect faces.

        Arguments:
            image: a numpy uint8 array with shape [height, width, 3],
                that represents a RGB image.
            score_threshold: a float number.
        Returns:
            boxes: a float numpy array of shape [num_faces, 4].
            scores: a float numpy array of shape [num_faces].

        Note that box coordinates are in the order: ymin, xmin, ymax, xmax!
        """
        h, w, _ = image.shape
        image = np.expand_dims(image, 0)

        boxes, scores, num_boxes = self.sess.run(
            self.output_ops, feed_dict={self.input_image: image}
        )
        num_boxes = num_boxes[0]
        boxes = boxes[0][:num_boxes]
        scores = scores[0][:num_boxes]

        to_keep = scores > score_threshold
        boxes = boxes[to_keep]
        scores = scores[to_keep]

        scaler = np.array([h, w, h, w], dtype='float32')
        boxes = boxes * scaler
        box_list = cast(List[List[int]], [])
        for box in boxes:
            box /= ratio
            box = box.astype(int)
            box_list.append(box.tolist())
        return box_list


def main(image: Any, threshold: float=0.5,
         desired_size: int=768) -> None:
    face_detector = FaceDetector('model.pb')
    old_size = image.size
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    image = image.resize(new_size, Image.ANTIALIAS)
    image = np.array(image)
    boxes = face_detector(image, ratio=ratio, score_threshold=threshold)
    print(json.dumps({'boxes': boxes}))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_path',
        type=str,
        help='Image path')
    parser.add_argument(
        '--threshold',
        type=float,
        help='Threshold for finding faces',
        default=0.5)
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    if args:
        image = Image.open(args.image_path)
        main(image, args.threshold)
