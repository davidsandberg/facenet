import tensorflow as tf
from facenet.src.align import detect_face
from facenet.src import facenet
from facenet.src.facedetector import model_downloader

import copy
import os
import cv2
import numpy as np
from scipy import spatial

class FaceDetector(object):
	default_three_step_threshold=[0.6,0.7,0.7]
	def __init__(self, minsize=20, factor=0.709):
		self.minsize=minsize
		self.factor=factor

	def detect_faces_in_images(self, images, threshold=default_three_step_threshold):
		with tf.Graph().as_default():
			sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
			with sess.as_default():
				pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

		tmp_image_paths = copy.copy(images)
		list_of_bounding_boxes = []
		for image in tmp_image_paths:
			img = cv2.imread(os.path.expanduser(image))[:, :, ::-1]
			bounding_box, _ = detect_face.detect_face(
				img, self.minsize, pnet, rnet, onet, threshold, self.factor)
			if len(bounding_box) < 1:
				# DO NOT APPEND EMPTY BOUNDING BOXES
				print("A face was not detected, proceed with error handling on the following image:  ", image)
				continue
			list_of_bounding_boxes.append(self.pretty_output(bounding_box))
		return list_of_bounding_boxes


	def detect_faces_in_image(self, image, threshold=default_three_step_threshold):
		with tf.Graph().as_default():
			sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
			with sess.as_default():
				pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

		tmp_image_path = copy.copy(image)

		img = cv2.imread(os.path.expanduser(tmp_image_path))[:, :, ::-1]
		bounding_boxes, _ = detect_face.detect_face(
			img, self.minsize, pnet, rnet, onet, threshold, self.factor)
		return self.pretty_output(bounding_boxes)

	def align_face(self, images, threshold=default_three_step_threshold, image_size=160, margin=11):
		with tf.Graph().as_default():
			sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
			with sess.as_default():
				pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

		tmp_image_paths = copy.copy(images)
		img_list = []
		for image in tmp_image_paths:
			img = cv2.imread(os.path.expanduser(image))[:, :, ::-1]
			img_size = np.asarray(img.shape)[0:2]
			bounding_boxes, _ = detect_face.detect_face(
				img, self.minsize, pnet, rnet, onet, threshold, self.factor)
			if len(bounding_boxes) < 1:
				print("A face was not detected, proceed with error handling on the following image:  ", image)
				continue
			det = np.squeeze(bounding_boxes[0, 0:4])
			bb = np.zeros(4, dtype=np.int32)
			bb[0] = np.maximum(det[0] - margin / 2, 0)
			bb[1] = np.maximum(det[1] - margin / 2, 0)
			bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
			bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
			cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
			aligned = cv2.resize(cropped[:, :, ::-1],
								 (image_size, image_size))[:, :, ::-1]
			prewhitened = facenet.prewhiten(aligned)
			img_list.append(prewhitened)
		images = np.stack(img_list)
		return images


	def embedding(self, images):
		# check is model exists
		model_path = self.find_model_in_folder()
		if model_path is None:
			model_path = './.facenet_model/20180402-114759'
		if not os.path.exists(model_path):
			print("Model was not found - downloading default model from https://github.com/davidsandberg/facenet")
			model_downloader.download()
			print("Model has been downloaded to " + model_path)

		with tf.Graph().as_default():
			with tf.Session() as sess:
				facenet.load_model(model_path)

				# Get input and output tensors
				images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
				embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
				phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

				# Run forward pass to calculate embeddings
				feed_dict = {images_placeholder: images,
							 phase_train_placeholder: False}
				emb = sess.run(embeddings, feed_dict=feed_dict)

		return emb

	def find_model_in_folder(self):
		for root, dirs, files in os.walk("./"):
			for file in files:
				if file.endswith(".pb"):
					return os.path.join(root, file)



	def compare(self, images, threshold=0.7):
		emb = self.embedding(images)


		sims = np.zeros((len(images), len(images)))
		for i in range(len(images)):
			for j in range(len(images)):
				sims[i][j] = (
					1 - spatial.distance.cosine(emb[i], emb[j]) > threshold)

		return sims


	def pretty_output(self, array_of_default_output):
		box_and_confidence = []
		for element in array_of_default_output:
			box_and_confidence.append({
				'box':[element[0], element[1],
					   element[2], element[3]],
				'confidence': element[4]
			})
		return box_and_confidence

