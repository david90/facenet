# coding=utf-8
"""Face Detection and Recognition"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# This is the work of David Sandberg and shanren7 remodelled into a
# high level container. It's an attempt to simplify the use of such
# technology and provide an easy to use facial recognition package.
#
# https://github.com/davidsandberg/facenet
# https://github.com/shanren7/real_time_face_recognition
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pickle
import os

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from scipy import misc

import align.detect_face
import facenet


gpu_memory_fraction = 0.3
facenet_model_checkpoint = os.path.dirname(__file__) + "/../model_checkpoints/20170512-110547"
classifier_model = os.path.dirname(__file__) + "/../model_checkpoints/my_classifier_1.pkl"
debug = False

def import_images(images_dir_str):
    face_images = {}
    for _, directories, _ in os.walk(images_dir_str):
        for directory in directories:
            print(directory)
            face_images[directory] = []
            for _, _, image_names in os.walk("{}/{}".format(images_dir_str, directory)):
                for image_name in image_names:
                    if image_name == '.DS_Store':
                        continue
                    print("{}/{}/{}".format(images_dir_str, directory, image_name))
                    face_images[directory].append(
                        cv2.imread(
                            "{}/{}/{}".format(images_dir_str, directory, image_name)
                        )
                    )
    return face_images

images_dir_str = "../data/images"
face_images = import_images(images_dir_str)

class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None


class Recognition:
    def __init__(self):
        self.detect = Detection()
        self.encoder = Encoder()
        self.identifier = Identifier()

        sample_faces = []
        for name, images in face_images.items():
            for image in images:
                faces = self.detect.find_faces(image)
                if len(faces) == 1:
                    face = faces[0]
                    face.embedding = self.encoder.generate_embedding(face)
                    face.name = name
                    sample_faces.append(face)
        self.identifier.add_identities(sample_faces)

    def add_identity(self, image, person_name):
        faces = self.detect.find_faces(image)

        if len(faces) == 1:
            face = faces[0]
            face.name = person_name
            face.embedding = self.encoder.generate_embedding(face)
            self.identifier.add_identities([face])
            return faces

    def identify(self, image):
        faces = self.detect.find_faces(image)

        for i, face in enumerate(faces):
            if debug:
                cv2.imshow("Face: " + str(i), face.image)
            face.embedding = self.encoder.generate_embedding(face)
            face.name = self.identifier.identify(face)

        return faces


class Identifier:
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=1, n_jobs=4)
        self.threshold = 0.7
        self.data = {}
        self.data_classes = []

    def identify(self, face):
        if face.embedding is not None:
            nearestest_neighbour = self.model.kneighbors([face.embedding])
            distance = nearestest_neighbour[0][0][0]
            print('nearestest_neighbour: {}'.format(nearestest_neighbour))
            if distance >= self.threshold:
                return 'Unknown'

            return self.data_classes[nearestest_neighbour[1][0][0]]

    def add_identities(self, faces):
        for face in faces:
            if face.name not in self.data:
                self.data[face.name] = []
            self.data[face.name].append(face.embedding)
        data = []
        classes = []
        for name, deep_features in self.data.items():
            for deep_feature in deep_features:
                data.append(deep_feature)
                classes.append(name)
        self.data_classes = classes
        self.model.fit(data, classes)
        return face


class Encoder:
    def __init__(self):
        self.sess = tf.Session()
        with self.sess.as_default():
            facenet.load_model(facenet_model_checkpoint)

    def generate_embedding(self, face):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        prewhiten_face = facenet.prewhiten(face.image)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]


class Detection:
    # face detection parameters
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_crop_size=160, face_crop_margin=32):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return align.detect_face.create_mtcnn(sess, None)

    def find_faces(self, image):
        faces = []

        bounding_boxes, _ = align.detect_face.detect_face(image, self.minsize,
                                                          self.pnet, self.rnet, self.onet,
                                                          self.threshold, self.factor)
        for bb in bounding_boxes:
            face = Face()
            face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)

            img_size = np.asarray(image.shape)[0:2]
            face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
            face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
            cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
            face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')

            faces.append(face)

        return faces
