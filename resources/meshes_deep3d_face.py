from flask import send_file
import flask
import cv2
from flask_restful import Resource, reqparse
from flask_restful import inputs
import werkzeug
import algorithms.Deep3DFaceReconstruction as Deep3DFaceRecon
import json
import numpy as np
from skimage.io import imread, imsave
import os
import tensorflow as tf
from algorithms.Deep3DFaceReconstruction.preprocess_img import Preprocess
from imutils import face_utils
import models.responses as responses
from PIL import Image
from algorithms.Deep3DFaceReconstruction.reconstruct_mesh import Reconstruction
from algorithms.Deep3DFaceReconstruction.load_data import *
from scipy.io import loadmat, savemat


class MeshesDeep3DFaceGetMesh(Resource):
    def __init__(self):
        self.deep3d_face = Deep3DFaceRecon.deep3d_face
        self.dlib_face_detector = Deep3DFaceRecon.deep3d_face.dlib_detector
        self.dlib_featurepts_predictor = Deep3DFaceRecon.deep3d_face.dlib_predictor
        self.save_path = 'output'

    def get(self):
        return {
            'Message': 'Deep3D Face server is alive ...'
        }, 200

    def post(self):
        self.parser = reqparse.RequestParser()

        self.parser.add_argument(
            'image', type=werkzeug.datastructures.FileStorage, location='files', required=True)
        self.parser.add_argument(
            'is_front', type=inputs.boolean, location='form', required=True)

        args = self.parser.parse_args()

        img_file = args['image']
        is_front = args['is_front']

        with tf.Graph().as_default() as graph, tf.device('/gpu:0'):
            images = tf.placeholder(name='input_imgs', shape=[
                None, 224, 224, 3], dtype=tf.float32)

            tf.import_graph_def(self.deep3d_face.graph_def, name='resnet',
                                input_map={'input_imgs:0': images})

            # output coefficients of R-Net (dim = 257)
            coeff = graph.get_tensor_by_name('resnet/coeff:0')

            with tf.Session() as sess:
                print('reconstructing...')

                # load image and detect 5 facial landmarks
                image_file_name = img_file.filename
                ret_json_obj = responses.MeshesDeep3DFaceResponse(
                    file_name=image_file_name)

                filestr = img_file.read()
                npimg = np.fromstring(filestr, np.uint8)
                input_img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
                rects = self.dlib_face_detector(input_img, 1)

                if len(rects) == 0:
                    ret_json_obj.json['Message'] = 'Please upload a picture with one clear face.'
                    return ret_json_obj.json, 200

                rect = rects[0]
                shape = self.dlib_featurepts_predictor(input_img, rect)
                shape = face_utils.shape_to_np(shape)

                nose_tip = np.reshape(shape[30], (1, 2))
                left_mouth = np.reshape(shape[48], (1, 2))
                right_mouth = np.reshape(shape[54], (1, 2))
                left_eye = np.reshape(
                    ((shape[37] + shape[38] + shape[41] + shape[40]) / 4), (1, 2))
                right_eye = np.reshape(
                    ((shape[43] + shape[44] + shape[46] + shape[47]) / 4), (1, 2))

                lm = np.concatenate(
                    (left_eye, right_eye, nose_tip, left_mouth, right_mouth), axis=0)

                img = Image.open(img_file)

                # preprocess input image
                input_img, lm_new, transform_params = Preprocess(
                    img, lm, self.deep3d_face.lm3D)

                coef = sess.run(coeff, feed_dict={images: input_img})

                # reconstruct 3D face with output coefficients and face model
                face_shape, face_texture, face_color, tri, face_projection, z_buffer, landmarks_2d = Reconstruction(
                    coef, self.deep3d_face.facemodel)

                # reshape outputs
                input_img = np.squeeze(input_img)
                shape = np.squeeze(face_shape, (0))
                color = np.squeeze(face_color, (0))
                landmarks_2d = np.squeeze(landmarks_2d, (0))

                # save output files
                # cropped image, which is the direct input to our R-Net
                # 257 dim output coefficients by R-Net
                # 68 face landmarks of cropped image
                # mat_file_name = ''
                # obj_file_name = ''
                # if image_file_name.endswith('.png'):
                #     mat_file_name = image_file_name.split(
                #         '\\')[-1].replace('.png', '.mat')
                #     obj_file_name = image_file_name.split(
                #         '\\')[-1].replace('.png', '_mesh.obj')

                # if image_file_name.endswith('.jpg'):
                #     mat_file_name = image_file_name.split(
                #         '\\')[-1].replace('.jpg', '.mat')
                #     obj_file_name = image_file_name.split(
                #         '\\')[-1].replace('.jpg', '_mesh.obj')

                # savemat(os.path.join(self.save_path, mat_file_name), {
                #     'cropped_img': input_img[:, :, ::-1], 'coeff': coef, 'landmarks_2d': landmarks_2d, 'lm_5p': lm_new})

                # 3D reconstruction face (in canonical view)
                # save_obj(os.path.join(self.save_path, obj_file_name),
                #          shape, tri, np.clip(color, 0, 255)/255)

        # output
        [h, w, c] = input_img.shape
        if c > 3:
            input_img = input_img[:, :, :3]

        # flip ys
        # save_vertices[:, 1] = h - 1 - save_vertices[:, 1]
        # canonical_vertices[:, 1] = h - 1 - canonical_vertices[:, 1]
        # save_kpt[:, 1] = h - 1 - save_kpt[:, 1]
        # # pos[:, :, 1] = h - 1 - pos[:, :, 1]

        # zero based indexing
        tri = tri - 1

        ret_json_obj = responses.MeshesDeep3DFaceResponse(
            file_name=image_file_name,
            message='OK',
            vertex_count=len(shape),
            width_norm=w,
            height_norm=h,
            mesh_vertex_xs=shape[:, 0].tolist(),
            mesh_vertex_ys=shape[:, 1].tolist(),
            mesh_vertex_zs=shape[:, 2].tolist(),
            triangles=tri.flatten().tolist(),
            feature_points_count=len(landmarks_2d),
            feature_point_xs=landmarks_2d[:, 0].tolist(),
            feature_point_ys=landmarks_2d[:, 1].tolist(),
        )

        return ret_json_obj.json, 200

    def load_img(img_path, lm_path):
        image = Image.open(img_path)
        lm = np.loadtxt(lm_path)
        return image, lm
