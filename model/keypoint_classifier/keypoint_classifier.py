#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier.tflite',
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        landmark_list,
    ):
        # Ensure the input is reshaped to the correct size for the model (1, 42, 1)
        input_details_tensor_index = self.input_details[0]['index']

        # Reshape landmark_list to (1, 42, 1) if your model expects a 3D shape
        reshaped_landmarks = np.reshape(landmark_list, (1, 42, 1)).astype(np.float32)

        self.interpreter.set_tensor(
            input_details_tensor_index,
            reshaped_landmarks
        )

        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']
        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        return result_index

