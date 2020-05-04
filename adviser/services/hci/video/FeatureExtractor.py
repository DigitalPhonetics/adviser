###############################################################################
#
# Copyright 2020, University of Stuttgart: Institute for Natural Language Processing (IMS)
#
# This file is part of Adviser.
# Adviser is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3.
#
# Adviser is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Adviser.  If not, see <https://www.gnu.org/licenses/>.
#
###############################################################################

"""Feature extraction with openSMILE"""

import cv2
import dlib
import numpy as np
import os

from imutils import face_utils

from utils.domain.domain import Domain
from utils.domain.jsonlookupdomain import JSONLookupDomain
from services.service import PublishSubscribe
from services.service import Service


class VideoFeatureExtractor(Service):
    """TODO"""

    def __init__(self, domain: Domain = ""):
        Service.__init__(self, domain=domain)
        self.module_dir = os.path.dirname(os.path.abspath(__file__))
        # # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        self.CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # for detecting faces (returns coordinates of rectangle(s) of face area(s))
        self.DETECTOR = dlib.get_frontal_face_detector()
        # facial landmark predictor
        predictor_file = os.path.abspath(os.path.join(self.module_dir, '..', '..', '..', 'resources', 'models', 'video', 'shape_predictor_68_face_landmarks.dat'))
        self.PREDICTOR = dlib.shape_predictor(predictor_file)

    @PublishSubscribe(queued_sub_topics=["video_input"], sub_topics=["user_acts"],
                      pub_topics=["fl_features"])
    def extract_fl_features(self, video_input, user_acts):
        """TODO

        Returns:
            dict: TODO
        """
        def _distance(a, b):
            return np.linalg.norm(a-b)
        print(f'VIDEO FEATURE ENTER, len(video_input): {len(video_input)}')
        features = []
        aggregated_feats = None
        for frame in video_input[::2]:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = self.CLAHE.apply(frame)
            faces = self.DETECTOR(frame, 1)
            if len(faces) > 0:  # at least one face detected
                landmarks = self.PREDICTOR(frame, faces[0])
                landmarks = face_utils.shape_to_np(landmarks)
                norm_left_eye = _distance(landmarks[21], landmarks[39])
                norm_right_eye = _distance(landmarks[22], landmarks[42])
                norm_lips = _distance(landmarks[33], landmarks[52])
                eyebrow_left = sum(
                    [(_distance(landmarks[39], landmarks[i]) / norm_left_eye)
                        for i in [18, 19, 20, 21]]
                )
                eyebrow_right = sum(
                    [(_distance(landmarks[42], landmarks[i]) / norm_right_eye)
                        for i in [22, 23, 24, 25]]
                )
                lip_left = sum(
                    [(_distance(landmarks[33], landmarks[i]) / norm_lips)
                        for i in [48, 49, 50]]
                )
                lip_right = sum(
                    [(_distance(landmarks[33], landmarks[i]) / norm_lips)
                        for i in [52, 53, 54]]
                )
                mouth_width = _distance(landmarks[48], landmarks[54])
                mouth_height = _distance(landmarks[51], landmarks[57])
                features.append(np.array([
                    eyebrow_left,
                    eyebrow_right,
                    lip_left,
                    lip_right,
                    mouth_width,
                    mouth_height
                ]))

        # aggregate features across frames
        if len(features) > 0:
            mean = np.mean(features, axis=0)
            mini = np.amin(features, axis=0)
            maxi = np.amax(features, axis=0)
            std = np.std(features, axis=0)
            perc25 = np.percentile(features, q=25, axis=0)
            perc75 = np.percentile(features, q=75, axis=0)

            aggregated_feats = np.array([mean, mini, maxi, std, perc25, perc75]).reshape(1, 36)

        print("VIDEO FEAT PUB")
        return {'fl_features': aggregated_feats}


if __name__ == "__main__":
    domain = JSONLookupDomain('ImsLecturers')

    feat_extractor = VideoFeatureExtractor(domain=domain)
