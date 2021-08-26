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
"""Face detection (RetinaFace) and recognition (ArcFace)
https://github.com/tae898/face-detection-recognition

"""
from services.service import Service, PublishSubscribe
import python_on_whales
from utils.domain.jsonlookupdomain import JSONLookupDomain
from utils.logger import DiasysLogger
import time
import io
import cv2
from PIL import Image, ImageDraw, ImageFont
import jsonpickle
import requests
import numpy as np
import pickle
from glob import glob
from utils.topics import Topic


def unpickle(path: str):
    """Unpickle the pickled file, and return it."""
    with open(path, 'rb') as stream:
        foo = pickle.load(stream)
    return foo


def load_embeddings():
    """Load pre-defined face embeddings."""
    embeddings_predefined = {path.split(
        '/')[-1].split('.pkl')[0]: unpickle(path) for path in glob('./embeddings/*.pkl')}
    return embeddings_predefined


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute the cosine similarity of the two vectors.

    The returned value is between 1 and -1.
    """
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))


def start_docker_container(image: str, port_id: int, logger: DiasysLogger,
                           use_cuda=False, sleep_time=5) -> python_on_whales.Container:
    """Start docker container given the image name and port number.

    A docker container object is returned.
    """
    kwargs = {'image': image, 'detach': True, 'publish': [(port_id, port_id)]}
    if use_cuda:
        kwargs['gpus'] = 'all'

    container = python_on_whales.docker.run(**kwargs)

    logger.info(f"starting a {image} container ...")
    logger.debug(f"warming up the container for {sleep_time} seconds ...")
    time.sleep(sleep_time)

    return container


def kill_container(container: python_on_whales.Container, logger: DiasysLogger):
    """Kill docker container."""
    container.kill()
    logger.info(f"container killed.")
    logger.info(f"DONE!")


class FaceDetectionRecognition(Service):
    """Face detection and recognition module.

    Face detection: RetinaFace gives you a bounding box and 5 facial landmarks per face.
    Face recognition: ArcFAce gives you 512-dimensional face embedding vector per face.

    The docker image (tae898/face-detection-recognition) used here has both of
    them included. It works as a flask API server.

    Check out https://github.com/tae898/face-detection-recognition for details.
    """

    def __init__(self, domain: JSONLookupDomain = None, identifier: str = None,
                 logger: DiasysLogger = DiasysLogger(), use_cuda: bool = False,
                 port_number: int = 10002):
        Service.__init__(self, domain, identifier=identifier)
        self.logger = logger
        if use_cuda:
            docker_image = 'tae898/face-detection-recognition-cuda'
        else:
            docker_image = 'tae898/face-detection-recognition'
        self.container = start_docker_container(
            docker_image, port_number, logger, use_cuda)

        self.url = f"http://127.0.0.1:{port_number}/"

    @PublishSubscribe(sub_topics=['video_input'], pub_topics=['face_detection_recognition', 'img_rgb_pillow'])
    def run_face_detection_recognition(self, video_input) -> dict:
        """Run face detection and recogntion.

        Published topics are face detection/recognition results and 
        pillow rgb image. The pillow image is published so that we can be 
        certain which image was used for face detection/recognition.
        """
        img_rgb_pillow = Image.fromarray(video_input)

        face_detection_recognition = self.request_inference(img_rgb_pillow)

        embeddings = [fdr['normed_embedding']
                      for fdr in face_detection_recognition]

        faces_detected = self.match_face(embeddings)

        for idx, (fdr, name) in enumerate(zip(face_detection_recognition, faces_detected)):
            face_detection_recognition[idx]['name'] = name

        return {'face_detection_recognition': face_detection_recognition,
                'img_rgb_pillow': img_rgb_pillow}

    def request_inference(self, img_rgb_pillow: Image) -> list:
        """Make a RESTful HTTP request to the face API server.

        The PIL rgb image is encoded with jsonpickle. I know this is not 
        conventional, but encoding and decoding is so easy with jsonpickle somehow.

        Returns a list of face detection/recognition results (bboxes, det_scores, 
        landmarks, and embeddings)
        """
        self.logger.debug(f"loading image ...")
        binary_image = io.BytesIO()
        img_rgb_pillow.save(binary_image, format='JPEG')
        binary_image = binary_image.getvalue()

        data = {'image': binary_image}
        self.logger.info(f"image loaded!")

        self.logger.debug(f"sending image to fdr server...")
        data = jsonpickle.encode(data)
        response = requests.post(self.url, json=data)
        self.logger.info(f"got {response} from fdr server!...")
        response = jsonpickle.decode(response.text)

        face_detection_recognition = response['face_detection_recognition']
        self.logger.info(f"{len(face_detection_recognition)} faces deteced!")

        return face_detection_recognition

    def match_face(self, embeddings: list) -> list:
        """Perform face recognition based on the cosine similarity.

        The pre-defined faces (embeddings) are saved at `./embeddings`.

        The cosine similarity threshold is fixed to 0.65. Feel free to play around 
        with this number.

        Returns a list of faces matched. If there is no match, then the face is
        named as "Stranger"
        """
        COSINE_SIMILARITY_THRESHOLD = 0.65

        embeddings_predefined = load_embeddings()
        possible_names = list(embeddings_predefined.keys())

        cosine_similarities = []
        for embedding in embeddings:
            cosine_similarities_ = {name: cosine_similarity(embedding, embedding_pre)
                                    for name, embedding_pre in embeddings_predefined.items()}
            cosine_similarities.append(cosine_similarities_)

        self.logger.debug(f"cosine similarities: {cosine_similarities}")

        faces_detected = [max(sim, key=sim.get) for sim in cosine_similarities]
        faces_detected = [name.replace('-', ' ') if sim[name] > COSINE_SIMILARITY_THRESHOLD else "Stranger"
                          for name, sim in zip(faces_detected, cosine_similarities)]

        self.logger.info(f"faces_detected: {faces_detected}")

        return faces_detected

    def dialog_end(self):
        kill_container(self.container, self.logger)


class AgeGender(Service):
    """Age and gender estimation module.

    Docker image tae898/age-gender works as a flask API server.

    Check out https://github.com/tae898/age-gender for details.
    """

    def __init__(self, domain: JSONLookupDomain = None, identifier: str = None,
                 logger: DiasysLogger = DiasysLogger(), port_number: int = 10003):
        Service.__init__(self, domain, identifier=identifier)
        self.logger = logger
        docker_image = 'tae898/age-gender'
        self.container = start_docker_container(
            docker_image, port_number, logger, use_cuda=False)

        self.url = f"http://127.0.0.1:{port_number}/"

    @PublishSubscribe(sub_topics=['face_detection_recognition', 'img_rgb_pillow'], pub_topics=['ages', 'genders'])
    def run_age_gender(self, face_detection_recognition, img_rgb_pillow) -> dict:
        """Run age and gender estimation.

        The age-gender estimation model uses ArcFace face embedding vectors 
        as input.

        Published topics are ages and genders
        """
        bboxes = [fdr['bbox'] for fdr in face_detection_recognition]
        embeddings = [fdr['normed_embedding']
                      for fdr in face_detection_recognition]

        ages, genders = self.request_inference(embeddings)

        self.logger.info(f"ages found: {ages}")
        self.logger.info(f"genders found: {ages}")

        return {'ages': ages, 'genders': genders}

    def request_inference(self, embeddings):
        self.logger.debug(f"sending embeddings to age-gender server ...")

        # -1 accounts for the batch size.
        data = np.array(embeddings).reshape(-1, 512).astype(np.float32)

        data = pickle.dumps(data)

        data = {'embeddings': data}
        data = jsonpickle.encode(data)
        response = requests.post(self.url, json=data)
        self.logger.info(f"got {response} from age-gender server!...")

        response = jsonpickle.decode(response.text)
        ages = response['ages']
        genders = response['genders']

        return ages, genders

    def dialog_end(self):
        kill_container(self.container, self.logger)


class AnnotateDisplayImage(Service):
    """Annotate image and display using opencv."""

    def __init__(self, domain: JSONLookupDomain = None, identifier: str = None,
                 logger: DiasysLogger = DiasysLogger()):
        Service.__init__(self, domain, identifier=identifier)

    @PublishSubscribe(sub_topics=['face_detection_recognition', 'img_rgb_pillow', 'ages', 'genders'], pub_topics=[Topic.DIALOG_END])
    def annotate_and_display(self, face_detection_recognition, img_rgb_pillow, ages, genders):
        """Annotate and display.
        
        To annotate an image, we need face detection/recognition results, PIL 
        image, age, and gender.
        """
        bboxes = [fdr['bbox'] for fdr in face_detection_recognition]
        names = [fdr['name'] for fdr in face_detection_recognition]

        img_rgb_pillow = self.annotate_image(
            img_rgb_pillow, genders, ages, bboxes, names)
        imb_rgb_array = np.array(img_rgb_pillow)
        img_bgr_array = cv2.cvtColor(imb_rgb_array, cv2.COLOR_RGB2BGR)

        cv2.imshow('annotated', img_bgr_array)
        if cv2.waitKey(1) == ord('q'):
            return {Topic.DIALOG_END: True}

    def annotate_image(self, img_rgb_pillow, genders, ages, bboxes, names) -> Image:
        MAXIMUM_ENTROPY = {'gender': 0.6931471805599453,
                           'age': 4.615120516841261}

        draw = ImageDraw.Draw(img_rgb_pillow)
        font = ImageFont.truetype("fonts/arial.ttf", 25)

        for gender, age, bbox, name in zip(genders, ages, bboxes, names):
            gender = 'male' if gender['m'] > 0.5 else 'female'
            draw.rectangle(bbox.tolist(), outline=(0, 0, 0))

            draw.text(
                (bbox[0], bbox[1]), f"{name}, {round(age['mean'])} years old, {gender}", fill=(255, 0, 0), font=font)

        return img_rgb_pillow

    def dialog_end(self):
        cv2.destroyAllWindows()