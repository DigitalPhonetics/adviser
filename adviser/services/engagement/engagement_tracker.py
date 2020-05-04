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
import numpy as np
import cv2
from typing import List, Tuple
from threading import Thread
import datetime
import time
import copy 
import sys
import os
from statistics import mean
from math import sqrt
import subprocess
import zmq
from zmq import Context
import json

from utils.userstate import EngagementType
from services.service import Service, PublishSubscribe


def get_root_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class EngagementTracker(Service):
    """
    Start feature extraction with OpenFace.
    Requires OpenFace to be installed - instructions can be found in tool/openface.txt
    """
    def __init__(self, domain="", camera_id: int = 0, openface_port: int = 6004, delay: int = 2, identifier=None):
        """
        Args:
            camera_id: index of the camera you want to use (if you only have one camera: 0)
        """
        Service.__init__(self, domain="", identifier=identifier)
        self.camera_id = camera_id
        self.openface_port = openface_port
        self.openface_running = False
        self.threshold = delay   # provide number of seconds as parameter, one second = 15 frames

        ctx = Context.instance()
        self.openface_endpoint = ctx.socket(zmq.PAIR)
        self.openface_endpoint.bind(f"tcp://127.0.0.1:{self.openface_port}")

        startExtraction = f"{os.path.join(get_root_dir(), 'tools/OpenFace/build/bin/FaceLandmarkVidZMQ')} -device {self.camera_id} -port 6004"    # todo config open face port
        self.p_openface = subprocess.Popen(startExtraction.split(), stdout=subprocess.PIPE)	# start OpenFace
        self.extracting = False
        self.extractor_thread = None


    def dialog_start(self):
        # Set openface to publishing mode and wait until it is ready
        self.openface_endpoint.send(bytes(f"OPENFACE_START", encoding="ascii"))
        self.extracting = False
        while not self.extracting:
            msg = self.openface_endpoint.recv()    # receive started signal
            msg = msg.decode("utf-8")
            if msg == "OPENFACE_STARTED":
                print("START EXTRACTION")
                self.extracting = True
                self.extractor_thread = Thread(target=self.publish_gaze_directions)
                self.extractor_thread.start()
    

    @PublishSubscribe(pub_topics=["engagement", "gaze_direction"])
    def yield_gaze_direction(self, engagement: EngagementType, gaze_direction: Tuple[float, float]):
        """
        This is a helper function for the continuous publishing of engagement features.
        Call this function from a continuously running loop.

        Returns:
            engagement (EngagementType): high / low
            gaze_direction (float, float): tuple of gaze-x-angle and gaze-y-angle
        """
        return {"engagement": engagement, "gaze_direction": gaze_direction}


    def publish_gaze_directions(self):
        """
        Meant to be used in a thread.
        Runs an inifinte loop polling features from OpenFace library, parsing them and extracting engagement features.
        Calls `yield_gaze_direction` to publish the polled and processed engagement features.
        """
        
        x_coordinates=[]
        y_coordinates=[]
       						
        norm = 0.0			# center point of screen; should be close(r) to 0
        looking = True

        while self.extracting:
            req = self.openface_endpoint.send(bytes(f"OPENFACE_PULL", encoding="ascii"))
            msg =  self.openface_endpoint.recv()
            try:
                msg = msg.decode("utf-8")
                if msg == "OPENFACE_ENDED":
                    self.extracting = False
                msg_data = json.loads(msg)

                gaze_x = msg_data["gaze"]["angle"]["x"]
                gaze_y = msg_data["gaze"]["angle"]["y"]
                    
                gaze_x = sqrt(gaze_x**2)				# gaze_angle_x (left-right movement), square + root is done to yield only positive values
                gaze_y = sqrt(gaze_y**2)				# gaze_angle_y (up-down movement) 
                x_coordinates.append(gaze_x)
                y_coordinates.append(gaze_y)
                current = (len(x_coordinates))-1
                if current > self.threshold:
                    previous_x = mean(x_coordinates[current-(self.threshold+1):current])		# obtain the average of previous frames
                    previous_y = mean(y_coordinates[current-(self.threshold+1):current])
                    difference_x = sqrt((norm - previous_x)**2)					# compare current frame to average of previous frames
                    difference_y = sqrt((norm - previous_y)**2)
                    # print(difference_x, difference_y)
                    if difference_x < 0.15 and difference_y < 0.15:				# check whether difference between current and previous frames exceeds certain threshold (regulates tolerance/strictness)
                        if looking != True:
                            looking = True
                            self.yield_gaze_direction(engagement=EngagementType.High, gaze_direction=(gaze_x, gaze_y))
                    else:
                        if looking != False:
                            looking = False
                            self.yield_gaze_direction(engagement=EngagementType.Low, gaze_direction=(gaze_x, gaze_y))
            except:
                # import traceback
                # traceback.print_exc()
                pass

    def dialog_end(self):
        # Set openface to non-publishing mode and wait until it is ready
        self.openface_endpoint.send(bytes(f"OPENFACE_END", encoding="ascii"))
        if self.extractor_thread:
            self.extractor_thread.join()

    def dialog_exit(self):
        # close openface process
        self.p_openface.kill()
        
