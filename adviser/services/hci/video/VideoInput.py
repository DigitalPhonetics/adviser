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
import datetime
import time
from threading import Thread, Event
from typing import List

import cv2

from services.service import Service, PublishSubscribe


class VideoInput(Service):
    """
    Captures frames with a specified capture interval between two consecutive dialog turns and returns a list of frames.
    """

    def __init__(self, domain=None, camera_id: int = 0, capture_interval: int = 10e5, identifier: str = None):
        """
        Args:
            camera_id (int): device id (if only 1 camera device is connected, id is 0, if two are connected choose between 0 and 1, ...)
            capture_interval (int): try to capture a frame every x microseconds - is a lower bound, no hard time guarantees (e.g. 5e5 -> every >= 0.5 seconds)
        """
        Service.__init__(self, domain, identifier=identifier)
        
        self.cap = cv2.VideoCapture(camera_id)  # get handle to camera device
        if not self.cap.isOpened():
            self.cap.open()                     # open
        
        self.terminating = Event()
        self.terminating.clear()
        self.capture_thread = Thread(target=self.capture) # create thread object for capturing
        self.capture_interval = capture_interval

    def capture(self):
        """
        Continuous video capture, meant to be run in a loop.
        Calls `publish_img` once per interval tick to publish the captured image.
        """
        while self.cap.isOpened() and not self.terminating.isSet():
            start_time = datetime.datetime.now()

            # Capture frame-by-frame
            # cap.read() returns a bool (true when frame was read correctly)
            ret, frame = self.cap.read()
            # Our operations on the frame come here
            if ret:
                rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.publish_img(rgb_img=rgb_img)

            end_time = datetime.datetime.now()
            time_diff = end_time - start_time
            wait_seconds = (self.capture_interval - time_diff.microseconds)*1e-6   # note: time to wait for next capture to match specified sampling rate in seconds
            if wait_seconds > 0.0:
                time.sleep(wait_seconds)
            
        if self.cap.isOpened():
            self.cap.release()
    
    def dialog_end(self):
        self.terminating.set()

    def dialog_start(self):
        if not self.capture_thread.is_alive():
            print("Starting video capture...")
            self.capture_thread.start()

    @PublishSubscribe(pub_topics=['video_input'])
    def publish_img(self, rgb_img) -> dict(video_input=List[object]):
        """
        Helper function to publish images from a loop.
        """
        return {'video_input': rgb_img}  # NOTE: in the future, copy frames for more safety (capturing thread may overwrite them)
