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

"""Utility for the emotion recognition script that needs the utterance a s file"""
import os

import librosa


def sound_array_to_file(filepath, sampling_rate, sound_as_array):
    """
    Saves the recording of the recorder to a file
    
    Turns the audio from the recorder service into a wav file for 
    processing with opensmile c++ scripts 

    filepath (string): full path, including filename and .wav suffix
    at an arbitrary location. Careful: python takes paths as
    relative to the main script. The name should be unique, to
    ensure files don't get mixed up if there are multiple calls
    in short time and one file might get overwriteen or deleted
    before it's done being processed.
    sampling_rate (int): the sampling rate of the audio, as
    published by the recorder
    sound_as_array (np.array): the audio in form of an array as
    published by the recorder
    """
    librosa.output.write_wav(filepath, sound_as_array, sampling_rate)


def delete_file(filepath):
    """
    Deletes the file at the given path to clean up the audio file
    once it's not needed anymore. This is why unique filenames are
    important.
    
    filepath (string): path to the file that is to be deleted
    """
    if os.path.exists(filepath):
        os.remove(filepath)
    else:
        print("The file cannot be deleted, as it was not found. "
              "Please check the provided path for errors: \n{}".format(filepath))
