###############################################################################
# This is a face detection, recognition, age/gender estimation demo made by
# Taewoon Kim (https://taewoonkim.com/).

# I wanted to put this in `run_chat.py` but I thought first I make this separate
# file as a demo, and then when I get enough feedback, I'll try to merge with
# `run_chat.py`.

# First install the required packages in ../requirements_face_age_gender.txt,
# and then run `python run_face_age_gender.py`

# In order to match the face, this script uses the pre-defined faces in the
# directory `./embeddings`. At the moment, I'm the only one there. If you want
# your faces to be recognized, please send me five photos as in 
# https://photos.app.goo.gl/LzehxftJVV8M2s1V6

###############################################################################

from utils.logger import DiasysLogger, LogLevel
from services.hci.video import VideoInput, FaceDetectionRecognition, AgeGender, AnnotateDisplayImage
from services.service import DialogSystem
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='face age gender demo')
    parser.add_argument('--camera_id', type=int, default=0,
                        help="https://askubuntu.com/questions/348838/how-to-check-available-webcams-from-the-command-line")
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--capture_interval', default=10e5,
                        help='unit is microseconds')

    args = parser.parse_args()
    args = vars(args)

    # setup logger
    file_log_lvl = LogLevel['NONE']
    log_lvl = LogLevel['RESULTS']
    conversation_log_dir = './conversation_logs'

    logger = DiasysLogger(file_log_lvl=file_log_lvl,
                          console_log_lvl=log_lvl,
                          logfile_folder=conversation_log_dir,
                          logfile_basename="full_log")

    video_service = VideoInput(camera_id=args['camera_id'],
                               capture_interval=args['capture_interval'])
    fdr_service = FaceDetectionRecognition(logger=logger)
    ag_service = AgeGender(logger=logger)
    ad_service = AnnotateDisplayImage(logger=logger)
    services = []
    services.append(video_service)
    services.append(fdr_service)
    services.append(ag_service)
    services.append(ad_service)

    ds = DialogSystem(services=services, debug_logger=logger)
    error_free = ds.is_error_free_messaging_pipeline()
    if not error_free:
        ds.print_inconsistencies()
    ds.draw_system_graph()

    ds.run_dialog({'gen_user_utterance': ""})
    # free resources
    ds.shutdown()
