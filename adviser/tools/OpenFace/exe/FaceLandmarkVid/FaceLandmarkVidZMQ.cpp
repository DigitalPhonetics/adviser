/*
 Copyright 2019, University of Stuttgart: Institute for Natural Language Processing (IMS)

 This file is part of Adviser.
 Adviser is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3.

 Adviser is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with Adviser.  If not, see <https://www.gnu.org/licenses/>.

  Parts of this file contain code snippets from OpenFace, subject to the
  following license:
*/ 
///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
//
// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  
// IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
//
// License can be found in OpenFace-license.txt

//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace 2.0: Facial Behavior Analysis Toolkit
//       Tadas Baltru�aitis, Amir Zadeh, Yao Chong Lim, and Louis-Philippe Morency
//       in IEEE International Conference on Automatic Face and Gesture Recognition, 2018  
//
//       Convolutional experts constrained local model for facial landmark detection.
//       A. Zadeh, T. Baltru�aitis, and Louis-Philippe Morency,
//       in Computer Vision and Pattern Recognition Workshops, 2017.    
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltru�aitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-specific normalisation for automatic Action Unit detection
//       Tadas Baltru�aitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
///////////////////////////////////////////////////////////////////////////////
// FaceTrackingVid.cpp : Defines the entry point for the console application for tracking faces in videos.
#include <string>

// Libraries for landmark detection (includes CLNF and CLM modules)
#include "LandmarkCoreIncludes.h"
#include "GazeEstimation.h"

#include <SequenceCapture.h>
#include <Visualizer.h>
#include <VisualizationUtils.h>

#include <zmq.hpp>
#include <thread>
#include <condition_variable>
#include <pthread.h>
#include <opencv2/core/core.hpp>
#include <sstream>
#include <atomic>
#include <mutex>

#define PUBLISH_IMAGE false

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

static void printErrorAndAbort(const std::string & error)
{
	std::cout << error << std::endl;
	abort();
}

#define FATAL_STREAM( stream ) \
printErrorAndAbort( std::string( "Fatal error: " ) + stream )

std::vector<std::string> get_arguments(int argc, char **argv)
{

	std::vector<std::string> arguments;

	for (int i = 0; i < argc; ++i)
	{
		arguments.push_back(std::string(argv[i]));
	}
	return arguments;
}

void await_start_msg(void* socket)
{
	bool starting = false;
	while(!starting) {
		char buffer [100];
        zmq_recv (socket, buffer, 100, 0);
		std::string msg(buffer);
		if(msg.compare("OPENFACE_START") == 0)
			starting = true;
	}
}

std::string angle_to_json(cv::Vec2d& gazeAngle)
{	
	// convert gaze to json
	std::stringstream msg_builder;
	msg_builder << "{";
	msg_builder << "\"gaze\": {";
	msg_builder << 		  "\"angle\": {" << "\"x\": " << gazeAngle[0] << ", \"y\": " << gazeAngle[1] << "}";
	msg_builder << "}";
	msg_builder << "}";

	//if(PUBLISH_IMAGE == true){
//		msg_builder << ",";
		//msg_builder << "\"img\": \"[";  // "img": [...
		//for(int i = 0; i < vis_img.rows; i++) {
			//if(i > 0)
				//msg_builder << ",";
		//	msg_builder << "[";	// start new row [...,
		//	for(int j = 0; j < vis_img.cols; j++) {
		//		if(j > 0)
		//			msg_builder << ",";
				//msg_builder << vis_img.at<int>(i,j);
	//		}
	//		msg_builder << "]";	// end row ,...]
	//	}
	//	msg_builder << "]\"";  // closing "img": "[...]"
		
	//}
	//msg_builder << "}";
	return msg_builder.str();
}

int main(int argc, char **argv)
{
	std::vector<std::string> arguments = get_arguments(argc, argv);
	std::mutex m;

	// no arguments: output usage
	if (arguments.size() == 1)
	{
		std::cout << "For command line arguments see:" << std::endl;
		std::cout << " https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments";
		return 0;
	}
	
	// INIT Model
	LandmarkDetector::FaceModelParameters det_parameters(arguments);
	LandmarkDetector::CLNF face_model(det_parameters.model_location);
	if (!face_model.loaded_successfully)
	{
		std::cout << "ERROR: Could not load the landmark detector" << std::endl;
		exit(0);
	}
	if (!face_model.eye_model)
	{
		std::cout << "WARNING: no eye model found" << std::endl;
	}
	// Open a sequence
	Utilities::SequenceCapture sequence_reader;
	// A utility for visualizing the results (show just the tracks)
	Utilities::Visualizer visualizer(true, false, false, false);
	// The sequence reader chooses what to open based on command line arguments provided
	if (!sequence_reader.Open(arguments)) {
		std::cout << "Failed to open sequence reader" << std::endl;
		exit(0);
	}

	INFO_STREAM("Device or file opened");
	cv::Mat rgb_image = sequence_reader.GetNextFrame();
	while(rgb_image.empty()) {
		rgb_image = sequence_reader.GetNextFrame();
	}
	visualizer.SetImage(rgb_image, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy);
	char character_press = visualizer.ShowObservation();
	INFO_STREAM("Starting tracking");

	//int PUB_PORT = 6004;
	std::cout << "Setup connection on port 6004..." << std::endl;
	zmq::context_t context (1);
    zmq::socket_t socket (context, ZMQ_PAIR);
	socket.connect("tcp://127.0.0.1:6004");

	bool publish = false;
	while(!publish) {
		zmq::message_t reply;
		socket.recv (&reply, 0);
		std::string msg_str = std::string(static_cast<char*>(reply.data()), reply.size());
		if(msg_str.compare("OPENFACE_START") == 0) {
			// TODO SEND REPLY: OPENFACE_STARTED
			std::string start_msg = "OPENFACE_STARTED";
			zmq::message_t rep (start_msg.size());
			memcpy (rep.data (), start_msg.c_str(), start_msg.size());
			socket.send(rep, 0);
			publish = true;
		}
	}


	// get next message from client and parse to string
	while(publish) {
		zmq::message_t reply;
		socket.recv (&reply, 0);
		std::string msg_str = std::string(static_cast<char*>(reply.data()), reply.size());

		// switch functionallity on / off depending on message
		if(msg_str.compare("OPENFACE_PULL") == 0) {

			// get next feature
			// start extracting features and publishing
			rgb_image = sequence_reader.GetNextFrame();
			if (!rgb_image.empty()) 
			{
				// Reading the images
				cv::Mat_<uchar> grayscale_image = sequence_reader.GetGrayFrame();
				// The actual facial landmark detection / tracking
				bool detection_success = LandmarkDetector::DetectLandmarksInVideo(rgb_image, face_model, det_parameters, grayscale_image);

				// Gaze tracking, absolute gaze direction
				cv::Point3f gazeDirection0(0, 0, -1);
				cv::Point3f gazeDirection1(0, 0, -1);
				cv::Vec2d gazeAngle(0, 0);

				// If tracking succeeded and we have an eye model, estimate gaze
				if (detection_success && face_model.eye_model)
				{
					GazeAnalysis::EstimateGaze(face_model, gazeDirection0, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy, true);
					GazeAnalysis::EstimateGaze(face_model, gazeDirection1, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy, false);
					gazeAngle = GazeAnalysis::GetGazeAngle(gazeDirection0, gazeDirection1);
				}

				// Work out the pose of the head from the tracked model
				cv::Vec6d pose_estimate = LandmarkDetector::GetPose(face_model, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy);

				// Keeping track of FPS
				// fps_tracker.AddFrame();

				// Displaying the tracking visualizations
				visualizer.SetImage(rgb_image, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy);
				visualizer.SetObservationLandmarks(face_model.detected_landmarks, face_model.detection_certainty, face_model.GetVisibilities());
				visualizer.SetObservationPose(pose_estimate, face_model.detection_certainty);
				visualizer.SetObservationGaze(gazeDirection0, gazeDirection1, LandmarkDetector::CalculateAllEyeLandmarks(face_model), LandmarkDetector::Calculate3DEyeLandmarks(face_model, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy), face_model.detection_certainty);
				// visualizer.SetFps(fps_tracker.GetFPS());
				// detect key presses (due to pecularities of OpenCV, you can get it when displaying images)
				char character_press = visualizer.ShowObservation();

				std::string json = angle_to_json(gazeAngle);

				zmq::message_t rep (json.size());
				memcpy (rep.data (), json.c_str(), json.size());
				socket.send(rep, 0);


				// restart the tracker
				// if (character_press == 'r')
				// {
				// 	face_model.Reset();
				// }
				// // quit the application
				// else if (character_press == 'q')
				// {
				// 	break;
				// }
			}


		}
	}

	// std::cout << "MSG: " << msg_str << std::endl;


	sequence_reader.Close();

	return 0;	
}
