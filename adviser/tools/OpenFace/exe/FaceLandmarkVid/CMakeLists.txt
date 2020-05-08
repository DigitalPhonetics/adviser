# Local libraries
include_directories(${LandmarkDetector_SOURCE_DIR}/include)

# cppzmq
find_package(cppzmq)
	
add_executable(FaceLandmarkVid FaceLandmarkVid.cpp)
target_link_libraries(FaceLandmarkVid LandmarkDetector)
target_link_libraries(FaceLandmarkVid FaceAnalyser)
target_link_libraries(FaceLandmarkVid GazeAnalyser)
target_link_libraries(FaceLandmarkVid Utilities)

add_executable(FaceLandmarkVidZMQ FaceLandmarkVidZMQ.cpp)
target_link_libraries(FaceLandmarkVidZMQ LandmarkDetector)
target_link_libraries(FaceLandmarkVidZMQ FaceAnalyser)
target_link_libraries(FaceLandmarkVidZMQ GazeAnalyser)
target_link_libraries(FaceLandmarkVidZMQ Utilities)
target_link_libraries(FaceLandmarkVidZMQ cppzmq)

install (TARGETS FaceLandmarkVid DESTINATION bin)
install (TARGETS FaceLandmarkVidZMQ DESTINATION bin)