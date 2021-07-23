#!/bin/bash
# install dependencies & tools
sudo apt-get install build-essential
sudo apt-get install g++-8
sudo apt-get install cmake
sudo apt-get install libopenblas-dev
sudo apt-get install git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev
sudo apt-get install libboost-all-dev
# get, compile + install opencv 4.1
wget https://github.com/opencv/opencv/archive/4.1.0.zip
unzip 4.1.0.zip
cd opencv-4.1.0
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D BUILD_TIFF=ON -D WITH_TBB=ON ..
make -j4
sudo make install
cd ../..
# get, compile + install dlib 19.13
wget http://dlib.net/files/dlib-19.13.tar.bz2
tar xf dlib-19.13.tar.bz2
cd dlib-19.13
mkdir build
cd build
cmake ..
cmake --build . --config Release
sudo make install
sudo ldconfig
cd ../..
# get anc compile libzmq and cppzmq
wget https://github.com/zeromq/libzmq/releases/download/v4.3.2/zeromq-4.3.2.zip
unzip zeromq-4.3.2.zip
cd zeromq-4.3.2
mkdir build
cd build
cmake ..
sudo make -j4 install
cd ../..
wget https://github.com/zeromq/cppzmq/archive/v4.6.0.zip
unzip v4.6.0.zip
cd cppzmq-4.6.0/
mkdir build
cd build
cmake ..
sudo make -j4 install
cd ../..
# get openface, don't overwrite existing files, compile
echo Installing OpenFace from https://github.com/TadasBaltrusaitis/OpenFace/ - please read and respect the respective README.md and Copyright.txt files
wget https://github.com/TadasBaltrusaitis/OpenFace/archive/OpenFace_2.2.0.zip
unzip OpenFace_2.2.0.zip
cp -r -n OpenFace-OpenFace_2.2.0/* OpenFace
rm -rf OpenFace-OpenFace_2.2.0
rm OpenFace_2.2.0.zip 
cd OpenFace
mkdir build
cd build
cmake -D CMAKE_CXX_COMPILER=g++-8 -D CMAKE_C_COMPILER=gcc-8 -D CMAKE_BUILD_TYPE=RELEASE ..
make -j4
cd ../
# get models, copy them to correct loclation
sh download_models.sh
cp lib/local/LandmarkDetector/model/patch_experts/*.dat build/bin/model/patch_experts/
cp -r lib/3rdParty/OpenCV/classifiers build/bin
