# How to install OpenFace (required by engagement tracking service)

* Download the code from https://github.com/TadasBaltrusaitis/OpenFace
* Copy all file into tools/OpenFace BUT DON'T OVERWRITE EXISTING FILES

* For platform specific build instructions, see https://github.com/TadasBaltrusaitis/OpenFace/wiki 
* OpenFace Requirements:
    * boost (install via package manager)
    * TBB  (install via package manager)
    * dlib (install via package manager)
    * OpenBLAS (install via package manager)
    * wget (install via package manager)
    * opencv-4.1.0 (install via package manager / dowload source from https://github.com/opencv/opencv/archive/4.1.0.zip)
        * together with opencv_contrib-4.1.0 (install via package manager / download source from https://github.com/opencv/opencv_contrib/archive/4.1.0.zip)
        * Tip for Mac homebrew users to obtain correct version:
            * activate your python virtual environment
            * in your terminal, execute `brew edit opencv@`
            * change the url in the line starting with `url "http://https://github.com/opencv/opencv/archive/..".` to `url "https://github.com/opencv/opencv/archive/4.1.0.tar.gz"`
            * change the line underneath starting with `sha256 "..."` to `sha256 "8f6e4ab393d81d72caae6e78bd0fd6956117ec9f006fba55fcdb88caf62989b7"`
            * scroll to the line `resource "contrib" do``
            * change the line underneath starting with `url "https://github.com/opencv/opencv_contrib/archive/...."` to `url "https://github.com/opencv/opencv_contrib/archive/4.1.0.tar.gz"`
            * change the line underneath starting with `sha256 "..."` to `sha256 "e7d775cc0b87b04308823ca518b11b34cc12907a59af4ccdaf64419c1ba5e682"`
            * scroll down to the `args` section
            * change `-DWITH_QT=OFF` to `-DWITH_QT=ON`
            * add a new line with `-DBUILD_TBB=ON`
            * in your terminal, execute `brew install opencv@` - this will install opencv-4.1.0 
            * update `tools/OpenFace/CMakeLists.txt`
                * change line `find_package( OpenCV 4.0 REQUIRED COMPONENTS core imgproc calib3d highgui objdetect` to `find_package( OpenCV 4.0 REQUIRED COMPONENTS core imgproc calib3d highgui objdetect HINTS /usr/local/Cellar/opencv/4.1.0)` (or if your homebrew installs somewhere else, use this path instead for the `HINTS`)
    * qt4 (install via official installer / package manager)
        * For Mac homebrew users: `brew install cartr/qt4/pyqt@4`
* Build OpenFace:
    * `cd tools/OpenFace`
    * `mkdir build`
    * `cd build`
    * `cmake -D CMAKE_BUILD_TYPE=RELEASE ..`
    * `make` or `make -jx` where `x` is the number of parallel jobs (e.g. if you have 4 CPU's calling `make -j4` should be faster than `make`) 
* After-Build:
    * `cd .. ` brings you back to the OpenFace main folder.
    * `sh download_models.sh`
    * NOTE: the following paths assume `tools/OpenFace` as your current folder
    * check that `build/bin/model/patch_experts` contains `cen_patches_*.dat` files. If not, copy them from `lib/local/LandmarkDetector/model/patch_experts`
    * copy `lib/local/3rdParty/OpenCV/classifiers` to `build/bin`
    * verify installation by running `./tools/OpenFace/build/bin/FaceLandmarkVid -device 0` (or replace 0 with the desired camera device number)



OLD - ignore
<!-- OpenFace`
* requirements`
* mkdir build
* cd build 
* call CMake:
   (Example config: fill in qt path, extra_modules_path):
        cmake -D CMAKE_BUILD_TYPE=RELEASE \
            -D WITH_TBB=ON \
            -D WITH_V4L=ON \
            -D OPENCV_SKIP_PYTHON_LOADER=ON \
            -D CMAKE_PREFIX_PATH=/usr/local/opt/qt5 \
            -D CMAKE_MODULE_PATH="$QT5PATH"/lib/cmake \
            -D OPENCV_PYTHON3_INSTALL_PATH=~/.virtualenvs/OpenCV-"$cvVersion"-py3/lib/python3.7/site-packages \
        -D WITH_QT=ON \
        -D WITH_OPENGL=ON \
        -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.1.0/modules \
        -D BUILD_EXAMPLES=OFF ..
* make -->
