|release| |nbsp| |license|

.. |release| image:: https://img.shields.io/github/v/release/digitalphonetics/adviser?sort=semver
   :target: https://github.com/DigitalPhonetics/adviser/releases
.. |license| image:: https://img.shields.io/github/license/digitalphonetics/adviser
   :target: #license
.. |nbsp| unicode:: 0xA0
   :trim:

Documentation
=============

    Please see the `documentation <https://digitalphonetics.github.io/adviser/>`_ for more details.

Installation
============

Note: Adviser 2.0 is currently only tested on Linux and Mac (for M1 chips see the extra section near the bottom of this file).
(Windows is possible using WSL2 or check the instructions at the bottom for an experimental Windows setup)

Downloading the code
--------------------

If ``Git`` is not installated on your machine, just download the Adviser 2.0 file available in ``relases`` section. Then unzip and navigate to the main folder.
Note that this method has some disadvantages (you'll only be able to run basic text-to-text terminal conversations).

Cloning the repository (recommended)
------------------------------------

If ``Git`` is installed on your machine, you may instead clone the repository by entering in a terminal window:

.. code-block:: bash

    git clone https://github.com/DigitalPhonetics/adviser.git

System Library Requirements
---------------------------

* If you want to use speech in-/output, please make sure you have the `hdf5`, `portaudio` and `sndfile` libraries installed.
* If you want to make use of the function ``services.service.Service.draw_system_graph``,
you will need to install the ``graphviz`` library via your system's package manager.
If you can't install it (no sufficient user rights), don't use this function in your scripts.

On Ubuntu e.g.:

``sudo apt-get install graphviz``

On Mac, you will need to install homebrew by executing:

``/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"``

and then calling ``brew install graphviz``.

For other OS please see https://graphviz.gitlab.io/download/.


Install python requirements with pip
------------------------------------

ADvISER needs to be executed in a Python3 environment.

Once you obtained the code, navigate to its top level directory where you will find the file
``requirements_base.txt``, which lists all modules you need to run a basic text-to-text version of ADvISER. We suggest to create a
virtual environment from the top level directory, as shown below, followed by installing the necessary packages.


1. (Requires pip or pip3) Make sure you have virtualenv installed by executing

.. code-block:: bash

    python3 -m pip install --user virtualenv

2. Create the virtual environment (replace envname with a name of your choice)

.. code-block:: bash

    python3 -m venv <path-to-env>

3. Source the environment (this has to be repeated every time you want to use ADVISER inside a
new terminal session)

.. code-block:: bash

    source <path-to-env>/bin/activate

4. Install the required packages

.. code-block:: bash

    pip install -r requirements_base.txt 
 
(NOTE: or requirements_multimodal.txt if you want to use ASR / TTS)


5. Navigate to the adviser folder

.. code-block:: bash

    cd adviser

and, to make sure your installation is working, execute


.. code-block:: bash

    python run_chat.py lecturers
    
You can type text to chat with the system (confirm your utterance by pressing the ``Enter``-Key once) or type ``bye`` (followed by pressing the ``Enter``-Key once) to end the conversation.

To see more of the available options, run

.. code-block:: bash

    python run_chat.py --help


6. OPTIONAL: If you want to use multimodal functionallity, e.g. ASR / TTS/ ..., download the models via the script ``download_models.sh`` found in the top level folder

.. code-block:: bash

    sh download_models.sh
   
NOTE: this also requires you to install ``requirements_multimodal.txt`` in ``step 4``.

You can enable ASR / TTS by adding ``--asr`` and ``--tts`` to the command line options of ``run_chat.py`` (NOTE: for TTS, we recommend you run the code on a CUDA-enabled device and append ``--cuda`` to the command line options for drastic performance increase).

7. OPTIONAL: If you want to run the demo with all services enabled, please make sure you executed step 6 and installed the  ``requirements_multimodal.txt``. Then, additional requirements must be compiled by yourself - follow the guide in ``tools/OpenFace/how_to_install.md`` for this.

Then, try running 

``python run_demo_multidomain.py``



Instructions for Macs with M1 Chips 
===================================

In general, everything should work if you're using ``conda`` instead of ``pip``.
For pip users, the following installation instructions worked:

1. Install the system library requirements as stated above (using ``homebrew``).

2. Install additional reqiuirements: ``brew install rust`` and ``brew install portaudio``

2.  pip install -i https://pypi.anaconda.org/numba/label/wheels_experimental_m1/simple numba

3. Remove pyaudio from the requirements file and instead execute this command to install pyaudio:

.. code-block:: bash
    
    python -m pip install --global-option='build_ext' --global-option='-I/opt/homebrew/Cellar/portaudio/19.7.0/include' --global-option='-L/opt/homebrew/Cellar/portaudio/19.7.0/lib' pyaudio

4. Proceed with installing requirements as described above

5. Switch to the adviser folder ``cd adviser`` (containing the ``run_chat.py`` file)

6. Copy the snd library into the current folder:

.. code-block:: bash
    
    cp /opt/homebrew/lib/libsndfile.dylib
    

Experimental Windows Instructions
====================================

NOTE: Windows support is not thoroughly tested so far and in experimental stage! Only tested on Windows 11 so far.
If you encounter an error message about failing to build some library while installing the python dependencies, try installing the vcc build tools and repeat the failing step (https://visualstudio.microsoft.com/de/visual-cpp-build-tools/, yselect Desktop Development with C++ in installer).


0. Install Anaconda from https://www.anaconda.com/
   IMPORTANT: The following commands have to be executed from the Anaconda prompt!
   
1. Create a virtual env for python3.8 using conda 
   (there are no precompiled pyaudio packages for newer python versions at the time of writing)

.. code-block:: bash
   
   conda create -n YOURVIRTUALENV python=3.8

2. Install pytorch from https://pytorch.org/get-started/locally/ .
   Select options ``build: stable``, ``os0: windows``, ``package: conda``, ``language: python``, ``compute platform: cuda XX.X`` if you have an NVIDIA GPU, else ``platform: cpu``
   
3. Download sqlite3 precompiled library for Windows from https://www.sqlite.org/download.html .
   After unzipping, you will find a file ``sqlite3.dll``. 
   Copy this file into the DLL folder of your virtual environment (usually located at ``C:\Users\YOURSELF\anaconda3\envs\YOURVIRTUALENV\DLLs\``).

4. Download and install grapviz installer for windows (version 4.X): https://www.graphviz.org/download/

If you don't want a multimodal setup, SKIP STEPS 5) and 6)

5. Install precompiled pyaudio

.. code-block:: bash
   
   conda install pyaudio
   
6. Download trained models from http://adviserresources.ims.uni-stuttgart.de/models/adviser_models.zip and unzip into ``adviser/resources/models`` (folder 'models' does not exist initially)

7. Remove from the files ``requirements.txt`` and ``requirements_multimodal.txt`` the lines starting with ``torch``, ``torchaudio``, ``PyAudio``.

8. Install the requirements from either ```requirements.txt`` or ``requirements_multimodal.txt`` if you want a multimodal setup.

Building the documentation
==========================

1. Install the Python packages from ``requirements_doc.txt``.

2. Run ``PYTHONPATH=./adviser mkdocs build`` or ``PYTHONPATH=./adviser mkdocs gh-deploy`` for pushing directly to GitHub Pages.

Support
=======
You can ask questions by sending emails to adviser-support@ims.uni-stuttgart.de.

You can also post bug reports and feature requests in GitHub issues.

.. _home:how_to_cite:

How to cite
===========
If you use or reimplement any of this source code, please cite the following paper:

.. code-block:: bibtex

   @InProceedings{
    title =     {ADVISER: A Toolkit for Developing Multi-modal, Multi-domain and Socially-engaged Conversational Agents},
    author =    {Chia-Yu Li and Daniel Ortega and Dirk V{\"{a}}th and Florian Lux and Lindsey Vanderlyn and Maximilian Schmidt and Michael Neumann and Moritz V{\"{o}}lkel and Pavel Denisov and Sabrina Jenne and Zorica Karacevic and Ngoc Thang Vu},
    booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020) - System Demonstrations},
    publisher = {Association for Computational Linguistics},
    location =  {Seattle, Washington, USA},
    year =      {2020}
    }

License
=======
Adviser is published under the GNU GPL 3 license.
