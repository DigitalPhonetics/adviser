Documentation
=============

    Please see the `documentation <https://digitalphonetics.github.io/adviser/>`_ for more details.

New Version
===========
Adviser 2.0 is released! 

Installation
============

Note: Adviser 2.0 is currently only tested on Linux and Mac.

Downloading the code
--------------------

If `Git` is not installated on your machine, just download the Adviser 2.0 file available in ``relases`` section. Then unzip and navigate to the main folder.
Note that this method has some disadvantages (you'll only be able to run basic text-to-text terminal conversations).

Cloning the repository (recommended)
------------------------------------

If `Git` is installed on your machine, you may instead clone the repository by entering in a terminal window:

.. code-block:: bash

    git clone https://github.com/DigitalPhonetics/adviser.git

System Library Requirements
---------------------------

If you want to make use of the function ``services.service.Service.draw_system_graph``,
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
`requirements_base.txt`, which lists all modules you need to run a basic text-to-text version of ADvISER. We suggest to create a
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


6. OPTIONAL: If you want to use multimodal functionallity, e.g. ASR / TTS/ ..., execute

.. code-block:: bash

    git lfs pull
   
NOTE: this also requires you to install ``requirements_multimodal.txt`` in ``step 4``.

You can enable ASR / TTS by adding ``--asr`` and ``--tts`` to the command line options of ``run_chat.py`` (NOTE: for TTS, we recommend you run the code on a CUDA-enabled device and append ``--cuda`` to the command line options for drastic performance increase).

7. OPTIONAL: If you want to run the demo with all services enabled, please make sure you executed step 6 and installed the  ``requirements_multimodal.txt``. Then, additional requirements must be compiled by yourself - follow the guide in ``tools/OpenFace/how_to_install.md`` for this.

Then, try running 

``python run_demo_multidomain.py``

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
