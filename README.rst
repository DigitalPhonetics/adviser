  Please see the `documentation <https://digitalphonetics.github.io/adviser/>`_ for more details.

New Version
===========
Adviser 2.0 is released! 

Installation
============

Downloading the code
--------------------

If you are not familiar with `Git`, just download the zip file available in the ``Clone or Download``. Then unzip and enter the main folder.


Cloning the repository
-----------------------

If you feel comfortable with `Git`, you may instead clone the repository.

.. code-block:: bash

    git clone https://github.com/DigitalPhonetics/adviser.git


Install requirements with pip
------------------------------

ADvISER needs to be executed in a Python3 environment.

Once you have the code locally navigate to the top level directory, where you will find the file
`requirements.txt`, which lists all modules you need to run ADvISER. We suggest to create a
virtual environment from the top level directory, as shown below, followed by installing the necessary packages.


1. (You need to have pip, or pip3) Make sure you have virtualenv installed by executing

.. code-block:: bash

    python3 -m pip install --user virtualenv

2. Create the virtual environment (replace envname with a name of your choice)

.. code-block:: bash

    python3 -m venv <path-to-env>

3. Source the environment (this has to be repeated every time you want to use ADVISER inside a
new terminal session)

.. code-block:: bash

    source <path-to-env>/bin/activate

4. Navigate to the adviser folder

.. code-block:: bash

    cd adviser

5. Install the required packages

.. code-block:: bash

    pip install -r requirements_base.txt 
 
(or requirements_multimodal.txt if you want to use ASR / TTS)

and, to make sure your installation is working, execute

.. code-block:: bash

    python run_chat.py lecturers

6. If you want to use multimodal functionallity, execute

.. code-block:: bash

    git lfs pull


Support
=======
You can ask questions by sending emails to <adviser-support@ims.uni-stuttgart.de>.

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
