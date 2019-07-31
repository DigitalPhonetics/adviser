Please see the `documentation <https://digitalphonetics.github.io/adviser/>`_ for more details.

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


1. Make sure you have virtualenv installed by executing

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

    pip install -r requirements.txt

5. To make sure your installation is working, navigate to the adviser folder:

.. code-block:: bash

    cd adviser

and execute

.. code-block:: bash

    python run_chat.py --domain courses

6. Select a language by entering `english` or `german`, then chat with ADvISER. To end your
conversation, type `bye`.

Support
=======
You can ask questions by sending emails to adviser-support@ims.uni-stuttgart.de

You can also post bug reports and feature requests (only) in GitHub issues. Make sure to read our guidelines first.

.. _home:how_to_cite:

How to cite
===========
If you use or reimplement any of this source code, please cite the following paper:

.. code-block:: bibtex

   @InProceedings{adviser19,
   title =     {ADVISER: A Dialog System Framework for Education & Research},
   author =    {Daniel Ortega and Dirk V{\"{a}}th and Gianna Weber and Lindsey Vanderlyn and Maximilian Schmidt and Moritz V{\"{o}}lkel and Zorica Karacevic and Ngoc Thang Vu},
   booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019) - System Demonstrations},
   publisher = {Association for Computational Linguistics},
   location =  {Florence, Italy},
   year =      {2019}
   }

License
=======
Adviser is published under the GNU GPL 3 license.