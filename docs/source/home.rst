*******
ADvISER
*******

Introduction
============

ADVISER is an open source dialog system framework for education
and research purposes. This system supports multi-domain task-oriented conversations in
English and German. It additionally provides a flexible
architecture in which modules can be arbitrarily
combined or exchanged - allowing for
easy switching between rules-based and neural
network based implementations. Furthermore,
ADVISER offers a transparent, user-friendly
framework designed for interdisciplinary collaboration:
from a flexible back end, allowing easy integration of new features, to an intuitive
graphical user interface supporting non-technical users.

Guiding Principles
==================

Modularity: For each module in a classic dialog
system pipeline (NLU, BST, dialog policy and
NLG), we provide a handcrafted baseline module,
additionally we provide a machine learning based
implementation for the BST and policy. These can be used to quickly assemble
a working dialog system or as implementation
guidelines for custom modules. Additionally, because
all modules inherit from the same abstract
class, technical users can also easily write their
own implementations or combinations of modules.

Flexibility: In contrast to a more static dialog
system pipeline, we propose a graph structure
where the user is in control of the modules and
their order. This level of control allows users to
realise anything from pipelines to end-to-end systems.
Even branching scenarios are possible as
demonstrated by our meta policy which combines
multiple parallel subgraphs into a single dialog.

Transparency: Inputs to and outputs from each
module are captured by automatically generated
XML interface descriptions, providing a transparent
view of data flow through the dialog system.

User-friendly at different levels: technical users
have the full flexibility to explore and extend the
back end; non-technical users can use our defined
modules for building systems; students from different
disciplines could easily learn the concepts
and explore human machine interaction.

Quickstart
==========
The core data structure of Adviser is the module. 
All Adviser modules will have of a ``forward()``, 
``start_dialog()``, ``end_dialog()``, ``train()``, and 
``eval()`` method, however the exact content that goes 
into these methods remains fully flexible, allowing users 
to easily implement their own dialog system architectures. 
To get started, we provide rules-based impementations for 
a natural language understanding module (NLU), belief state 
tracking module (BST), policy, and natural language 
generation module (NLG). 

Creating a dialog system can be as easy as::

    domain(name='ImsCourses')
    modules = [ConsoleInput(domain),
	       HandcraftedNLU(domain),
	       HandcraftedBST(domain),
	       HandCraftedPolicy(domain),
	       HandCraftedNLG(domain),
	       ConsoleOutput(domain)]
    my_dialog_system = DialogSystem(modules)


Once you're happy with your dialog system, you can launch a dialog by typing::

    my_dialog_system.run_dialog()

Which will start a dialog session running through the command line terminal. 

For a more in-depth tutorial, see our `tutorial <../../tutorial.html>`_.


.. _home:installation:

Installation
============

You find ADviSER code in our `Git repository <https://github.com/DigitalPhonetics/adviser>`_.


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

.. code-block:: python

    python3 -m pip install --user virtualenv

2. Create the virtual environment (replace envname with a name of your choice)

.. code-block:: python

    python3 -m venv <path-to-env>

3. Source the environment (this has to be repeated every time you want to use ADVISER inside a
new terminal session)

.. code-block:: python

    source <path-to-env>/bin/activate

4. Install the required packages

.. code-block:: python

    pip install -r requirements.txt

5. To make sure your installation is working, navigate to the adviser folder:

.. code-block:: python

    cd adviser

and execute

.. code-block:: python

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
Adviser is published under the |gpl3| license.

.. |gpl3| raw:: html

   <a href="https://www.gnu.org/licenses/gpl-3.0.de.html" target="_blank">GNU GPL 3</a>