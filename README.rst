giMLi
=====

This repository is for the projects in gFEX with Machine Learning applications. At the moment, it is somewhat unofficial and we are tossing around some interesting ideas.

Running
=======

To generate an HDF5 file from the gFEX ntuples we produce right now, you can run::

  python gFEXtuple2hdf5.py user.gstark.DSID.OUTPUT_vXX/*.root --output user.gstark.DSID.OUTPUT_vXX.hdf5

To run on this generated HDF5 file, you can use::

  python train_pileup.py user.gstark.DSID.OUTPUT_vXX.hdf5

Edit `train_pileup.py <train_pileup.py>`_ to switch between a "simplified" neural network and the non-simplified version, as well as determining whether to normalize (default) the data or not between ``[0,1]``.

Installing
==========

First Time Setup
----------------

Clone this repository, then set up your ``.bash_profile`` (or ``.bashrc``)::

  export PATH=$HOME/.local/bin:$PATH
  export PYTHONPATH=$HOME/.local/lib/python2.7/site-packages:$PYTHONPATH

which will add locally installed binaries and update your python path if needed. Last, there is a custom helper function I wrote to set up a virtual environment after the first time use::

  venv(){
    export WORKON_HOME=$HOME/.virtualenvs
    export PROJECT_HOME=$HOME/Devel
    source $HOME/.local/bin/virtualenvwrapper.sh
  }

So you just need to run ``venv`` in your shell to have all paths/commands setup correctly. Next, run::

  lsetup "lcgenv -p LCG_89 x86_64-slc6-gcc62-opt pip"

to get ``pip`` set up. Then you can install `virtualenvwrapper <https://virtualenvwrapper.readthedocs.io/en/latest/>`_ to manage virtual environments::

  pip install --user virtualenvwrapper

Note that this package isn't part of CVMFS and will not be included with ``lsetup`` so we install it into ``$HOME/.local`` using the ``--user`` flag. Go ahead and make your first virtual environment with::

  mkvirtualenv gML

From now on, you can always type ``workon`` to see all environments you have, and ``workon gML`` to switch to ``gML`` for example. Go ahead and finish this setup with::

  pip install -r requirements.txt

Nth Time Setup
--------------

Simply set up virtualenv paths and switch to your python environment::

  venv
  workon gML

and keep on hacking!

Test GPU
--------

Mac OSX
~~~~~~~

I have a mid-2014 Mac with an NVidia GPU. After following all instructions, I can run it with the old backend::

  THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python test.py

Tutorials and References
========================

- `Keras Tutorial @ ATLAS Machine Learning Workshop (June 2017) <https://indico.cern.ch/event/630665/contributions/2605129/>`_
- `Michael Kagan's Keras setup on CERN LXPLUS <https://indico.cern.ch/event/615994/page/10686-lxplus-software-setup>`_
