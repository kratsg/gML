giMLi
=====

This repository is for the projects in gFEX with Machine Learning applications. At the moment, it is somewhat unofficial and we are tossing around some interesting ideas.

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
