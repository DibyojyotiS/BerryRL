## An D3QN agent trying to forage for berries

(python-3 is required)

The Latest development is the approach in 
"<i>Memory_and_LowResPath\long-term-memory</i>".

Install requirements by:

    pip install -r requirements.txt

One needs to install the gym enviromment "<i>berry-field</i>" by

    pip install -e berry-field

The library essential is DRLagents and can be downloaded from https://github.com/DibyojyotiS/DRLagents. And installed by the following command after unzipping the DRLagents repo.

    pip install -e DRLagents

To run the experiment, cd into <i> DDQN-way </i>, open terminal and run the file "<i>train.py</i>". Use python3.

Note:

1. "<i>Memory_and_LowResPath\long-term-memory\BACKUP</i>". They may or may-not run out of the box since the library DRLagetns has evolved through time. Some new arguments (and functionalities) have been added and positions of arguments shifted in the init of DDQN class and some class methods. The code might require minor changes to run.

2. You should also install ffmpeg if not already available.

    conda install -c conda-forge ffmpeg
