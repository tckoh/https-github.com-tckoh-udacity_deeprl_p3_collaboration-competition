# Project 3: Collaboration and Competition 


### Background

For this project, the Unity ML-Agents Tennis simulated environment was used to train two agents to control rackets to bounce a ball over a net. A reward of +0.1 is given to agents for hitting the ball over the net. Conversely, if an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.


### Environment Details
The state space consists of 24 variables (3 stacked vector observations of size 8) corresponding to the position and velocity of the ball and racket. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. The actions are clipped to between -1 and 1.

The environment is considered solved if an average reward (over 100 episodes) of at least +0.5 can be achieved, which is based on the maximum score for the two agents for each episode.


### System Setup on Linux Machine
Step 1: Clone the DRLND Repository ([click here](https://github.com/udacity/deep-reinforcement-learning#dependencies) for more information)

Step 2: Download the Unity Environment ([click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip) for more information)

Step 3: Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file

Step 4: Download and install Anaconda ([click here](https://www.anaconda.com/distribution/) for more information)

Step 5: Create and activate a virtual environment ("drlnd") with the following libraries:
* conda create --name drlnd python=3.6 
* activate drlnd

* UnityAgents (ver. 0.4.0) ([click here](https://pypi.org/project/unityagents/) for more information)
* Numpy ([click here](https://anaconda.org/anaconda/numpy) for more information)
* Random ([click here](https://pypi.org/project/random2/) for more information)
* Sys 
* Torch ([click here](https://pytorch.org/) for more information)
* Matplotlib ([click here](https://anaconda.org/conda-forge/matplotlib) for more information)
* Collections ([click here](https://anaconda.org/lightsource2-tag/collection) for more information)

Step 6: Create an IPython kernel for the "drlnd" environment
* python -m ipykernel install --user --name drlnd --display-name "drlnd"

### Instructions
Step 1: Start Jupyter Notebook

Step 2: Navigate to the folder for this project (in the `p3_collab-compet/` folder)

Step 3: Open the Tennis jupyter notebook

Step 4: Change Kernel to 'drlnd'

Step 5: Run the cells as required 


### Other Information:

Refer to Report.pdf for more information on the learning algorithms, hyperparameters, architecture for neural network models etc.
