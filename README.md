# PPO with SAN

## Inspired by this paper https://arxiv.org/pdf/1904.03367.pdf

### Tutorial
- Networks are defined in *networks.py*
- PPO is defined in *agents.py*
- Training is defined in *train.py*

Remaining classess suplement training. For instance *workers.py* define worker class so our agent can train in multiple instances of an environment simultaneously and learn more generalized policies. *gym_wrappers.py* simplifies GYM environment by scalling it down, removing RGB, stacking observations and ommiting some frames. *colab_train.py* is used for training in Google Colab. 

Models can be found in *models* folder. Stats are recorded with tensorboard and can be found in *runs* folder.

To see agent playing, use *play.py*. You can specify model in "suffix" parameter (last model is the best).

Best average score is 2800.
Best observer score (with play.py) is 3200.