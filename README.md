# Duel_DQN
reinforcement learning, tensorflow + openAI gym implementation of dueling DDQN, dueling DQN
# Requirement main
	python3.6

	gym[atari]

	opencv-python

	tensorflow-1.10-gpu
# Usage
For dueling_DQN train:

	python game_main.py --episode=15000 --env_name=MsPacman-v0 --model_type=dueldqn --train=True --load_network=False

For dueling_DDQN train:

	--model_type=duelddqn
	
For test model:
	
	--train=False --load_network=True

For game name:

    --env_name=MsPacman-v0
# Result
![game_test](https://github.com/demomagic/Duel_DQN/blob/master/img/game.gif)
# Summary
	tensorboard --logdir=./summary/Solaris-v0/dueldqn
	tensorboard --logdir=./summary/Solaris-v0/duelddqn

For Dueling_DQN summary:

![duel_dqn_summary](https://github.com/demomagic/Duel_DQN/blob/master/img/dueldqn.png)

For Dueling_DDQN summary:

![duel_ddqn_summary](https://github.com/demomagic/Duel_DQN/blob/master/img/duelddqn.png)
# Reference
[Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
