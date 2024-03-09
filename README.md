# E2E-CARLA-ReinforcementLearning-PPO

## Overview

This project is an end-to-end application of reinforcement learning for collision avoidance in autonomous vehicles using the CARLA simulator. It uses the Proximal Policy Optimization (PPO) algorithm, within the CARLA environment (version 0.9.12) to teach a virtual car to avoid collisions at several speeds. This is an end-to-end approach wherethe input is a camera RGB image and the output are directly control values for acceleration and steering.

## Model

The reinforcement learning model used in this project is the [recurrent version]([https://stable-baselines3.readthedocs.io/](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html)) of the Proximal Policy Optimization (PPO) algorithm, implemented through the Stable Baselines3 library. This  version allows the model to maintain a memory of its past decisions, improving its decision-making process over time, particularly in environments that require understanding sequences of events for optimal actions.

For more information on Stable Baselines3, visit their official website: [Stable Baselines3](https://stable-baselines3.readthedocs.io/).

## Training

The model was trained for 80,000 steps at a speed of 90 km/h. This initial training allowed the model to successfully learn collision avoidance strategies. It is important to note that further training could potentially improve the model's performance and effectiveness in avoiding collisions under various conditions.

## Results

Below are the results showcasing the model's ability to avoid collisions:

- **Reward vs. Step Graph**: The following graph displays the model's learning progress over 80,000 steps. It highlights the improvement in the model's ability to avoid collisions as training progresses.

![Reward vs. Step Graph](https://github.com/gustavomoers/E2E-CARLA-ReinforcementLearning-PPO/assets/69984472/8b3722f2-2156-4021-8175-ade4e0d5a600)

- **Model in Action**: Here are some GIFs demonstrating the model successfully avoiding collisions in the CARLA simulator.

![Model Working 90 km/h](https://github.com/gustavomoers/E2E-CARLA-ReinforcementLearning-PPO/assets/69984472/464d9dda-9524-40d8-92f6-dbf9c9f86c3f)

![Model Working 70 km/h](https://github.com/gustavomoers/E2E-CARLA-ReinforcementLearning-PPO/assets/69984472/fdee28d5-aedf-45db-9611-18ab532d81b0)

## Dependencies

- CARLA Simulator (version 0.9.12)
- Stable Baselines3
- Other dependencies (requirements.txt)

## How to Run

To train the model, execute train.py which supports adjustable parameters via --arg (for details, see argparser within the script). Additionally, direct adjustments may be needed in World.py.Note that self.distance_parked should be set to allow the ego car to reach desired speeds effectively.


## Conclusion

The PPO model trained on CARLA 0.9.12 for collision avoidance at 90 km/h shows promising results. While the initial training of 80,000 steps demonstrates the model's capability to learn effective avoidance strategies, further training could improve its performance. This project lays the groundwork for more advanced applications of reinforcement learning in autonomous vehicle technology.

## Contributions

I welcome contributions and suggestions to improve this project. Feel free to fork the repository, submit pull requests, or open issues to discuss potential improvements.
I plan to train this model in dynamic scenarios as well.
I am also working on a hybrid approach where I use reinforcement learning as path planner and a MPC controller to control the ego car, soon I will also publish this approach.

