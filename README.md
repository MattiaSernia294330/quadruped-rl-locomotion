# ⁠Toward Realistic Locomotion: Domain and Goal Randomization for Quadruped Policy Learning

The following repository, forked from [quadruped-rl-locomotion](https://github.com/nimazareian/quadruped-rl-locomotion), contains all the code developed by Group 4 (Salvi, Sernia, Vittimberga, Zoccatelli) concerning the project extension of course *01TXFSM-Machine learning and Deep learning*, Master's program in Data Science and Engineering at Politecnico di Torino. 

## Scripts edited

Here is a list of the files inside this repository which have been edited by the group throughout the developing of the project
- `go1_mujoco_env.py` for the agent development 
- `go1_torque.xml`, `scene_torque.xml` inside **inside unitree_g01** folder for the shaping of the environment
- `train.py` for quadruped training and testing

Furthermore, the best trained models can be found in the folder **models**.


## Bash command to test models 

```bash
python train.py --run test --model_path <path to model zip file> --domain target --point {fixed, random} 
```
