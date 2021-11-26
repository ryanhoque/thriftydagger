# ThriftyDAgger

Code for the ThriftyDAgger algorithm. See the [website](https://sites.google.com/view/thrifty-dagger/home) for the paper and other materials.

# Instructions for generating data + training
1. Create a virtual environment:
    ```
    conda create --name thriftydagger python=3.8;
    conda activate thriftydagger
    ```
2. Install necessary packages:
    - Install MuJoCo (see the https://github.com/openai/mujoco-py README). You will need an access key.
    - Run: 
        ```
        pip install -e .
        ```
3. Collect demonstration data. Run:
    ```
    python run_thriftydagger.py --generate_data
    ```



# ======= original readme =======
## Installation

0. Start a Python virtual environment to self-contain all the code:
```
virtualenv [name-of-env]
source [name-of-env]/bin/activate
```
1. Install MuJoCo (see the https://github.com/openai/mujoco-py README). You will need an access key.
2. Install this python package: 
```
pip install -e .
```

Note that the teleoperation code has only been tested extensively on Mac OS X, but should work for Linux too.

## Running ThriftyDAgger on Peg Insertion

Run the following command:
```
python scripts/run_thriftydagger.py [experiment_name] [--targetrate SWITCH_RATE] [--gen_data]
``` 
where `[experiment_name]` defines where the logging will happen (i.e. in `data/experiment_name`), `[--targetrate]` allows you to specify the intervention budget as the target ratio of switches to autonomous actions, and `--gen_data` is True if you need to collect offline demonstrations beforehand (the repo already has a `robosuite-30.pkl` with 30 demos of the peg insertion task so you can skip this if you want). Feel free to change other hyperparameters in the call to `thrifty()` in `thrifty/algos/thriftydagger.py`.

The robot in the simulator will stop moving and `Switch to Human` will be printed when it wants help. You can then teleoperate the robot with the keyboard (instructions will be printed to the terminal immediately after running the command above). This takes a bit of practice to become intuitive. Since we model the policy as *deterministic*, try to tie-break consistently when there are "multiple right answers" (e.g. try not to accidentally rotate right when you mean to rotate left). Rotation around the x/y axes is disabled as they are quite unintuitive, so the only operations are translating the arm in the XY-plane, moving the arm up and down the Z-axis, rotating the gripper around the Z-axis, and opening/closing the gripper. 

When collecting `robosuite-30.pkl` and providing interventions, we had the following procedure:
1. Move the arm in the XY-plane until it is over the protruding part of the gray washer.
2. Rotate the arm around the Z-axis until the gripper is perpendicular to the protrusion. Rotate in the direction that is closer.
3. Lower the arm until the gripper is at the washer's height.
4. Close the gripper.
5. Lift the washer off the ground.
6. Rotate the arm so that the washer faces the gray cylinder. Rotate in the direction that is closer.
7. Translate over to the gray cylinder.
8. Lift the washer until it is above the top of the gray cylinder.
9. Align the washer with the cylinder.
10. Lower the washer until it hits the ground, threading the cylinder through.

This procedure is shown briefly on the [website](https://sites.google.com/view/thrifty-dagger/home).

## Running ThriftyDAgger on Other Robosuite Environments

To run on other Robosuite environments such as block stacking, simply use the `--environment` option when running `scripts/run_thriftydagger.py` and specify the name of the Robosuite environment. You may wish to modify the episode horizon in Line 167 of `scripts/run_thriftydagger.py`.

## Running Baselines

To run HG-DAgger, simply run the same command but with the `--hgdagger` option set:
```
python scripts/run_thriftydagger.py [experiment_name] --hgdagger
``` 
Press the 'Z' key to start or end interventions with HG-DAgger.

To run LazyDAgger, run:
```
python scripts/run_thriftydagger.py [experiment_name] --lazydagger
```

To run Behavior Cloning, or evaluate ThriftyDAgger/HG-DAgger after training, run:
```
python scripts/run_thriftydagger.py [experiment_name] --eval [PATH_TO_MODEL] --iters 0
``` 
Models are saved in `data/experiment_name/experiment_name_s[seed]/pyt_save/model.pt` during training.

## Other

Contact `ryanhoque@berkeley.edu` for the code for physical experiments, which is largely the same but has extra code to interface with the da Vinci surgical robot.
