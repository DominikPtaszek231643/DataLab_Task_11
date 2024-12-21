# Robotic Pipette Simulation Environment

## **Overview**

This project simulates a robotic pipette within a defined working envelope using the PyBullet physics engine. The
simulation allows for sending movement commands to the robot and receiving observations about its state. This setup is
ideal for testing robotic control algorithms, understanding robot kinematics, and validating operational boundaries. The
environment was installed according to the instructions from this
website: https://adsai.buas.nl/Study%20Content/Robotics%20and%20Reinforcement%20Learning/2.%20Robotic%20Simulation%20Primer.html

## **Contents**

- `main.py`: Main script to run the simulation.
- `sim_class.py`: Contains the `Simulation` class for interacting with PyBullet.
- `requirements.txt`: Lists all Python dependencies.
- `working_envelope.txt`: Outputs the recorded coordinates of the pipette's working envelope.

- `README.md`: This documentation file.

## **Prerequisites**

- **Python 3.7 or higher**: Ensure Python is installed on your system.
- **PyBullet**: Physics engine for simulation.

## **List of dependencies**

In file `requirements.txt`

## **Working envelope**

corner_1: [-0.1881, -0.1713, 0.1195]
corner_2: [-0.1872, -0.1705, 0.2907]
corner_3: [-0.187, 0.2213, 0.1692]
corner_4: [-0.187, 0.2195, 0.2907]
corner_5: [0.2539, -0.171, 0.1694]
corner_6: [0.2532, -0.1705, 0.2906]
corner_7: [0.253, 0.2213, 0.1693]
corner_8: [0.253, 0.2195, 0.2907]


