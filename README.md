# DiffTaichi-Evolutionary-Rigid-Body-Simulation

https://www.youtube.com/watch?v=0LJmqNRU0Ps

**A Taichi-based rigid body simulation with evolutionary optimization for robotic locomotion.**

This project implements a 2D physics simulation of rigid bodies connected by springs and joints, optimized using an evolutionary algorithm and gradient based optimization to achieve locomotion toward a target goal.

**How To Run**

Install necessary dependencies
pip install -r requirements.txt

Run the simulation
python rigid_body.py

**How It Works**

**Robot Definition:** Each robot consists of rigid bodies (position, size, rotation) connected by springs or joints, defined in robot_config.py.

**Simulation:** Forward dynamics compute positions over 2048 timesteps with a sinusoidal actuation signal.
Loss Function: Defined by distance to goal ([0.9, 0.4] by default)

**Evolution:**

Generates an initial population with random mutations.

Applies crossover to create children.

Mutates children derived from crossover.

Optimizes each child using gradient descent (5 iterations).

Selects the top 5 performers based on fitness (-loss).

Continues for designated generations (10 at default) with the subsequent populations

**Output:** Visualizes the best robot and plots fitness over generations.

**Acknowledgments**

Built with Taichi and adapted from

https://github.com/taichi-dev/difftaichi/blob/master/examples/rigid_body.py

https://github.com/taichi-dev/difftaichi/blob/master/examples/robot_config.py
