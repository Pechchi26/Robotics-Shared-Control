# Setup Guide - MuJoCo Simulation with Go2Py Control

## Prerequisites

This project requires MuJoCo simulation environment with Unitree robot models and Go2Py control library for the Unitree Go2 robot.

## Installation Steps

### 1. Unitree MuJoCo Installation

First, install the Unitree MuJoCo environment which provides the robot models and simulation framework.

Follow the installation instructions from the official repository:
[https://github.com/unitreerobotics/unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)

This provides:
- MuJoCo models for Unitree robots (Go2, G1, H1, etc.)
- Simulation environment setup
- Basic visualization tools

### 2. Go2Py Installation

After setting up the MuJoCo environment, install Go2Py for robot control interfaces.

Follow the installation instructions from the official repository:
[https://github.com/machines-in-motion/Go2Py](https://github.com/machines-in-motion/Go2Py)

## Go2Py Integration

### Low-Level Control
This project utilizes Go2Py's pre-implemented low-level control interface for:
- Direct motor commands in MuJoCo simulation
- Real-time state feedback
- Joint position/velocity/torque control

### High-Level Control
Go2Py's high-level control layer provides:
- Motion planning capabilities
- Gait pattern generation
- State estimation
- Seamless integration with MuJoCo physics simulation

### MuJoCo Environment Integration
Go2Py provides native MuJoCo simulation support, enabling direct control of the Unitree Go2 robot within the MuJoCo environment without additional middleware.

## References

- [Unitree MuJoCo Repository](https://github.com/unitreerobotics/unitree_mujoco)
- [Go2Py Repository](https://github.com/machines-in-motion/Go2Py)

## Notes

- Ensure MuJoCo is properly installed before installing Go2Py
- The control interfaces work directly with the MuJoCo simulation environment
- Both low-level and high-level control modes are supported
- Computational requirement is moderate; still, CUDA is better than CPU
