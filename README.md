# masters-thesis

This repository contains three projects:
* GPSymbReg
  - symbolic regression of a real-value function using genetic programming
* BoolGPSymbReg
  - symbolic regression of a boolean function using genetic programming
* CudaNetEvolution
  - implementation of a feed-forward neural network with GPU evaluation support and ClonAlg learning algorithm


## Environment

These projects are successfully tested in the following environment:
* nVidia Titan Xp
* Ubuntu 18.04 LTS
* nVidia Graphics Driver 410
* nVidia CUDA Toolkit 9.1
* GCC 6.5.0
* CMake 3.9


## Installation

Install graphics drivers:

    sudo add-apt-repository ppa:graphics-drivers/ppa
    sudo apt update && sudo apt upgrade
    sudo apt install nvidia-410	# LTS version

Install dependencies:

    sudo apt install gcc-6 g++-6 cmake build-essential
    sudo apt install nvidia-cuda-toolkit

Confirm `nvcc` version is >=9.1

    nvcc --version


## Usage

To run specific project, go to its directory and follow instructions.

