# [SATAY: A Streaming Architecture Toolflow for Accelerating YOLO Models on FPGA Devices](https://arxiv.org/abs/2309.01587)

_SATAY_ is a project for generating FPGA-based DNN accelerators which extends [fpgaconvnet](https://icidsl.github.io/fpgaconvnet-website/) in order to target the YOLO family of models for object detection.
fpgaConvNet is a toolflow for automating the mapping of DNN models onto FPGA devices, by taking into account performance targets and platform constraints.
By adding additional support for YOLO specific layers as well as addressing on-chip memory constraints by using a _software-based_ FIFO, we are able to achieve state-of-the-art latency and performance for YOLO model acceleration, with lower latency than GPU devices.
This repo contains the code artefacts from the project, with submodules indicating the changes made to the fpgaConvNet framework.

The structure of this repository is as follows:

- __fpgaconvnet-model:__ This repository contains code for modelling performance and resources for the FPGA accelerator.
- __fpgaconvnet-optimiser:__ Code for optimising the accelerator for the lowest latency given platform constraints.
- __platforms:__ FPGA platform description with resource constraints and memory bandwidth limits.
- __runs:__ Scripts for generating results using the modelling and optimiser repositories.
- __plot_utils:__ Collated results and scripts for re-generating figures from the paper.

To reproduce the results in the paper, simply run the following commands:
```
bash setup.sh
cd runs
bash run_all.sh
```
The generated results can be found in `report.json` files inside the `runs` folder.

Further to this repository, we have also made available a design for a YOLOv5n accelerator on a ZCU104 board, which came first on the PhD category of AMD's OpenHW competition, which can be found here:
https://github.com/ptoupas/amd-open-hardware-23
