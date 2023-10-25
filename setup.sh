#!/bin/bash
git submodule update --init --recursive
conda create -n satay python=3.10
conda activate satay
cd fpgaconvnet-model
python -m pip install .
cd ../fpgaconvnet-optimiser
python -m pip install .
pip install nvidia-pyindex
pip install onnx-graphsurgeon
cd ../onnx_models
./get_yolo_models.sh
cd ..
