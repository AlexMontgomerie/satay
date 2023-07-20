#!/bin/bash

## perform preprocessing
python ../yolov5n-preprocess.py ../../onnx_models/yolov5/yolov5n_imgsz320_fp16.onnx yolov5n-320-fpgaconvnet.onnx

## optimise the model
python -m fpgaconvnet.optimiser -n yolov5n-320-opt -m yolov5n-320-fpgaconvnet.onnx \
    -p ../../platforms/zcu104.toml --optimiser greedy_partition --objective latency \
    --optimiser_config_path ../optimiser-config.toml -o opt/

## post process the config to make it suitable for chisel

## copy over to chisel

