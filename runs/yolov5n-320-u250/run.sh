#!/bin/bash

family=yolov5
variant=n
imgsz=320
platform=u250

identifier=${family}${variant}-${imgsz}-${platform}

## perform preprocessing
# python ../${family}${variant}-preprocess.py \
#     ../../onnx_models/${family}/${family}${variant}_imgsz${imgsz}_fp16.onnx \
#     ${identifier}-fpgaconvnet.onnx

## optimise the model
# python -m fpgaconvnet.optimiser -n ${identifier}-opt -m ${identifier}-fpgaconvnet.onnx \
#     -p ../../platforms/${platform}.toml --optimiser greedy_partition --objective latency \
#     --optimiser_config_path ../optimiser-config.toml -o opt/

## post process the config to make it suitable for chisel
python ../postprocess.py opt/config.json opt/config-chisel.json

## copy over to chisel
mkdir -p ../../fpgaconvnet-chisel/data/partitions/$identifier/
cp opt/config-chisel.json ../../fpgaconvnet-chisel/data/partitions/$identifier/partition_info.json
cp ${identifier}-fpgaconvnet_optimized.onnx ../../fpgaconvnet-chisel/data/partitions/$identifier/model.onnx
