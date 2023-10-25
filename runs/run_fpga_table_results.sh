#!/bin/bash

# YOLOv3 models
./run_model.sh -f=yolov3 -v=-tiny -s=416 -p=vcu110
./run_model.sh -f=yolov3 -v=-tiny -s=416 -p=vcu118

# YOLOv5 models
./run_model.sh -f=yolov5 -v=s -s=640 -p=vcu110
./run_model.sh -f=yolov5 -v=s -s=640 -p=vcu118

# YOLOv8 models
./run_model.sh -f=yolov8 -v=m -s=640 -p=vcu110
./run_model.sh -f=yolov8 -v=m -s=640 -p=vcu118

