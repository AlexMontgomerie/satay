#!/bin/bash

./run_model.sh -f=yolov3 -v=-tiny -s=416 -p=u250
./run_model.sh -f=yolov5 -v=n -s=640 -p=u250
./run_model.sh -f=yolov5 -v=s -s=640 -p=u250
./run_model.sh -f=yolov5 -v=m -s=640 -p=u250
./run_model.sh -f=yolov8 -v=n -s=640 -p=u250
./run_model.sh -f=yolov8 -v=s -s=640 -p=u250
./run_model.sh -f=yolov8 -v=m -s=640 -p=u250
