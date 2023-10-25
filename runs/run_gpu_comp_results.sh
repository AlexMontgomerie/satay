#!/bin/bash

./run_model.sh -f=yolov5 -v=n -s=320 -p=zcu104
./run_model.sh -f=yolov5 -v=n -s=320 -p=zcu216
./run_model.sh -f=yolov5 -v=n -s=320 -p=vcu110
./run_model.sh -f=yolov5 -v=n -s=320 -p=vcu118
./run_model.sh -f=yolov5 -v=n -s=320 -p=u250

./run_model.sh -f=yolov5 -v=n -s=640 -p=zcu104
./run_model.sh -f=yolov5 -v=n -s=640 -p=zcu216
./run_model.sh -f=yolov5 -v=n -s=640 -p=vcu110
./run_model.sh -f=yolov5 -v=n -s=640 -p=vcu118
./run_model.sh -f=yolov5 -v=n -s=640 -p=u250
