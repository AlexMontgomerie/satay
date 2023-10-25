#!/bin/bash

# YOLOv3

## VCU118
./run_model.sh -f=yolov3 -v=-tiny -s=320 -p=vcu118
./run_model.sh -f=yolov3 -v=-tiny -s=640 -p=vcu118

## VCU110
./run_model.sh -f=yolov3 -v=-tiny -s=320 -p=vcu110
./run_model.sh -f=yolov3 -v=-tiny -s=640 -p=vcu110


# YOLOv5n

## VCU118
./run_model.sh -f=yolov5 -v=n -s=320  -p=vcu118
./run_model.sh -f=yolov5 -v=n -s=640  -p=vcu118

## VCU110
./run_model.sh -f=yolov5 -v=n -s=320  -p=vcu110
./run_model.sh -f=yolov5 -v=n -s=640  -p=vcu110


# YOLOv5m

## VCU118
./run_model.sh -f=yolov5 -v=m -s=320 -p=vcu118
./run_model.sh -f=yolov5 -v=m -s=640 -p=vcu118


# YOLOv8n

## VCU118
./run_model.sh -f=yolov8 -v=n -s=320 -p=vcu118
./run_model.sh -f=yolov8 -v=n -s=640 -p=vcu118

## VCU110
./run_model.sh -f=yolov8 -v=n -s=320 -p=vcu110
./run_model.sh -f=yolov8 -v=n -s=640 -p=vcu110


# YOLOv8s

## VCU118
./run_model.sh -f=yolov8 -v=s -s=320 -p=vcu118
./run_model.sh -f=yolov8 -v=s -s=640 -p=vcu118

## VCU110
./run_model.sh -f=yolov8 -v=s -s=320 -p=vcu110
./run_model.sh -f=yolov8 -v=s -s=640 -p=vcu110


# YOLOv8m

## VCU118
./run_model.sh -f=yolov8 -v=m -s=320 -p=vcu118
./run_model.sh -f=yolov8 -v=m -s=640 -p=vcu118

