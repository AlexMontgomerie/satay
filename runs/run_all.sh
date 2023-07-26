#!/bin/bash

# YOLOv3

## U250
./run_model.sh -f=yolov3 -v=-tiny -s=320 -p=u250
./run_model.sh -f=yolov3 -v=-tiny -s=640 -p=u250

## VCU118
./run_model.sh -f=yolov3 -v=-tiny -s=320 -p=vcu118
./run_model.sh -f=yolov3 -v=-tiny -s=640 -p=vcu118

## VCU110
./run_model.sh -f=yolov3 -v=-tiny -s=320 -p=vcu110
./run_model.sh -f=yolov3 -v=-tiny -s=640 -p=vcu110


# YOLOv5n

## U250
./run_model.sh -f=yolov5 -v=n -s=320 -p=u250
./run_model.sh -f=yolov5 -v=n -s=640 -p=u250
./run_model.sh -f=yolov5L -v=n -s=1280 -p=u250

## ZCU104
./run_model.sh -f=yolov5 -v=n -s=320 -p=zcu104
./run_model.sh -f=yolov5 -v=n -s=640 -p=zcu104
./run_model.sh -f=yolov5L -v=n -s=1280 -p=zcu104

## ZCU216
./run_model.sh -f=yolov5 -v=n -s=320  -p=zcu216
./run_model.sh -f=yolov5 -v=n -s=640  -p=zcu216
./run_model.sh -f=yolov5L -v=n -s=1280 -p=zcu216

## VCU118
./run_model.sh -f=yolov5 -v=n -s=320  -p=vcu118
./run_model.sh -f=yolov5 -v=n -s=640  -p=vcu118
./run_model.sh -f=yolov5L -v=n -s=1280 -p=vcu118

## VCU110
./run_model.sh -f=yolov5 -v=n -s=320  -p=vcu110
./run_model.sh -f=yolov5 -v=n -s=640  -p=vcu110
./run_model.sh -f=yolov5L -v=n -s=1280 -p=vcu110


# YOLOv5s

## U250
./run_model.sh -f=yolov5 -v=s -s=320  -p=u250
./run_model.sh -f=yolov5 -v=s -s=640  -p=u250
./run_model.sh -f=yolov5L -v=s -s=1280 -p=u250

## ZCU216
./run_model.sh -f=yolov5 -v=s -s=320  -p=zcu216
./run_model.sh -f=yolov5 -v=s -s=640  -p=zcu216
./run_model.sh -f=yolov5L -v=s -s=1280 -p=zcu216

## VCU118
./run_model.sh -f=yolov5 -v=s -s=320  -p=vcu118
./run_model.sh -f=yolov5 -v=s -s=640  -p=vcu118
./run_model.sh -f=yolov5L -v=s -s=1280 -p=vcu118

## VCU110
./run_model.sh -f=yolov5 -v=s -s=320  -p=vcu110
./run_model.sh -f=yolov5 -v=s -s=640  -p=vcu110
./run_model.sh -f=yolov5L -v=s -s=1280 -p=vcu110


# YOLOv5m

## U250
./run_model.sh -f=yolov5 -v=m -s=320 -p=u250
./run_model.sh -f=yolov5 -v=m -s=640 -p=u250
./run_model.sh -f=yolov5L -v=m -s=1280 -p=u250

## VCU118
./run_model.sh -f=yolov5 -v=m -s=320 -p=vcu118
./run_model.sh -f=yolov5 -v=m -s=640 -p=vcu118
./run_model.sh -f=yolov5L -v=m -s=1280 -p=vcu118


# YOLOv8n

## U250
./run_model.sh -f=yolov8 -v=n -s=320 -p=u250
./run_model.sh -f=yolov8 -v=n -s=640 -p=u250

## ZCU104
./run_model.sh -f=yolov8 -v=n -s=320 -p=zcu104
./run_model.sh -f=yolov8 -v=n -s=640 -p=zcu104

## ZCU216
./run_model.sh -f=yolov8 -v=n -s=320 -p=zcu216
./run_model.sh -f=yolov8 -v=n -s=640 -p=zcu216

## VCU118
./run_model.sh -f=yolov8 -v=n -s=320 -p=vcu118
./run_model.sh -f=yolov8 -v=n -s=640 -p=vcu118

## VCU110
./run_model.sh -f=yolov8 -v=n -s=320 -p=vcu110
./run_model.sh -f=yolov8 -v=n -s=640 -p=vcu110


# YOLOv8s

## U250
./run_model.sh -f=yolov8 -v=s -s=320 -p=u250
./run_model.sh -f=yolov8 -v=s -s=640 -p=u250

## VCU118
./run_model.sh -f=yolov8 -v=s -s=320 -p=vcu118
./run_model.sh -f=yolov8 -v=s -s=640 -p=vcu118

## VCU110
./run_model.sh -f=yolov8 -v=s -s=320 -p=vcu110
./run_model.sh -f=yolov8 -v=s -s=640 -p=vcu110


# YOLOv8m

## U250
./run_model.sh -f=yolov8 -v=m -s=320 -p=u250
./run_model.sh -f=yolov8 -v=m -s=640 -p=u250

## VCU118
./run_model.sh -f=yolov8 -v=m -s=320 -p=vcu118
./run_model.sh -f=yolov8 -v=m -s=640 -p=vcu118

