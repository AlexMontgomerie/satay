#!/bin/bash

# example usage:
#   ./run_model.sh -f=yolov5 -v=n -s=320 -p=u25

# argument parser
for i in "$@"; do
  case $i in
    -f=*|--family=*)
      FAMILY="${i#*=}"
      shift # past argument=value
      ;;
    -v=*|--variant=*)
      VARIANT="${i#*=}"
      shift # past argument=value
      ;;
    -s=*|--size=*)
      SIZE="${i#*=}"
      shift # past argument=value
      ;;
    -p=*|--platform=*)
      PLATFORM="${i#*=}"
      shift # past argument=value
      ;;
    -*|--*)
      echo "Unknown option $i"
      exit 1
      ;;
    *)
      ;;
  esac
done

# setup strings and paths
ID=${FAMILY}${VARIANT}-${SIZE}-${PLATFORM}
MODEL_NAME=${FAMILY}${VARIANT}_imgsz${SIZE}_fp16
MODEL_PATH=../onnx_models/${FAMILY}/${MODEL_NAME}.onnx
FPGACONVNET_MODEL_PATH=../onnx_models/${FAMILY}/${MODEL_NAME}-fpgaconvnet.onnx
PLATFORM_PATH=../platforms/${PLATFORM}.toml
OUTPUT_PATH=${ID}/
CHISEL_PATH=../fpgaconvnet-chisel/data/partitions/$ID/
## perform preprocessing
python ${FAMILY}${VARIANT}-preprocess.py $MODEL_PATH $FPGACONVNET_MODEL_PATH

## optimise the model
python optimise.py $FPGACONVNET_MODEL_PATH $PLATFORM_PATH $OUTPUT_PATH

## post process the config to make it suitable for chisel
python postprocess.py $OUTPUT_PATH/config.json $OUTPUT_PATH/config-chisel.json

## copy over to chisel
mkdir -p $CHISEL_PATH
cp $OUTPUT_PATH/config-chisel.json $CHISEL_PATH/partition_info.json
cp $FPGACONVNET_MODEL_PATH $CHISEL_PATH/model.onnx
