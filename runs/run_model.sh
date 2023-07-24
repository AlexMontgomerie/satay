#!/bin/bash

# example usage:
#   ./run_model.sh -f=yolov5 -v=n -s=320 -p=u250

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
OUTPUT_PATH=${ID}
CHISEL_PATH=../fpgaconvnet-chisel/data/partitions/$ID/

## make the output directory
mkdir -p $OUTPUT_PATH

## perform preprocessing
# python ${FAMILY}${VARIANT}-preprocess.py $MODEL_PATH $FPGACONVNET_MODEL_PATH
python ${FAMILY}-preprocess.py $MODEL_PATH $FPGACONVNET_MODEL_PATH $SIZE

## optimise the model
python optimise.py $FPGACONVNET_MODEL_PATH $PLATFORM_PATH $OUTPUT_PATH

## post process the config to make it suitable for chisel
python ${FAMILY}-postprocess.py $OUTPUT_PATH/config.json $OUTPUT_PATH

## copy over to chisel
mkdir -p ${CHISEL_PATH}-rsc
cp $OUTPUT_PATH/config-chisel-rsc.json ${CHISEL_PATH}-rsc/partition_info.json
cp $FPGACONVNET_MODEL_PATH ${CHISEL_PATH}-rsc/model.onnx
mkdir -p ${CHISEL_PATH}-sim
cp $OUTPUT_PATH/config-chisel-sim.json ${CHISEL_PATH}-sim/partition_info.json
cp $FPGACONVNET_MODEL_PATH ${CHISEL_PATH}-sim/model.onnx


