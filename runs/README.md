# Runs

This folder contains all the scripts for running and reproducing the results for this paper.
The main script which automates the result collection is `run_model.sh`, which can be used as follows:

```
./run_model.sh -f={yolo model family} -v={version} -s={spatial size} -p={FPGA}
```

For example, `./run_model.sh -f=yolov5 -v=n -s=320 -p=zcu104` will automate the design of a YOLOv5n accelerator with a spatial size of 320x320, for a ZCU104 FPGA device.
This script the _preprocess_ method corresponding to the YOLO family, which partitions the ONNX model into components which are accelerated on the FPGA.
It then uses the `optimise.py` script which uses the pre-processed model and FPGA platform constraints to find an optimal hardware configuration.
Finally, the _postprocess_ method is used to make the ONNX model ready for hardware generation.
This script is used in the other scripts to collect certain results.
