import sys
import onnx
import json
import numpy as np
import onnx_graphsurgeon as gs

from fpgaconvnet.parser.Parser import Parser
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

# definition of the model name
input_model_path = sys.argv[1]
output_model_path = sys.argv[2]
input_shape = int(sys.argv[3])

# edit the onnx graph to remove the post-processing
graph = onnx.load(input_model_path)

# load with graph surgeon
graph = gs.import_onnx(graph)

# get the extra operations to remove
max_idx = 0
for idx, node in enumerate(graph.nodes):
    if node.name == "/model.33/Reshape":
        reshape_l_0_0_idx = idx
    if node.name == "/model.33/Reshape_1":
        reshape_l_0_1_idx = idx
    if node.name == "/model.33/Reshape_2":
        reshape_l_1_0_idx = idx
    if node.name == "/model.33/Reshape_3":
        reshape_l_1_1_idx = idx
    if node.name == "/model.33/Reshape_4":
        reshape_r_0_0_idx = idx
    if node.name == "/model.33/Reshape_5":
        reshape_r_0_1_idx = idx
    if node.name == "/model.33/Reshape_6":
        reshape_r_1_0_idx = idx
    if node.name == "/model.33/Reshape_7":
        reshape_r_1_1_idx = idx

# remove extra operations
del graph.nodes[reshape_r_1_0_idx:reshape_r_1_1_idx+2]
del graph.nodes[reshape_r_0_0_idx:reshape_r_0_1_idx+1]
del graph.nodes[reshape_l_1_0_idx:reshape_l_1_1_idx+1]
del graph.nodes[reshape_l_0_0_idx:reshape_l_0_1_idx+1]

# get output layers
conv_l_1 = next(filter(lambda x: x.name == "/model.33/m.3/Conv", graph.nodes))
conv_l_0 = next(filter(lambda x: x.name == "/model.33/m.2/Conv", graph.nodes))
conv_r_1 = next(filter(lambda x: x.name == "/model.33/m.1/Conv", graph.nodes))
conv_r_0 = next(filter(lambda x: x.name == "/model.33/m.0/Conv", graph.nodes))

# get the resize layers
resize = next(filter(lambda x: x.name == "/model.13/Resize", graph.nodes))
resize.inputs[1] = gs.Constant("roi_0", np.array([0.0,0.0,0.0,0.0]))
resize = next(filter(lambda x: x.name == "/model.17/Resize", graph.nodes))
resize.inputs[1] = gs.Constant("roi_1", np.array([0.0,0.0,0.0,0.0]))
resize = next(filter(lambda x: x.name == "/model.21/Resize", graph.nodes))
resize.inputs[1] = gs.Constant("roi_2", np.array([0.0,0.0,0.0,0.0]))

# pad 255 to 256
conv_l_1_channels = conv_l_1.inputs[1].values.shape[1]
conv_l_1.inputs[1] = gs.Constant("model.33.m.3.weight", np.concatenate((conv_l_1.inputs[1].values, np.zeros((1, conv_l_1_channels, 1, 1), dtype=np.float16))))
conv_l_1.inputs[2] = gs.Constant("model.33.m.3.bias", np.concatenate((conv_l_1.inputs[2].values, np.zeros((1), dtype=np.float16))))

conv_l_0_channels = conv_l_0.inputs[1].values.shape[1]
conv_l_0.inputs[1] = gs.Constant("model.33.m.2.weight", np.concatenate((conv_l_0.inputs[1].values, np.zeros((1, conv_l_0_channels, 1, 1), dtype=np.float16))))
conv_l_0.inputs[2] = gs.Constant("model.33.m.2.bias", np.concatenate((conv_l_0.inputs[2].values, np.zeros((1), dtype=np.float16))))

conv_r_1_channels = conv_r_1.inputs[1].values.shape[1]
conv_r_1.inputs[1] = gs.Constant("model.33.m.1.weight", np.concatenate((conv_r_1.inputs[1].values, np.zeros((1, conv_r_1_channels, 1, 1), dtype=np.float16))))
conv_r_1.inputs[2] = gs.Constant("model.33.m.1.bias", np.concatenate((conv_r_1.inputs[2].values, np.zeros((1), dtype=np.float16))))

conv_r_0_channels = conv_r_0.inputs[1].values.shape[1]
conv_r_0.inputs[1] = gs.Constant("model.33.m.0.weight", np.concatenate((conv_r_0.inputs[1].values, np.zeros((1, conv_r_0_channels, 1, 1), dtype=np.float16))))
conv_r_0.inputs[2] = gs.Constant("model.33.m.0.bias", np.concatenate((conv_r_0.inputs[2].values, np.zeros((1), dtype=np.float16))))

# create the output nodes
output_l_1 = gs.Variable("/model.33/m.3/Conv_output_0", shape=[1, 256, input_shape//64, input_shape//64], dtype="float16")
output_l_0 = gs.Variable("/model.33/m.2/Conv_output_0", shape=[1, 256, input_shape//32, input_shape//32], dtype="float16")
output_r_1 = gs.Variable("/model.33/m.1/Conv_output_0", shape=[1, 256, input_shape//16, input_shape//16], dtype="float16")
output_r_0 = gs.Variable("/model.33/m.0/Conv_output_0", shape=[1, 256, input_shape//8,  input_shape//8], dtype="float16")

# connect the output nodes
conv_l_1.outputs = [ output_l_1 ]
conv_l_0.outputs = [ output_l_0 ]
conv_r_1.outputs = [ output_r_1 ]
conv_r_0.outputs = [ output_r_0 ]

# update the graph outputs
graph.outputs = [
        conv_l_1.outputs[0], conv_l_0.outputs[0],
        conv_r_1.outputs[0], conv_r_0.outputs[0], ]

# cleanup graph
graph.cleanup()

# save the reduced network
graph = gs.export_onnx(graph)
graph.ir_version = 8 # need to downgrade the ir version
onnx.save(graph, output_model_path)
