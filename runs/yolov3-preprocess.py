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
    if node.name == "/model.20/Reshape":
        reshape_l_0_idx = idx
    if node.name == "/model.20/Reshape_1":
        reshape_l_1_idx = idx
    if node.name == "/model.20/Reshape_2":
        reshape_r_0_idx = idx
    if node.name == "/model.20/Reshape_3":
        reshape_r_1_idx = idx

# remove extra operations
del graph.nodes[reshape_r_0_idx:reshape_r_1_idx+2]
del graph.nodes[reshape_l_0_idx:reshape_l_1_idx+1]

# get output layers
conv_l = next(filter(lambda x: x.name == "/model.20/m.1/Conv", graph.nodes))
conv_r = next(filter(lambda x: x.name == "/model.20/m.0/Conv", graph.nodes))

# get the resize layers
resize = next(filter(lambda x: x.name == "/model.17/Resize", graph.nodes))
resize.inputs[1] = gs.Constant("roi_0", np.array([0.0,0.0,0.0,0.0]))

# # fix the padding layer
# pad = next(filter(lambda x: x.name == "/model.11/Pad", graph.nodes))
# pad.inputs[1] = gs.Constant("pad", np.array([0,0,0,0,1,1,0,0]))

# pad 255 to 256
conv_l_channels = conv_l.inputs[1].values.shape[1]
conv_l.inputs[1] = gs.Constant("model.20.m.1.weight", np.concatenate((conv_l.inputs[1].values, np.zeros((1, conv_l_channels, 1, 1), dtype=np.float16))))
conv_l.inputs[2] = gs.Constant("model.20.m.1.bias", np.concatenate((conv_l.inputs[2].values, np.zeros((1), dtype=np.float16))))

conv_r_channels = conv_r.inputs[1].values.shape[1]
conv_r.inputs[1] = gs.Constant("model.20.m.0.weight", np.concatenate((conv_r.inputs[1].values, np.zeros((1, conv_r_channels, 1, 1), dtype=np.float16))))
conv_r.inputs[2] = gs.Constant("model.20.m.0.bias", np.concatenate((conv_r.inputs[2].values, np.zeros((1), dtype=np.float16))))

# create the output nodes
output_l = gs.Variable("/model.20/m.1/Conv_output_0", shape=[1, 256, input_shape//32, input_shape//32], dtype="float16")
output_r = gs.Variable("/model.20/m.0/Conv_output_0", shape=[1, 256, input_shape//16, input_shape//16], dtype="float16")

# connect the output nodes
conv_l.outputs = [ output_l ]
conv_r.outputs = [ output_r ]

# update the graph outputs
graph.outputs = [ conv_l.outputs[0], conv_r.outputs[0] ]

# cleanup graph
graph.cleanup()

# save the reduced network
graph = gs.export_onnx(graph)
graph.ir_version = 8 # need to downgrade the ir version
onnx.save(graph, output_model_path)
