import onnx
import json
import numpy as np
import onnx_graphsurgeon as gs

from fpgaconvnet.parser.Parser import Parser
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

# definition of the model name
model_name = "yolov8n"

# edit the onnx graph to remove the post-processing
graph = onnx.load(f"../onnx_models/{model_name}.onnx")

# load with graph surgeon
graph = gs.import_onnx(graph)

# get the extra operations to remove
max_idx = 0
for idx, node in enumerate(graph.nodes):
    if node.name == "/model.22/Reshape":
        reshape_l_idx = idx
    if node.name == "/model.22/Reshape_1":
        reshape_m_idx = idx
    if node.name == "/model.22/Reshape_2":
        reshape_r_idx = idx

# remove extra operations
del graph.nodes[reshape_r_idx:-1]
del graph.nodes[reshape_m_idx:]
del graph.nodes[reshape_l_idx:]

# get output layers
concat_l = next(filter(lambda x: x.name == "/model.22/Concat", graph.nodes))
concat_m = next(filter(lambda x: x.name == "/model.22/Concat_1", graph.nodes))
concat_r = next(filter(lambda x: x.name == "/model.22/Concat_2", graph.nodes))

# get the resize layers
resize = next(filter(lambda x: x.name == "/model.10/Resize", graph.nodes))
resize.inputs[1] = gs.Constant("roi_0", np.array([0.0,0.0,0.0,0.0]))
resize = next(filter(lambda x: x.name == "/model.13/Resize", graph.nodes))
resize.inputs[1] = gs.Constant("roi_1", np.array([0.0,0.0,0.0,0.0]))

# create the output nodes
output_l = gs.Variable("/model.22/Concat_output_0",   shape=[1, 144, 40, 40], dtype="float32")
output_m = gs.Variable("/model.22/Concat_1_output_0", shape=[1, 144, 20, 20], dtype="float32")
output_r = gs.Variable("/model.22/Concat_2_output_0", shape=[1, 144, 10, 10], dtype="float32")

# connect the output nodes
concat_l.outputs = [ output_l ]
concat_m.outputs = [ output_m ]
concat_r.outputs = [ output_r ]

# update the graph outputs
graph.outputs = [ concat_l.outputs[0], concat_m.outputs[0], concat_r.outputs[0] ]

# cleanup graph
graph.cleanup()

# save the reduced network
graph = gs.export_onnx(graph)
graph.ir_version = 8 # need to downgrade the ir version
onnx.save(graph, f"../onnx_models/{model_name}-fpgaconvnet.onnx")

# import taso

# old_model = taso.load_onnx(f"../onnx_models/{model_name}-fpgaconvnet.onnx")
# taso_graph = taso.optimize(old_model)
# new_model = taso.export_onnx(taso_graph)
# onnx.save(new_model, f"../onnx_models/{model_name}-taso-fpgaconvnet.onnx")

# create a parser
parser = Parser(backend="chisel", quant_mode="auto", convert_gemm_to_conv=False, custom_onnx=False)

# parse the network and perform all optimisations
net = parser.onnx_to_fpgaconvnet(f"../onnx_models/{model_name}-fpgaconvnet.onnx",
        "../platforms/zcu104.toml", False, save_opt_model=True)
net.update_partitions()


