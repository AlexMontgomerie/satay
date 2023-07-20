import os

import onnx_tool
import toml

# Load config
models_config = toml.load('yolo_models_cfg.toml')

for model_family, variants in models_config.items():
    for variant in variants:
        yolo_model_name = variant['name']
        subgraph_in_tensor_names = variant['head_in_tensor_names']
        subgraph_out_tensor_names = variant['head_out_tensor_names']

        onnx_model_path = os.path.join(model_family, yolo_model_name + '.onnx')

        if not os.path.exists(os.path.join(model_family, yolo_model_name)):
            os.makedirs(os.path.join(model_family, yolo_model_name))

        onnx_tool.model_subgraph(onnx_model_path, in_tensor_names=subgraph_in_tensor_names, out_tensor_names=subgraph_out_tensor_names, savefolder=os.path.join(model_family, yolo_model_name))
