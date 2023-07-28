import sys
import json

# paths for configs
input_config_path = sys.argv[1]
output_config_path = sys.argv[2]

# get the json paths
rsc_config_path=f"{output_config_path}/config-chisel-rsc.json"
sim_config_path=f"{output_config_path}/config-chisel-sim.json"

# load the input config
with open(input_config_path, "r") as f:
    config = json.load(f)

## correct input and output nodes
config["partition"][0]["input_nodes"] = [
        "images",
        "/model.6/cv2/act/Mul_output_0",
        "/model.4/cv2/act/Mul_output_0",
        "/model.9/cv2/act/Mul_output_0",
    ]
config["partition"][0]["output_nodes"] = [
        "/model.4/cv2/act/Mul_output_0",
        "/model.6/cv2/act/Mul_output_0",
        "/model.9/cv2/act/Mul_output_0",
        "/model.22/Concat_output_0",
        "/model.22/Concat_1_output_0",
        "/model.22/Concat_2_output_0",
    ]

# buffers to remove from the hardware graph
OFF_CHIP_BUFFERS=["Concat_73", "Concat_85", "Concat_111"]

# iterate over layers of the network
for i, layer in enumerate(config["partition"][0]["layers"]):

    # fix resize nodes to have correct scaling
    if layer["type"] == "RESIZE":
        config["partition"][0]["layers"][i]["parameters"]["scale"] = [2, 2, 1]

    # if use uram is set, set weights style
    if layer["type"] == "CONVOLUTION":
        if layer["parameters"].get("use_uram", False):
            config["partition"][0]["layers"][i]["parameters"]["weights_ram_style"] = "ultra"
        else:
            config["partition"][0]["layers"][i]["parameters"]["weights_ram_style"] = "block"

    # reduce buffer size for synthesis
    if layer["name"] in OFF_CHIP_BUFFERS:
        config["partition"][0]["layers"][i]["streams_in"][1]["buffer_depth"] = 2

# save the post-processed configuration
with open(rsc_config_path, "w") as f:
    json.dump(config, f)

# iterate over layers of the network
for i, layer in enumerate(config["partition"][0]["layers"]):

    # increase the depth of the longest paths
    if layer["type"] == "CONCAT":
        config["partition"][0]["layers"][i]["streams_in"][1]["buffer_depth"] = \
                4*config["partition"][0]["layers"][i]["streams_in"][1]["buffer_depth"]

    # remove concat connections to long buffer connections
    if layer["name"] in OFF_CHIP_BUFFERS:
        config["partition"][0]["layers"][i]["streams_in"][1]["buffer_depth"] = 64
        config["partition"][0]["layers"][i]["streams_in"][1]["node"] = config["partition"][0]["layers"][i]["name"]

    # remove split connections to long buffer connections
    if layer["type"] == "SPLIT":
        for j, stream_out in enumerate(config["partition"][0]["layers"][i]["streams_out"]):
            if stream_out["node"] in OFF_CHIP_BUFFERS:
                config["partition"][0]["layers"][i]["streams_out"][j]["node"] = layer["name"]

# save the post-processed configuration
with open(sim_config_path, "w") as f:
    json.dump(config, f)

