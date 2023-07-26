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

# add the correct outputs
config["partition"][0]["output_nodes"] = [
        "/model.24/m.0/Conv_output_0",
        "/model.24/m.1/Conv_output_0",
        "/model.24/m.2/Conv_output_0",
]

# iterate over layers of the network
for i, layer in enumerate(config["partition"][0]["layers"]):

    # fix resize nodes to have correct scaling
    if layer["type"] == "RESIZE":
        config["partition"][0]["layers"][i]["parameters"]["scale"] = [2, 2, 1]

    # if use uram is set, set weights style
    if layer.get("use_uram", False):
        config["partition"][0]["layers"][i]["parameters"]["weghts_ram_style"] = "ultra"

# save the post-processed configuration
with open(rsc_config_path, "w") as f:
    json.dump(config, f)

# iterate over layers of the network
for i, layer in enumerate(config["partition"][0]["layers"]):

    # increase the depth of the longest paths
    # if layer["name"] in ["Concat_84", "Concat_99", "Concat_127"] :
    if layer["type"] == "CONCAT":
        config["partition"][0]["layers"][i]["streams_in"][1]["buffer_depth"] = 4*config["partition"][0]["layers"][i]["streams_in"][1]["buffer_depth"]

# save the post-processed configuration
with open(sim_config_path, "w") as f:
    json.dump(config, f)

