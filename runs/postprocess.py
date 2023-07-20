import sys
import json

# paths for configs
input_config_path = sys.argv[1]
output_config_path = sys.argv[2]

# load the input config
with open(input_config_path, "r") as f:
    config = json.load(f)

# iterate over layers of the network
for i, layer in enumerate(config["partition"][0]["layers"]):

    # fix resize nodes to have correct scaling
    if layer["type"] == "RESIZE":
        config["partition"][0]["layers"][i]["parameters"]["scale"] = [2, 2, 1]

    # if use uram is set, set weights style
    if layer.get("use_uram", False):
        config["partition"][0]["layers"][i]["parameters"]["weghts_ram_style"] = "ultra"

# save the post-processed configuration
with open(output_config_path, "w") as f:
    json.dump(config, f)
