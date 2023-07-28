import sys
import json

partition_path = sys.argv[1]

with open(partition_path, "r") as f:
    config = json.load(f)

# iterate over layers of the network
for i, layer in enumerate(config["partition"][0]["layers"]):

    # if use uram is set, set weights style
    if layer["type"] == "CONVOLUTION":
        config["partition"][0]["layers"][i]["parameters"]["weghts_ram_style"] = "auto"

# save the post-processed configuration
with open(f"{partition_path}.auto.json", "w") as f:
    json.dump(config, f)

