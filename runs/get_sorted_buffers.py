import json
import sys
import pandas as pd

# get the configuration path
config_path = sys.argv[1]

# open the config
with open(config_path, "r") as f:
    conf = json.load(f)

# get all the buffer depths
buffer_depths = []

# iterate over nodes in the partition
for node in conf["partition"][0]["layers"]:

    # only look at concat and eltwise nodes
    if node["type"] not in ["CONCAT", "ELTWISE"]:
        continue

    # get all the buffers in
    for s in node["streams_in"]:

        # get key and depth
        path = s["node"].rstrip("_split") + " - " + node["name"]
        depth = s["buffer_depth"]

        # add to all buffer depths
        buffer_depths.append([path, depth])

# sort buffer depths
buffer_depths = list(reversed(sorted(buffer_depths, key=lambda x : x[1])))

# save to CSV
df = pd.DataFrame(buffer_depths)
df.to_csv(config_path+".csv", index=False, header=False)

# print some statistics
print(f"Total Buffer Size (MB): {sum([x[1] for x in buffer_depths[3:]])*2/1000000}")

