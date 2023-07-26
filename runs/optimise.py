import os
import sys
import toml
import json

from fpgaconvnet.parser.Parser import Parser
from fpgaconvnet.tools.layer_enum import from_cfg_type
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

from fpgaconvnet.optimiser.solvers import Improve
from fpgaconvnet.optimiser.solvers import SimulatedAnnealing
from fpgaconvnet.optimiser.solvers import GreedyPartition

import fpgaconvnet.optimiser.transforms.partition
import fpgaconvnet.optimiser.transforms.coarse
import fpgaconvnet.optimiser.transforms.fine

model_path=sys.argv[1]
platform_path=sys.argv[2]
output_path=sys.argv[3]

# load platform configuration
with open(platform_path, "r") as f:
    platform_config = toml.load(f)

# parse the network
fpgaconvnet_parser = Parser()

# create network
net = fpgaconvnet_parser.onnx_to_fpgaconvnet(
        model_path, platform_path)

#for partition in net.partitions:
#    fpgaconvnet.optimiser.transforms.fine.apply_complete_fine(partition)

# greedy optimiser
opt = GreedyPartition(net)

# set latency objective
opt.objective  = 0

# set only fine and coarse transforms
opt.transforms = ["fine", "coarse"]

# disable weights reloading
for partition in opt.net.partitions:
    partition.enable_wr = False

# apply max fine factor
for partition in net.partitions:
    fpgaconvnet.optimiser.transforms.fine.apply_complete_fine(partition)

# update network
opt.net.update_partitions()

# run optimiser
opt.run_solver()

# update all partitions
opt.net.update_partitions()
opt.merge_memory_bound_partitions()
opt.net.update_partitions()

## update buffer depths
for node in opt.net.partitions[0].graph.nodes:
    if opt.net.partitions[0].graph.nodes[node]["type"] \
            in [ LAYER_TYPE.EltWise, LAYER_TYPE.Concat ]:
        opt.net.partitions[0].update_multiport_buffer_depth(node)

# create report
opt.net.create_report(os.path.join(output_path,"report.json"))

# save all partitions
opt.net.save_all_partitions(os.path.join(output_path, "config.json"))

# create scheduler
opt.net.get_schedule_csv(os.path.join(output_path,"scheduler.csv"))


