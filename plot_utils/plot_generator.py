import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
import seaborn as sns

FIG_SIZE_X = 12
FIG_SIZE_Y = 8

FONT_SIZE_TICKS = 15
FONT_SIZE_LABELS = 25
FONT_SIZE_TITLE = 30
FONT_SIZE_LEGEND = 12

COLOUR_PALETTE = 'deep'

plt.style.use(["science", "grid"])
plt.rcParams["figure.figsize"] = (FIG_SIZE_X, FIG_SIZE_Y)

cpu_gpu_df = pd.read_csv('Results_CPU_and_GPU.csv')
cpu_gpu_cols = cpu_gpu_df.columns.to_list()
print(cpu_gpu_cols)

# Create a combined column for Model and Input Shape named Model-Input Shape
cpu_gpu_df['Model Family-Input Shape'] = cpu_gpu_df['Model Family'] + '-' + cpu_gpu_df['Input Shape']

# filter based on Platform
cpu_gpu_df = cpu_gpu_df[cpu_gpu_df['Platform'].isin(['ARM Cortex-A72', 'Jetson TX2'])]
cpu_gpu_arm_df = cpu_gpu_df[cpu_gpu_df['Platform'].isin(['ARM Cortex-A72'])]
cpu_gpu_tx2_df = cpu_gpu_df[cpu_gpu_df['Platform'].isin(['Jetson TX2'])]
# TODO: Add here the filtering for the FPGA results following the same pattern as above


fig, ax = plt.subplots()
sns.lineplot(data=cpu_gpu_arm_df, x='Inf. Time (ms)', y='mAP50-95 @ coco128', ax=ax, hue='Model Family-Input Shape', style='Platform', markers=['o', 's'],  dashes=False, linewidth=2, markersize=10, palette=COLOUR_PALETTE, linestyle='solid')
sns.lineplot(data=cpu_gpu_tx2_df, x='Inf. Time (ms)', y='mAP50-95 @ coco128', ax=ax, hue='Model Family-Input Shape', style='Platform', markers=['s', 'o'],  dashes=False, linewidth=2, markersize=10, palette=COLOUR_PALETTE, linestyle='dashed')
# TODO: Add here the plotting for the FPGA results following the same pattern as above


# Add some text for each point in the line
for line in range(0,cpu_gpu_df.shape[0]):
    curr_series = cpu_gpu_df.iloc[line]
    ax.text(curr_series['Inf. Time (ms)'], curr_series['mAP50-95 @ coco128'], curr_series['Model'], ha='center', va='bottom', size='large', color='black', weight='bold')

ax.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABELS)
ax.set_ylabel('mAP50-95', fontsize=FONT_SIZE_LABELS)
ax.set_xscale('log')
handles, labels = ax.get_legend_handles_labels()
handles = handles[1:7] + [handles[8]] + [handles[-1]]
labels = labels[1:7] + [labels[8]] + [labels[-1]]
ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=FONT_SIZE_LEGEND, frameon=False)


ax.set_xticklabels(ax.get_xticklabels(), fontsize=FONT_SIZE_TICKS)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=FONT_SIZE_TICKS)

plt.tight_layout()
# plt.savefig('yolo_comparison_devices.png', bbox_inches='tight')
plt.savefig('yolo_comparison_devices.pdf', bbox_inches='tight', format='pdf')