import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
import seaborn as sns

FIG_SIZE_X = 10
FIG_SIZE_Y = 6

FONT_SIZE_TICKS = 18
FONT_SIZE_LABELS = 28
FONT_SIZE_TITLE = 30
FONT_SIZE_LEGEND = 13
FONT_SIZE_TEXT = 15

COLOUR_PALETTE = 'deep'

plt.style.use(["science", "grid"])
plt.rcParams["figure.figsize"] = (FIG_SIZE_X, FIG_SIZE_Y)

yolo_comparison_version = 2

if yolo_comparison_version == 1:
    # Version 1: Plotting the results for the different models on different devices
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
        ax.text(curr_series['Inf. Time (ms)'], curr_series['mAP50-95 @ coco128'], curr_series['Model'], ha='center', va='bottom', size=FONT_SIZE_TEXT, color='black', weight='bold')

    ax.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel('mAP50-95', fontsize=FONT_SIZE_LABELS)
    ax.set_xscale('log')
    handles, labels = ax.get_legend_handles_labels()
    handles = handles[1:7] + [handles[8]] + [handles[-1]]
    labels = labels[1:7] + [labels[8]] + [labels[-1]]
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3, fontsize=FONT_SIZE_LEGEND, frameon=False)

elif yolo_comparison_version == 2:
    # Version 2: Plotting the results for the different models on different input sizes
    cpu_gpu_df = pd.read_csv('Results_CPU_and_GPU.csv')
    cpu_gpu_cols = cpu_gpu_df.columns.to_list()
    print(cpu_gpu_cols)

    # filter based on Input Shape
    cpu_gpu_df = cpu_gpu_df[cpu_gpu_df['Input Shape'].isin(['640x640'])]

    # filter based on Platform
    cpu_gpu_df = cpu_gpu_df[cpu_gpu_df['Platform'].isin(['ARM Cortex-A72', 'Jetson TX2'])]
    # TODO: Add in the list above the FPGA filtering

    fig, ax = plt.subplots()
    sns.lineplot(data=cpu_gpu_df, x='Inf. Time (ms)', y='mAP50-95 @ coco128', ax=ax, hue='Platform', style='Model Family', markers=['o', 's', 'p'],  dashes=True, linewidth=2, markersize=10, palette=COLOUR_PALETTE, linestyle='solid')

    # Add some text for each point in the line
    for line in range(0,cpu_gpu_df.shape[0]):
        curr_series = cpu_gpu_df.iloc[line]
        ax.text(curr_series['Inf. Time (ms)'], curr_series['mAP50-95 @ coco128'], curr_series['Model'], ha='center', va='bottom', size=FONT_SIZE_TEXT, color='black', weight='bold')

    ax.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel('mAP50-95', fontsize=FONT_SIZE_LABELS)
    ax.set_xscale('log')
    handles, labels = ax.get_legend_handles_labels()
    handles = handles[1:3] + handles[4:]
    labels = labels[1:3] + labels[4:]
    order = [0, 2, 1, 3, 4]
    handles = [handles[idx] for idx in order]
    labels = [labels[idx] for idx in order]
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.155), ncol=3, fontsize=FONT_SIZE_LEGEND, frameon=False)


ax.set_xticklabels(ax.get_xticklabels(), fontsize=FONT_SIZE_TICKS)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=FONT_SIZE_TICKS)

plt.tight_layout()
# plt.savefig(f'yolo_comparison_devices_v{yolo_comparison_version}.png', bbox_inches='tight')
plt.savefig(f'yolo_comparison_devices.pdf', bbox_inches='tight', format='pdf')