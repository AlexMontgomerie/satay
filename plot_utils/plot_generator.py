import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
import seaborn as sns
import numpy as np

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

# plot generation can be "yolo_comparison" or "quantization_comparison" or "all"
plot_generation = "ablation"
assert plot_generation in ["ablation", "yolo_comparison", "quantization_comparison", "buffer_depths", "energy_comparison", "all"], "plot_generation must be one of 'yolo_comparison', 'quantization_comparison', 'buffer_depths', 'energy_comparison', or 'all'"

# you can choose the version of the yolo_comparison plot to generate (available versions are 1 and 2)
yolo_comparison_version = 2
assert yolo_comparison_version in [1, 2], "yolo_comparison_version must be one of 1 or 2"

# you can choose the version of the energy_comparison plot to generate (available versions are 1 and 2)
energy_comparison_plot_type = 'bar'
assert energy_comparison_plot_type in ['bar', 'scatter'], "energy_comparison_plot_type must be one of 'bar' or 'scatter'"

# you can choose the version of the quantization_comparison plot to generate (available versions are "combined" and "separate")
quantization_comparison_version = "separate"
assert quantization_comparison_version in ["combined", "separate"], "quantization_comparison_version must be one of 'combined' or 'separate'"

if plot_generation in ["yolo_comparison", "all"]:
    cpu_gpu_df = pd.read_csv('Results_CPU_and_GPU.csv')
    cpu_gpu_cols = cpu_gpu_df.columns.to_list()
    print(cpu_gpu_cols)

    # Version 1: Plotting the results for the different models on different devices
    if yolo_comparison_version == 1:

        # Create a combined column for Model and Input Shape named Model-Input Shape
        cpu_gpu_df['Model Family-Input Shape'] = cpu_gpu_df['Model Family'] + '-' + cpu_gpu_df['Input Shape']

        # filter based on Platform
        cpu_gpu_df = cpu_gpu_df[cpu_gpu_df['Platform'].isin(['ARM Cortex-A72', 'Jetson TX2'])]
        cpu_gpu_arm_df = cpu_gpu_df[cpu_gpu_df['Platform'].isin(['ARM Cortex-A72'])]
        cpu_gpu_tx2_df = cpu_gpu_df[cpu_gpu_df['Platform'].isin(['Jetson TX2'])]
        # TODO: Add here the filtering for the FPGA results following the same pattern as above

        fig, ax = plt.subplots()
        sns.lineplot(data=cpu_gpu_arm_df, x='Inf. Time (ms)', y='mAP50-95 @ coco128', ax=ax, hue='Model Family-Input Shape', style='Platform', markers=['o', 's'],  dashes=False, linewidth=3, markersize=10, palette=COLOUR_PALETTE, linestyle='solid')
        sns.lineplot(data=cpu_gpu_tx2_df, x='Inf. Time (ms)', y='mAP50-95 @ coco128', ax=ax, hue='Model Family-Input Shape', style='Platform', markers=['s', 'o'],  dashes=False, linewidth=3, markersize=10, palette=COLOUR_PALETTE, linestyle='dashed')
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

    # Version 2: Plotting the results for the different models on different input sizes
    elif yolo_comparison_version == 2:

        # filter based on Input Shape
        cpu_gpu_df = cpu_gpu_df[cpu_gpu_df['Input Shape'].isin(['640x640', '416x416'])]

        # remove rows that are in the yolov3 model family and have 'Input Shape' 640x640
        cpu_gpu_df = cpu_gpu_df[~((cpu_gpu_df['Model Family'] == 'yolov3') & (cpu_gpu_df['Input Shape'] == '640x640'))]

        # filter based on Platform
        cpu_gpu_df = cpu_gpu_df[cpu_gpu_df['Platform'].isin(['ARM Cortex-A72', 'Jetson TX2', 'Alveo U250'])]
        # TODO: Add in the list above the FPGA filtering

        fig, ax = plt.subplots()
        sns.lineplot(data=cpu_gpu_df, x='Inf. Time (ms)', y='mAP50-95 @ coco128', ax=ax, hue='Platform', style='Model Family', markers=['o', 's', 'p'],  dashes=True, linewidth=3, markersize=10, palette=COLOUR_PALETTE, linestyle='solid')

        # Add some text for each point in the line
        for line in range(0,cpu_gpu_df.shape[0]):
            curr_series = cpu_gpu_df.iloc[line]
            ax.text(curr_series['Inf. Time (ms)'], curr_series['mAP50-95 @ coco128'], curr_series['Model'], ha='center', va='bottom', size=FONT_SIZE_TEXT, color='black', weight='bold')

        ax.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABELS)
        ax.set_ylabel('mAP50-95', fontsize=FONT_SIZE_LABELS)
        ax.set_xscale('log')
        handles, labels = ax.get_legend_handles_labels()
        handles = handles[1:4] + handles[5:]
        labels = labels[1:4] + labels[5:]
        order = [0, 3, 1, 4, 2, 5]
        handles = [handles[idx] for idx in order]
        labels = [labels[idx] for idx in order]
        ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.155), ncol=3, fontsize=FONT_SIZE_LEGEND, frameon=False)


    ax.set_xticklabels(ax.get_xticklabels(), fontsize=FONT_SIZE_TICKS)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=FONT_SIZE_TICKS)

    # plt.tight_layout()
    # plt.savefig(f'yolo_comparison_devices_v{yolo_comparison_version}.png', bbox_inches='tight')
    plt.savefig('yolo_comparison_devices.pdf', bbox_inches='tight', format='pdf')

if plot_generation in ["quantization_comparison", "all"]:
    quant_df = pd.read_csv('Results_Quantization.csv')
    quant_cols = quant_df.columns.to_list()
    print(quant_cols)

    # filter based on Input Shape
    quant_df = quant_df[quant_df['Input Shape'].isin(['640x640', '416x416'])]

    # remove rows that are in the yolov3 model family and have 'Input Shape' 640x640
    quant_df = quant_df[~((quant_df['Model Family'] == 'yolov3') & (quant_df['Input Shape'] == '640x640'))]

    # get the Model Families
    model_families = quant_df['Model Family'].unique().tolist()

    if quantization_comparison_version == "combined":
        #TODO: There is an issue with the final (yolov8) plot for this one. The mAP seems to be wrong
        FONT_SIZE_TICKS = 15
        FONT_SIZE_LABELS = 22
        FONT_SIZE_TITLE = 20
        FONT_SIZE_LEGEND = 13
        FONT_SIZE_TEXT = 15

        fig = plt.figure()
        gs = gridspec.GridSpec(2, 4)
        errorbar = lambda x: (x.min(), x.max())

        for idx, model_family in enumerate(model_families):
            curr_df = quant_df[quant_df['Model Family'].isin([model_family])]
            if idx == 0:
                curr_ax = fig.add_subplot(gs[0, :2])
            elif idx == 1:
                curr_ax = fig.add_subplot(gs[0, 2:])
            elif idx == 2:
                curr_ax = fig.add_subplot(gs[1, 1:3])
            sns.lineplot(data=curr_df, x='Weights Wordlength', y='mAP50-95 @ coco-val2017\nwith LAYER_BFP', ax=curr_ax, hue='Model Variant', palette=COLOUR_PALETTE, errorbar=errorbar, estimator="mean", err_style='band', linewidth=3, markersize=7, marker='o')

            # Make the title bold
            curr_ax.set_title(model_family, fontsize=FONT_SIZE_TITLE)
            curr_ax.set_xlabel('Weights Wordlength', fontsize=FONT_SIZE_LABELS)
            curr_ax.set_ylabel('mAP50-95', fontsize=FONT_SIZE_LABELS)
            # curr_ax.set_yscale('log')
            handles, labels = curr_ax.get_legend_handles_labels()
            curr_ax.legend(handles, labels, fontsize=FONT_SIZE_LEGEND, frameon=False)

            curr_ax.set_xticklabels(curr_ax.get_xticklabels(), fontsize=FONT_SIZE_TICKS)
            curr_ax.set_yticklabels(curr_ax.get_yticklabels(), fontsize=FONT_SIZE_TICKS)

        # plt.tight_layout()
        # plt.savefig('quant_comparison_devices_combined.png', bbox_inches='tight')
        plt.savefig('quant_comparison_devices_combined.pdf', bbox_inches='tight', format='pdf')

    elif quantization_comparison_version == "separate":

        FONT_SIZE_TICKS = 20
        FONT_SIZE_LABELS = 30
        FONT_SIZE_TITLE = 30
        FONT_SIZE_LEGEND = 20
        FONT_SIZE_TEXT = 15

        for idx, model_family in enumerate(model_families):
            curr_df = quant_df[quant_df['Model Family'].isin([model_family])]

            fig, ax = plt.subplots()

            # errorbar = 'ci' or 'pi' or 'sd' or 'se'
            errorbar = lambda x: (x.min(), x.max())
            sns.lineplot(data=curr_df, x='Weights Wordlength', y='mAP50-95 @ coco-val2017\nwith LAYER_BFP', ax=ax, hue='Model Variant', palette=COLOUR_PALETTE, errorbar=errorbar, estimator="mean", err_style='band', linewidth=3, markersize=7, marker='o')

            ax.set_xlabel('Weights Wordlength', fontsize=FONT_SIZE_LABELS)
            ax.set_ylabel('mAP50-95', fontsize=FONT_SIZE_LABELS)
            # ax.set_yscale('log')
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=4, fontsize=FONT_SIZE_LEGEND, frameon=False)

            ax.set_xticklabels(ax.get_xticklabels(), fontsize=FONT_SIZE_TICKS)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=FONT_SIZE_TICKS)

            # plt.tight_layout()
            # plt.savefig(f'quant_comparison_devices_{model_family}.png', bbox_inches='tight')
            plt.savefig(f'quant_comparison_devices_{model_family}.pdf', bbox_inches='tight', format='pdf')


if plot_generation in ["buffer_depths", "all"]:
    FONT_SIZE_TICKS = 17
    FONT_SIZE_LABELS = 25

    buffer_depths_df = pd.read_csv('Buffer-Depths_yolov5n-640-zcu104.csv')
    buffer_depths_cols = buffer_depths_df.columns.to_list()
    print(buffer_depths_cols)

    fig, ax = plt.subplots()

    sns.barplot(x=buffer_depths_df['Buffer'], y=buffer_depths_df['Buffer Size (KB)'], color=(0.702, 0.698, 0.984), edgecolor=(0.106, 0.062, 0.972), ax=ax, dodge=False)

    ax.set_xlabel('Buffers', fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel('Buffer Size (KB)', fontsize=FONT_SIZE_LABELS)

    ax.set_xticklabels([])
    # ax.set_xticklabels(ax.get_xticklabels(), fontsize=FONT_SIZE_TICKS - 7, rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=FONT_SIZE_TICKS)

    # add extra space before the first tick and after the last tick
    ax.relim()
    # ax.autoscale_view()
    ax.margins(x=0.015)

    # plt.tight_layout()
    # plt.savefig('buffer_depths.png', bbox_inches='tight')
    plt.savefig('buffer_depths.pdf', bbox_inches='tight', format='pdf')

if plot_generation in ["energy_comparison", "all"]:
    energy_df = pd.read_csv('Energy_Comparison.csv')
    energy_cols = energy_df.columns.to_list()
    print(energy_cols)

    fig, ax = plt.subplots()

    if energy_comparison_plot_type == 'bar':
        FONT_SIZE_TICKS = 20
        FONT_SIZE_LEGEND = 16

        sns.barplot(x=energy_df['Device'], y=energy_df['Energy (mJ)'], hue=energy_df['Input Shape'], ax=ax, width=0.7, edgecolor='black', palette=COLOUR_PALETTE)
        # sns.barplot(x=energy_df['Input Shape'], y=energy_df['Energy (mJ)'], hue=energy_df['Device'], ax=ax, width=0.7, edgecolor='black', palette=COLOUR_PALETTE)

        ax.set_xlabel('', fontsize=FONT_SIZE_LABELS)
        ax.set_ylabel('Energy (mJ)', fontsize=FONT_SIZE_LABELS)

        ax.set_xticklabels(ax.get_xticklabels(), fontsize=FONT_SIZE_TICKS)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=FONT_SIZE_TICKS)

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize=FONT_SIZE_LEGEND, frameon=False, markerscale=3)

        # plt.tight_layout()
        # plt.savefig('energy_comparison_barplot.png', bbox_inches='tight')
        plt.savefig('energy_comparison.pdf', bbox_inches='tight', format='pdf')

    elif energy_comparison_plot_type == 'scatter':

        sns.scatterplot(data=energy_df, x='Latency (ms)', y='Energy (mJ)', hue='Device', size='Input Shape', sizes=(300, 150), ax=ax, palette=COLOUR_PALETTE)

        ax.set_xlabel('Latency (ms)', fontsize=FONT_SIZE_LABELS)
        ax.set_ylabel('Energy (mJ)', fontsize=FONT_SIZE_LABELS)

        ax.set_xticklabels(ax.get_xticklabels(), fontsize=FONT_SIZE_TICKS)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=FONT_SIZE_TICKS)

        handles, labels = ax.get_legend_handles_labels()
        handles = handles[1:]
        labels = labels[1:]
        labels[5] = ''
        ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize=FONT_SIZE_LEGEND, frameon=False)

        # plt.tight_layout()
        # plt.savefig('energy_comparison_scatter.png', bbox_inches='tight')
        plt.savefig('energy_comparison.pdf', bbox_inches='tight', format='pdf')

if plot_generation in ["ablation", "all"]:



    buffer_size = np.array([ 1217.438, 919.428, 764.088, 638.008, 588.508, 534.508])
    buffer_bw   = np.cumsum([ 0, 0.455, 0.228, 0.057, 0.028, 0.028 ])
    buffer_lutram = np.array([ 136688, 97708, 77064, 61368, 56368, 51248])

    print(buffer_bw)

    sliding_window_size = 726.24
    weights_size = 1982

    io_bw = 1.365

    num_points = len(buffer_size)


    FONT_SIZE_TICKS = 20
    FONT_SIZE_LEGEND = 16
    FIG_HEIGHT=4
    FIG_WIDTH=8

    """
    Buffer Size Plot
    """

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    buffer_size_bar_plot = pd.DataFrame({
        "Weights" : [weights_size]*num_points,
        "Sliding Window" : [sliding_window_size]*num_points,
        "Buffers" : buffer_size,
    })

    buffer_size_bar_plot.plot(kind="bar", stacked=True, ax=ax)

    ax.set_xlabel('', fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel('On-Chip Memory (KB)', fontsize=FONT_SIZE_LABELS)

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=FONT_SIZE_TICKS)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=FONT_SIZE_TICKS)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4,
              fontsize=FONT_SIZE_LEGEND, frameon=False, markerscale=3)

    plt.savefig('ablation-buffer-size.pdf', bbox_inches='tight', format='pdf')

    """
    Bandwidth Plot
    """

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    bandwidth_bar_plot = pd.DataFrame({
        "IO" : [io_bw]*num_points,
        "Buffers" : buffer_bw,
    })

    bandwidth_bar_plot.plot(kind="bar", stacked=True, ax=ax)

    ax.set_xlabel('', fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel('Off-Chip Memory Bandwidth (Gbps)', fontsize=FONT_SIZE_LABELS)

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=FONT_SIZE_TICKS)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=FONT_SIZE_TICKS)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4,
              fontsize=FONT_SIZE_LEGEND, frameon=False, markerscale=3)

    plt.savefig('ablation-buffer-bw.pdf', bbox_inches='tight', format='pdf')

    """
    LUTRAM Plot
    """

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    lutram_bar_plot = pd.DataFrame({
        "lutram" : buffer_lutram,
    })

    lutram_bar_plot.plot(kind="bar", stacked=True, ax=ax, legend=False)

    ax.set_xlabel('', fontsize=FONT_SIZE_LABELS)
    ax.set_ylabel('LUTRAM', fontsize=FONT_SIZE_LABELS)

    ax.hlines(y=[101760], xmin=-1, xmax=6, colors='red', linestyles='--', lw=2)

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=FONT_SIZE_TICKS)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=FONT_SIZE_TICKS)

    plt.savefig('ablation-lutram.pdf', bbox_inches='tight', format='pdf')


