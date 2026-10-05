import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap, LogNorm
import numpy as np
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix
import itertools

# colors
orange = np.array([256 / 256, 128 / 256, 0 / 256, 1])  # orange
blue = np.array([51 / 256, 153 / 256, 1, 1])
purple = np.array([127 / 256, 0 / 256, 250 / 256, 1])
gray = np.array([60 / 256, 60 / 256, 60 / 256, 1])  # gray
green = np.array([50 / 256, 205 / 256, 50 / 256, 0.8])
light_green = np.array([200 / 256, 250 / 256, 90 / 256, 0.7])
red = np.array([256 / 256, 0 / 256, 50 / 256, 1])
black = np.array([0 / 256, 0 / 256, 0 / 256, 1])
dark_green = np.array([2 / 256, 128 / 256, 90 / 256, 0.7])
pink = np.array([256 / 256, 60 / 256, 200 / 256, 1])  # PINK
light_blue = np.array([0 / 256, 243 / 256, 256 / 256, 1])


def plot_two_pointclouds_z(pc, pc_2, labels=None, labels_2=None, name='', path_plot='', point_size=1, xyz=None, target_class=4,
                           label_names=['Surrounding', 'Tower', 'Power lines', 'Surrounding', 'Wind turbine']):
    """# Segmentation labels:
    # 0 -> ground
    # 1 -> tower
    # 2 -> cables
    # 3 -> surrounding env
    # 4 -> wind turbines
    """

    # labels = labels.astype(int)
    fig = plt.figure(figsize=[20, 10])
    N_CLASSES = len(label_names)

    # colormap
    viridisBig = plt.cm.get_cmap('viridis', 10)
    newcolors = viridisBig(np.linspace(0, 0.8, N_CLASSES))
    newcolors[:1, :] = green
    newcolors[1:2, :] = purple
    newcolors[2:3, :] = blue
    # newcolors[3:4, :] = green
    # newcolors[4:5, :] = red

    cmap = ListedColormap(newcolors)

    # =============
    # First subplot: Labels
    # =============
    ax = fig.add_subplot(1, 2, 1, projection='3d')  # xlim=(-1, 1), ylim=(-1, 1) zlim=(0, max(pc[:, 2]))
    # ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2]*200, c=labels, s=point_size, marker='o', cmap=cmap, vmin=0, vmax=N_CLASSES-1)
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 3], c=pc[:, 3], s=point_size, marker='o', cmap='viridis')

    # plt.title(f'Labeled point cloud: {pc.shape[0]} pts - class {target_class}: {len(labels[labels==target_class])} pts')
    plt.title(f'Ground truth')

    # ==============
    # Second subplot: Predictions
    # ==============
    ax = fig.add_subplot(1, 2, 2, projection='3d')# xlim=(-1, 1), ylim=(-1, 1))
    # ax.scatter(pc_2[:, 0], pc_2[:, 1], pc_2[:, 2]*200, c=labels_2, s=point_size, marker='o', cmap=cmap, vmin=0, vmax=N_CLASSES-1)
    ax.scatter(pc_2[:, 0], pc_2[:, 1], pc_2[:, 3], c=labels_2, s=point_size, marker='o', cmap=cmap, vmin=0, vmax=N_CLASSES-1)
    # plt.title(f'Prediction - class {target_class}: {len(labels_2[labels_2==target_class])} pts')
    plt.title(f'Prediction')

    name = 'z_color_' + name + '_' + str(len(labels_2[labels_2==target_class])) +'p' 
    if xyz is not None:
        coords= xyz[0, :]
    
    # Legend
    # ==============
    # Dynamically generate legend elements
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=label_names[i], markerfacecolor=newcolors[i], markersize=10)
        for i in range(N_CLASSES)
    ]

    ax.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.20, 0.5))  # , bbox_to_anchor=(1.04, 0.5)
    fig.set_dpi(300)
    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(right=0.85)
    if path_plot:
        # plt.gcf()
        if not os.path.exists(path_plot):
            os.makedirs(path_plot)
        plt.savefig(os.path.join(path_plot, name) + '.png')  # str(len(labels[labels==1])) +

    plt.close(fig)


def plot_pointcloud_with_labels_DALES(pc, labels, preds, miou=None, name='plot', path_plot='', point_size=1, n_classes=9):
    """# Segmentation labels:
    # 0 -> ground
    # 1 -> tower
    # 2 -> poles
    # 3 -> vegetation
    # 4 -> fences-buildings
    # 5 -> cars-trucks
    """

    labels = labels.astype(int)
    fig = plt.figure(figsize=[14, 6])

    # colormap
    viridisBig = plt.cm.get_cmap('viridis', 10)
    newcolors = viridisBig(np.linspace(0, 1, n_classes))

    newcolors[:1, :] = light_green
    newcolors[1:2, :] = blue
    newcolors[2:3, :] = purple
    newcolors[3:4, :] = green
    newcolors[4:5, :] = red
    newcolors[5:6, :] = pink

    class_labels = ['Ground', 'Tower', 'Poles', 'Vegetation', 'Fences-Buildings', 'Cars-Trucks']
    class_colors = [light_green, blue, purple, green, red, pink]

    cmap = ListedColormap(newcolors)

    # =============
    # First subplot
    # =============
    ax = fig.add_subplot(1, 2, 1, projection='3d', xlim=(-1, 1), ylim=(-1, 1), zlim=(min(pc[:, 2]), max(pc[:, 2])))
    sc = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=labels, s=point_size, marker='o', cmap=cmap, vmin=0, vmax=5)
    # plt.colorbar(sc, fraction=0.03, pad=0.1)
    plt.title('Ground Truth')

    # ==============
    # Second subplot
    # ==============
    ax = fig.add_subplot(1, 2, 2, projection='3d', xlim=(-1, 1), ylim=(-1, 1), zlim=(min(pc[:, 2]), max(pc[:, 2])))
    sc2 = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=preds, s=point_size, marker='o', cmap=cmap, vmin=0, vmax=5)
    # plt.colorbar(sc2, fraction=0.03, pad=0.1)
    plt.title('Preds')

    # Title
    xstr = lambda x: "None" if x is None else str(round(x, 2))
    plt.suptitle("Preds vs. Ground Truth #pts=" + str(len(pc)) ,
                #  'mIoU=' + xstr(miou) + ']\n',
                 fontsize=16)

   # ============
    # Add legend
    # ============
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=class_colors[i], markersize=8, label=class_labels[i]) for i in range(len(class_labels))]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1))

    ax.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.35, 0.5))  # , bbox_to_anchor=(1.04, 0.5)
    fig.set_dpi(200)
    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(right=0.85)
    plt.gcf()
    if path_plot:
        if not os.path.exists(path_plot):
            os.makedirs(path_plot)
        plt.savefig(os.path.join(path_plot, name) + '.png')  # str(len(labels[labels==1])) +

    fig.clear()
    plt.close(fig)


def show_confusion_matrix(cm, classes, normalize=False, log_scale=False,
                          title='Confusion matrix', cmap=plt.cm.Blues, fontsize=12, plot_bar=False, save_path=None,
                          n_colors = 4 ):
    """
    Display a confusion matrix with options for normalization and logarithmic scaling.
    
    Parameters:
    -----------
    cm : array-like
        Confusion matrix
    classes : list
        List of class names
    normalize : bool, optional
        Whether to normalize the confusion matrix (default: False)
    log_scale : bool, optional
        Whether to use logarithmic scale for colors (default: False)
    title : str, optional
        Title for the plot (default: 'Confusion matrix')
    cmap : colormap, optional
        Colormap for the plot (default: plt.cm.Blues)
    fontsize : int, optional
        Font size for labels (default: 12)
    plot_bar : bool, optional
        Whether to plot the colorbar (default: False)
    save_path : str, optional
        Path to save the figure. If None, figure is not saved (default: None)
    """
    # Normalize by row (true class)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm) * 100  # convert to %
        fmt = '.1f'
        value_label = "%"
    else:
        fmt = '.0e'
        value_label = "count"

    # Choose normalization for color scaling 
    norm = LogNorm(vmin=cm.min() + 1, vmax=cm.max()) if log_scale else None

    # Use discrete colormap with fewer colors
     # Reduced number of colors in the colorbar
    cmap_discrete = plt.get_cmap(cmap, n_colors)

    # Plot
    plt.figure(figsize=(4, 3))
    plt.imshow(cm, interpolation='nearest', cmap=cmap_discrete, norm=norm)
    plt.title(title, fontsize=fontsize + 2)
    plt.xticks(np.arange(len(classes)), classes, rotation=45, ha='right', fontsize=fontsize)
    plt.yticks(np.arange(len(classes)), classes, fontsize=fontsize)
    if plot_bar: 
        plt.colorbar()

    # Annotate cells
    thresh = np.nanmax(cm) / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        val = cm[i, j]
        text_color = 'black'
        if normalize:
            text_color = "white" if val > thresh else "black"
        plt.text(j, i, format(val, fmt),
                 ha="center", va="center", color=text_color, fontsize=fontsize)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=fontsize)
    plt.xlabel(f'Predicted label ({value_label})', fontsize=fontsize)
    plt.grid(False)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=True)
        print(f"Saved to {save_path}")

    plt.show()
