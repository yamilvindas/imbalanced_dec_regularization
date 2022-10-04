#!/usr/bin/env python3
"""
    This code implements some useful functions. For instance, it implements
    different functions allowing to point 3D and 2D points.
    They can be used to plot the 3D points datasets or their 2D projections.
    It also implements some functions allowing to plot the results of an experiment.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


# Global variables
OPACITY_POINTS = 0.2
AXIS_TICK_SIZE = 30

def plot_points_by_class_2D(data_points, labels, centroids=None, show_fig=True, save_fig=False, file_name_fig="", close_fig=True):
    """
        Plot the 2D points based on the giving labels. If centroids of the clusters are available, they are plotted
        too.

        Arguments:
        ----------
        data_points: list or numpy array
            List of 2D points to plot.
        labels: list or numpy array
            List of the corresponding labels of the 2D points.
        centroids:
            List of 2D points corresponding to the centroids of the points in data_points
        show_fig: bool
            True if the figure need to be showed
        save_fig: bool
            True if the figure need to be saved
        file_name_fig:
            Path and name where the user want to save the figure. It only works if save_fig=True
        close_fig: bool
            Boolean indicating if the figure has to be closed or not.

    """
    # Sorting the samples by cluster
    points_to_plot = {}
    for i in range(len(labels)):
        label = int(labels[i])
        if (label not in points_to_plot):
            points_to_plot[label] = [data_points[i]]
        else:
            points_to_plot[label].append(data_points[i])
    # Plot
    plt.figure()
    for class_ID in reversed(list(set(labels))):
        current_points_to_plot = np.array(points_to_plot[class_ID])
        plt.scatter(current_points_to_plot[:,0], current_points_to_plot[:,1], label="Cluster {}".format(class_ID), alpha=OPACITY_POINTS)
    # Plotting the centroids
    if (centroids is not None):
        i = 0
        for centroid in centroids:
            if (len(centroid) == 2):
                plt.scatter(centroid[0], centroid[1], label="Centroid of cluster {}".format(i), alpha=OPACITY_POINTS)
            elif (len(centroid) == 1):
                plt.scatter(centroid[0], 2, label="Centroid of cluster {}".format(i), alpha=OPACITY_POINTS)

            i += 1
    plt.legend()

    # Saving the figure
    if (save_fig):
        plt.savefig('./tmpImagesForGIF/'+file_name_fig)

    # Showing the figure
    if (show_fig):
        plt.show()

    # Closing the figure
    if (close_fig):
        plt.close()


def plot_points_by_class_3D(data_points, labels, centroids=None, show_fig=True, save_fig=False, file_name_fig="", close_fig=True):
    """
        Plot the 3D points based on the giving labels. If centroids of the clusters are available, they are plotted
        too.

        Arguments:
        ----------
        data_points: list or numpy array
            List of 3D points to plot.
        labels: list or numpy array
            List of the corresponding labels of the 2D points.
        centroids:
            List of 3D points corresponding to the centroids of the points in data_points
        show_fig: bool
            True if the figure need to be showed
        save_fig: bool
            True if the figure need to be saved
        file_name_fig:
            Path and name where the user want to save the figure. It only works if save_fig=True
        close_fig: bool
            Boolean indicating if the figure has to be closed or not.

    """
    # Sorting the samples by cluster
    points_to_plot = {}
    for i in range(len(labels)):
        label = int(labels[i])
        if (label not in points_to_plot):
            points_to_plot[label] = [data_points[i]]
        else:
            points_to_plot[label].append(data_points[i])
    # Plot
    fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    for class_ID in reversed(list(set(labels))):
        current_points_to_plot = np.array(points_to_plot[class_ID])
        ax.scatter(
                        current_points_to_plot[:,0],
                        current_points_to_plot[:,1],
                        current_points_to_plot[:,2],
                        label="Cluster {}".format(class_ID)
                  )
    ax.set_xlabel('X', fontsize=AXIS_TICK_SIZE )
    ax.set_ylabel('Y', fontsize=AXIS_TICK_SIZE )
    ax.set_zlabel('Z', fontsize=AXIS_TICK_SIZE )
    for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(AXIS_TICK_SIZE )
    for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(AXIS_TICK_SIZE )
    for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(AXIS_TICK_SIZE )
    # Plotting the centroids
    if (centroids is not None):
        i = 0
        for centroid in centroids:
            if (len(centroid) == 1):
                plt.scatter(centroid[0], 2, label="Centroid of cluster {}".format(i), alpha=OPACITY_POINTS)
            elif (len(centroid) == 2):
                plt.scatter(centroid[0], centroid[1], label="Centroid of cluster {}".format(i), alpha=OPACITY_POINTS)
            elif (len(centroid) == 3):
                plt.scatter(centroid[0], centroid[1], centroid[2], label="Centroid of cluster {}".format(i), alpha=OPACITY_POINTS)
            i += 1
    plt.legend()

    # Saving the figure
    if (save_fig):
        plt.savefig('./tmpImagesForGIF/'+file_name_fig)

    # Showing the figure
    if (show_fig):
        plt.show()

    # Closing the figure
    if (close_fig):
        plt.close()

def plot_animation(files_image_list, animation_file_name="./tmpGIFs/tmpGIF.mp4"):
    """
        Creates an animation (mp4 file) using the images in the list file_image_list.

        Arguments:
        ----------
        files_image_list: list
            List of paths to the images that are going to be used to create the animation.
        animation_file_name: str
            Path and name where the created animation should be stored.
    """
    # Plotting the animation
    fig, ax = plt.subplots()
    ims = []
    for filename in files_image_list:
        im = ax.imshow(plt.imread(filename), animated=True)
        ims.append([im])
    ax.axis('off')
    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=5000)
    ani.save(animation_file_name)
    plt.show()

    return ani

def get_mean_metrics(repetitions_fixed_epoch_metrics, metric_type):
    """
        Compute the mean metrics (over the repetitions) of an experiment at a
        given and fixed epoch.

        Arguments:
        ----------
        repetitions_fixed_epoch_metrics: list
            List of dictionaries. Each element corresponds to a dictionary with
            two keys, 'Train' and 'Test', and the values are also dictionaries
            with three keys 'MCC', 'F1-Score', and 'Accuracy'
        metric_type: str
            Metric to use for the computation. Three options: MCC, F1-Score, and
            Accuracy.
    """
    nb_repetitions = len(repetitions_fixed_epoch_metrics)
    metric_dict = {
                        'Train': [repetitions_fixed_epoch_metrics[i]['Train'][metric_type] for i in range(nb_repetitions)],
                        'Test': [repetitions_fixed_epoch_metrics[i]['Test'][metric_type] for i in range(nb_repetitions)]
                    }
    mean_train_metric, std_train_metric = np.mean(metric_dict['Train']), np.std(metric_dict['Train'])
    mean_test_metric, std_test_metric = np.mean(metric_dict['Test']), np.std(metric_dict['Test'])
    print("\tTest {}: {} +- {}".format(metric_type, mean_test_metric*100, std_test_metric*100))
