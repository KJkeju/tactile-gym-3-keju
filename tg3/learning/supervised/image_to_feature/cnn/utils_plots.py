import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")


class LearningPlotter:
    """
    Plots learning curves for loss and accuracy during training and validation.

    Args:
        save_dir (str, optional): Directory to save the plot images.
        name (str, optional): Filename for the saved plot.
        max_epochs (int, optional): Maximum number of epochs to display on the x-axis.
        final_only (bool, optional): If True, only plot at the end (no interactive updates).
    """

    def __init__(self, save_dir=None, name="plot_learning.png", max_epochs=None, final_only=False):
        """ 
        Initialize the LearningPlotter. 
        """
        self.save_dir = save_dir
        self.name = name
        self.max_epochs = max_epochs
        self.final_only = final_only

        if not self.final_only:
            plt.ion()
            self._fig, self._axs = plt.subplots(1, 2, figsize=(12, 4))

    def update(self, train_loss, val_loss, train_acc, val_acc):
        """ 
        Update the learning curves for loss and accuracy. 
        """
        # Clear previous plots
        for ax in self._axs.flat:
            ax.clear()

        # Convert inputs to numpy arrays
        train_loss = np.array(train_loss)
        val_loss = np.array(val_loss)
        train_acc = np.array(train_acc) if len(train_acc) else np.zeros_like(val_acc)
        val_acc = np.array(val_acc)

        epochs = np.arange(1, train_loss.shape[0] + 1)
        max_epochs = self.max_epochs or train_loss.shape[0]

        # Plot Loss
        for loss, color, label in zip([train_loss, val_loss], ['r', 'b'], ['Train', 'Val']):
            mean = loss.mean(axis=1)
            std = loss.std(axis=1)
            self._axs[0].plot(epochs, mean, color=color, label=label)
            self._axs[0].fill_between(epochs, mean - std, mean + std, color=color, alpha=0.25)
        self._axs[0].set_yscale('log')
        self._axs[0].set_xlabel('Epoch')
        self._axs[0].set_ylabel('Loss')

        # Plot Accuracy
        for acc, color, label in zip([train_acc, val_acc], ['r', 'b'], ['Train', 'Val']):
            mean = acc.mean(axis=1)
            std = acc.std(axis=1)
            self._axs[1].plot(epochs, mean, color=color, label=label)
            self._axs[1].fill_between(epochs, mean - std, mean + std, color=color, alpha=0.25)
        self._axs[1].set_xlabel('Epoch')
        self._axs[1].set_ylabel('Accuracy')
        self._axs[1].legend()

        # Save figure if save directory is specified
        if self.save_dir:
            save_file = os.path.join(self.save_dir, self.name)
            self._fig.savefig(save_file, dpi=320, pad_inches=0.01, bbox_inches='tight')

        # Adjust axes limits and redraw
        for ax in self._axs.flat:
            ax.set_xlim([0, max_epochs])
            ax.relim()
            ax.autoscale_view()

        self._fig.canvas.draw()
        # plt.pause(0.01) # to show plot

    def final_plot(self, train_loss, val_loss, train_acc, val_acc):
        """ 
        Plot the final learning curves for loss and accuracy. 
        """
        if self.final_only:
            self._fig, self._axs = plt.subplots(1, 2, figsize=(12, 4))
        self.update(train_loss, val_loss, train_acc, val_acc)


class ClassificationPlotter:
    """
    Plots a confusion matrix for classification results.

    Args:
        save_dir (str, optional): Directory to save the plot image.
        class_names (list, optional): List of class names for axis labels.
        name (str, optional): Filename for the saved plot.
        final_only (bool, optional): If True, only plot at the end (no interactive updates).
        normalize (bool, optional): If True, display normalized confusion matrix values.
    """

    def __init__(self, save_dir=None, class_names=None, 
            name="plot_classify.png", final_only=False, normalize=True):
        """
        Initializes the ClassificationPlotter for visualizing classification results.
        """
        self.save_dir = save_dir
        self.class_names = class_names
        self.name = name
        self.final_only = final_only
        self.normalize = normalize

        if not self.final_only:
            plt.ion()
            self._fig = plt.figure(figsize=(12, 12))


    def update(self, pred_df, targ_df, metrics=None):
        """ 
        Update the confusion matrix plot for classification results. 
        """
        self._fig.clf()
        cm = metrics['conf_mat']
        ax = self._fig.add_subplot(111)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

        # Set class names as axis tick labels if provided
        if self.class_names:
            tick_marks = np.arange(len(self.class_names))
            ax.set_xticks(tick_marks)
            ax.set_xticklabels(self.class_names, rotation=90, fontsize=12)
            ax.set_yticks(tick_marks)
            ax.set_yticklabels(self.class_names, fontsize=12)

        # Format for cell text: float if normalized, int otherwise
        fmt = '.2f' if self.normalize else 'd'
        thresh = cm.max() / 2.
        # Annotate each cell with its value
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)

        ax.set_xlabel('Target class', fontsize=16, fontweight='bold')
        ax.set_ylabel('Predicted class', fontsize=16, fontweight='bold')
        self._fig.tight_layout()

        # Save the figure if a save directory is specified
        if self.save_dir:
            save_file = os.path.join(self.save_dir, self.name)
            self._fig.savefig(save_file, dpi=320, pad_inches=0.01, bbox_inches='tight')

        self._fig.canvas.draw()
        # plt.pause(0.01) # to show plot

    def final_plot(self, pred_df, targ_df, metrics=None):
        """
        Plot the final confusion matrix for classification results.
        """
        if self.final_only:
            self._fig = plt.figure(figsize=(12, 12))
        self.update(pred_df, targ_df, metrics)


class RegressionPlotter:
    """
    Plots regression results: predicted vs. target values for each label.

    Args:
        save_dir (str, optional): Directory to save the plot image.
        label_names (list, optional): List of label names to plot.
        name (str, optional): Filename for the saved plot.
        final_only (bool, optional): If True, only plot at the end (no interactive updates).
    """

    def __init__(self, save_dir=None, label_names=None, name="plot_regress.png", final_only=False):
        """
        Initialize the RegressionPlotter.
        """
        self.save_dir = save_dir
        self.label_names = label_names or []
        self.name = name
        self.final_only = final_only

        self.n_plots = len(self.label_names)
        self.n_rows = int(np.ceil(self.n_plots / 3))
        self.n_cols = min(self.n_plots, 3)

        # Pad to fill grid if needed
        pad = (self.n_rows * self.n_cols) - self.n_plots
        self._plot_labels = self.label_names + [None] * pad

        if not self.final_only:
            plt.ion()
            self._fig, self._axs = plt.subplots(self.n_rows, self.n_cols, figsize=(4 * self.n_cols, 3.5 * self.n_rows))
            self._fig.subplots_adjust(wspace=0.3)

    def update(self, pred_df, targ_df, metrics=None):
        """ 
        Update the regression plots with new predictions and targets.
        """
        metrics = metrics or {}
        for ax in self._axs.flat:
            ax.clear()

        n_smooth = max(1, int(pred_df.shape[0] / 20))

        for ax, label_name in zip(self._axs.flat, self._plot_labels):
            if label_name:
                # Sort values for smooth plotting
                idx = targ_df[label_name].argsort()
                targ = targ_df[label_name].iloc[idx].astype(float)
                pred = pred_df[label_name].iloc[idx].astype(float)
                # Get uncertainty or error values for coloring, default to zeros
                unc = metrics.get('stdev', {}).get(label_name, pred * 0) if 'stdev' in metrics else \
                    metrics.get('err', {}).get(label_name, pred * 0) if 'err' in metrics else pred * 0
                cmap = 'inferno' if 'stdev' in metrics else 'gray'

                # Scatter plot of target vs predicted values, colored by uncertainty/error
                ax.scatter(targ, pred, s=1, c=unc, cmap=cmap)
                
                # Plot rolling average for smoother trend visualization
                av_targ, av_pred = targ.rolling(n_smooth).mean(), pred.rolling(n_smooth).mean()
                ax.plot(av_targ, av_pred, lw=1, c='r')

                # Display mean absolute error if available
                if 'err' in metrics and label_name in metrics['err']:
                    mae = metrics['err'][label_name].mean()
                    ax.text(0.05, 0.9, f'MAE = {mae:.4f}', transform=ax.transAxes)
 
                # Set axis labels and limits
                lims = [np.round(min(targ.min(), pred.min())), np.round(max(targ.max(), pred.max()))]
                ax.set(xlabel=f"target {label_name}", ylabel=f"predicted {label_name}", xlim=lims, ylim=lims)
                ax.set_xticks(ax.get_xticks()), ax.set_yticks(ax.get_xticks())
                ax.grid(True)
            else:
                ax.axis('off')

        if self.save_dir:
            save_file = os.path.join(self.save_dir, self.name)
            self._fig.savefig(save_file, dpi=320, pad_inches=0.01, bbox_inches='tight')

        self._fig.canvas.draw()
        # plt.pause(0.01) # to show plot

    def final_plot(self, pred_df, targ_df, metrics=None):
        """ 
        Plot the final regression results.
        """
        if self.final_only:
            self._fig, self._axs = plt.subplots(self.n_rows, self.n_cols, figsize=(4 * self.n_cols, 3.5 * self.n_rows))
            self._fig.subplots_adjust(wspace=0.3)
        self.update(pred_df, targ_df, metrics)


if __name__ == '__main__':
    pass
