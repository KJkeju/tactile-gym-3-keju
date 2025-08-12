import os
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train_model(
    predict_mode,
    model,
    label_encoder,
    train_generator,
    val_generator,
    learning_params,
    save_dir,
    error_plotters=[None],
    calculate_train_metrics=False,
    device='cpu'
):
    """
    Trains a PyTorch model using the provided data generators, optimizer, and learning parameters.

    Args:
        predict_mode (str): Mode of prediction, e.g., 'classify' or 'regress', determines loss function.
        model (torch.nn.Module): The neural network model to train.
        label_encoder: Object with encode_label, decode_label, calc_metrics, print_metrics, write_metrics methods.
        train_generator: Training dataset generator.
        val_generator: Validation dataset generator.
        learning_params (dict): Dictionary of training hyperparameters (batch_size, lr, epochs, etc.).
        save_dir (str): Directory to save model checkpoints and logs.
        error_plotters (list): Optional list of plotter objects for error visualization.
        calculate_train_metrics (bool): Whether to calculate and log training metrics.
        device (str): Device to use for training ('cpu' or 'cuda').

    Returns:
        tuple: (lowest_val_loss, total_training_time)
    """
    # Initialize TensorBoard writer
    writer = SummaryWriter(os.path.join(save_dir, 'tensorboard_runs'))

    # Create DataLoaders for training and validation datasets
    train_loader = torch.utils.data.DataLoader(
        train_generator,
        batch_size=learning_params['batch_size'],
        shuffle=learning_params['shuffle'],
        num_workers=learning_params['n_cpu']
    )
    val_loader = torch.utils.data.DataLoader(
        val_generator,
        batch_size=learning_params['batch_size'],
        shuffle=learning_params['shuffle'],
        num_workers=learning_params['n_cpu']
    )

    # Choose loss function based on prediction mode
    loss = nn.CrossEntropyLoss() if 'classify' in predict_mode else nn.MSELoss()

    # Initialize Adam optimizer with specified hyperparameters
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_params['lr'],
        betas=(learning_params["adam_b1"], learning_params["adam_b2"]),
        weight_decay=learning_params['adam_decay']
    )

    # Learning rate scheduler to reduce LR on plateau of validation loss
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=learning_params['lr_factor'],
        patience=learning_params['lr_patience']
    )

    def run_epoch(loader, n_batches_per_epoch, training=True):
        """
        Runs one epoch of training or validation.

        Args:
            loader: DataLoader for the current dataset.
            n_batches_per_epoch (int): Max number of batches per epoch.
            training (bool): Whether to run in training mode.

        Returns:
            tuple: (batch_loss, batch_acc, pred_df, targ_df)
        """
        batch_loss, batch_acc = [], []
        pred_df, targ_df = pd.DataFrame(), pd.DataFrame()

        for i, batch in enumerate(loader):
            if n_batches_per_epoch and i >= n_batches_per_epoch:
                break

            # Prepare input tensors and move to the specified device
            inputs = Variable(batch['inputs']).float().to(device)
            labels = label_encoder.encode_label(batch['labels'])

            if training:
                optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss_val = loss(outputs, labels)
            batch_loss.append(loss_val.item())

            # Compute accuracy for classification tasks
            if "classify" in predict_mode:
                batch_acc.append((outputs.argmax(1) == labels.argmax(1)).float().mean().item())
            else:
                batch_acc.append(0.0)

            # Backward pass and optimizer step if in training mode
            if training:
                loss_val.backward()
                optimizer.step()

            # Collect predictions and targets for metric calculation
            if not training or calculate_train_metrics:
                pred_df = pd.concat([pred_df, pd.DataFrame.from_dict(label_encoder.decode_label(outputs))])
                targ_df = pd.concat([targ_df, pd.DataFrame.from_dict(batch['labels'])])

        return (
            batch_loss, batch_acc,
            pred_df.reset_index(drop=True).fillna(0.0), targ_df.reset_index(drop=True).fillna(0.0)
        )

    # Initialize training state variables
    training_start_time = time.time()
    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    lowest_val_loss = float('inf')

    # Main training loop over epochs using tqdm progress bar
    with tqdm(total=learning_params['epochs']) as pbar:
        for epoch in range(1, learning_params['epochs'] + 1):
            pbar.update(1)

            # Training & Validation for one epoch
            train_loss_epoch, train_acc_epoch, train_pred_df, train_targ_df = run_epoch(
                train_loader, learning_params['n_train_batches_per_epoch'], training=True
            )
            model.eval()
            val_loss_epoch, val_acc_epoch, val_pred_df, val_targ_df = run_epoch(
                val_loader, learning_params['n_val_batches_per_epoch'], training=False
            )
            model.train()

            # Log losses and accuracies
            train_loss.append(train_loss_epoch)
            train_acc.append(train_acc_epoch)
            val_loss.append(val_loss_epoch)
            val_acc.append(val_acc_epoch)

            # Print epoch summary
            print(f"\nEpoch: {epoch}")
            print(f"Train Loss: {np.mean(train_loss_epoch):.6f} | Train Acc: {np.mean(train_acc_epoch):.6f}")
            print(f"Val Loss:   {np.mean(val_loss_epoch):.6f} | Val Acc:   {np.mean(val_acc_epoch):.6f}")

            # Write metrics to TensorBoard
            writer.add_scalar('loss/train', np.mean(train_loss_epoch), epoch)
            writer.add_scalar('loss/val', np.mean(val_loss_epoch), epoch)
            writer.add_scalar('accuracy/train', np.mean(train_acc_epoch), epoch)
            writer.add_scalar('accuracy/val', np.mean(val_acc_epoch), epoch)
            writer.add_scalar('learning_rate', get_lr(optimizer), epoch)

            # Calculate and log additional metrics
            if calculate_train_metrics:
                print("Train Metrics")
                train_metrics = label_encoder.calc_metrics(train_pred_df, train_targ_df)
                label_encoder.print_metrics(train_metrics)
                label_encoder.write_metrics(writer, train_metrics, epoch, mode='train')
            val_metrics = label_encoder.calc_metrics(val_pred_df, val_targ_df)
            print("Validation Metrics")
            label_encoder.print_metrics(val_metrics)
            label_encoder.write_metrics(writer, val_metrics, epoch, mode='val')

            # Update plots
            for plotter in error_plotters:
                if plotter and not plotter.final_only:
                    if 'max_epochs' in plotter.__dict__:
                        plotter.update(train_loss, val_loss, train_acc, val_acc)
                    else:
                        plotter.update(val_pred_df, val_targ_df, val_metrics)

            # Save best model if validation loss improves
            val_loss_mean = np.mean(val_loss_epoch)
            if val_loss_mean < lowest_val_loss:
                print('Saving Best Model')
                lowest_val_loss = val_loss_mean
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                with open(os.path.join(save_dir, 'train_val_loss_acc.pkl'), 'bw') as f:
                    pickle.dump([train_loss, val_loss, train_acc, val_acc], f)
                with open(os.path.join(save_dir, 'val_pred_targ_err.pkl'), 'bw') as f:
                    pickle.dump([val_pred_df, val_targ_df, val_metrics], f)

            # Step the learning rate scheduler
            lr_scheduler.step(val_loss_mean)

        # Print total training time
        total_training_time = time.time() - training_start_time
        print(f"Training finished, took {total_training_time:.2f}s")

        # Save the final model state
        torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))

        # Return the lowest validation loss and total training time
        return lowest_val_loss, total_training_time


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == "__main__":
    pass
