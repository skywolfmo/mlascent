#!/usr/bin/env python3

"""
Author: Taha Bouhsine
Email: contact@tahabouhsine.com
Created on: February 4th 2024
Last Modified: January 4th 2024
Description:
This Python script (train.py) is designed for pre-training an autoencoder  model. 
"""

import argparse
from utils.setup_gpus import setup_gpus
from utils.set_seed import set_seed
from utils.load_data import load_data
from utils.models.build_autoencoder_cnn import AutoEncoder
import os
import tensorflow as tf
import numpy as np
import wandb
from wandb.keras import WandbCallback


# Define the training step function
@tf.function
def train_step(inputs, model, im_loss_fn, optimizer, train_loss):
    with tf.GradientTape() as tape:
        # Forward pass
        rec_im = model(inputs)
        im_loss_value = im_loss_fn(inputs, rec_im)
        loss_value = (im_loss_value + 1) /2

    # Identify variables of encoder, decoder, and im_rec layers
    trainable_layers = ['encoder', 'decoder', 'im_rec', 'batch_normalization']
    trainable_vars = [var for var in model.trainable_variables if any(layer in var.name for layer in trainable_layers)]

    # Compute gradients
    gradients = tape.gradient(loss_value, trainable_vars)
    # Update weights
    optimizer.apply_gradients(zip(gradients, trainable_vars))
    # Update training metrics
    train_loss(loss_value)



def train(config):
    print(config)

    model_name = 'AutoEncoder'


    wandb.login()

    run = wandb.init(
        project="MLAscent2",
        entity="Skywofl",
        name=model_name
    )

    wandb.config.update(vars(config))

    print('Loading the Dataset...')
    # Load data
    train_data, val_data = load_data(config)

    # Build model
    print('Building Model:')
    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_policy(policy)

    with strategy.scope():
        model = build_model(config)
        model.summary()
        im_loss_fn = tf.keras.losses.CosineSimilarity(axis=1)

    wandb_callback = WandbCallback(save_model=False)

    wandb_callback.set_model(model)
    
    best_val_loss = 10000
        # Train for a few epochs

        # Log metrics with Wandb
        logs = {
            'epoch': epoch,
            'train_loss': train_loss.result(),
            'train_accuracy': train_accuracy.result(),
            'train_precision': train_precision.result(),
            'train_recall': train_recall.result(),
            'val_loss': val_loss.result(),
            'val_accuracy': val_accuracy.result(),
            'val_precision': val_precision.result(),
            'val_recall': val_recall.result()
        }
        for class_id in range(config.num_classes):
            logs[f'class_{class_id}_accuracy'] = class_accuracy[class_id].result(
            ).numpy()
            logs[f'class_{class_id}_precision'] = class_precision[class_id].result(
            ).numpy()
            logs[f'class_{class_id}_recall'] = class_recall[class_id].result(
            ).numpy()

        # # # After each epoch, we call the 'on_epoch_end' method of our callback
        wandb_callback.on_epoch_end(epoch, logs=logs)

        print(f'Epoch {epoch + 1}, '
              f'Train Loss: {train_loss.result().numpy()}, '
              f'Train Accuracy: {train_accuracy.result().numpy()}, '
              f'Val Loss: {val_loss.result().numpy()}, '
              f'Val Accuracy: {val_accuracy.result().numpy()}'
              )

        for class_id in range(config.num_classes):
            print(f'  Class {class_id}: '
                  f'Accuracy: {class_accuracy[class_id].result().numpy()}, '
                  f'Precision: {class_precision[class_id].result().numpy()}, '
                  f'Recall: {class_recall[class_id].result().numpy()}')

        current_val_loss = val_loss.result().numpy()
        if current_val_loss < best_val_loss:
            model.save(os.path.join(wandb.run.dir, f'{model_name}_best_val_loss_model.keras'))
            best_val_loss = current_val_loss

    print('Saving Last Model')
    # Save the trained model
    model.save(os.path.join(wandb.run.dir, f"last_{model_name}_model.keras"))

    


    print("Training completed successfully!")


if __name__ == '__main__':
    # Parsing command-line arguments
    parser = argparse.ArgumentParser(description='Train a CNN for image classification.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--img_height', type=int, default=64, help='Image height.')
    parser.add_argument('--img_width', type=int, default=64, help='Image width.')
    parser.add_argument('--seed', type=int, default=64, help='Random Seed.')
    parser.add_argument('--gpu', type=str, default='2,3', help='GPUs.')
    parser.add_argument('--num_img_lim', type=int, required=True, help='The number of images per class')
    parser.add_argument('--val_split', type=float, required=True, help='Validation Split')
    parser.add_argument('--n_cross_validation', type=int, required=True, help='Number of Bins for Cross Validation')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of Classes')
    parser.add_argument('--trainable_epochs', type=int, required=True, help='The number of epochs before the backbone become trainable')
    args = parser.parse_args()



    # Setup GPUs
    setup_gpus(args.gpu)
    set_seed(args.seed)

    train(args)