
import sys, os, time, shutil, random, argparse
import tensorflow as tf
import numpy as np

from model import build_network
from instance_loader import InstanceLoader
from util import load_weights
from train import run_batch, summarize_epoch

if __name__ == '__main__':
    dev = 0.01
    checkpoint_version = '0.5146'

    # Define argument parser
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-d', default=64, type=int, help='Embedding size for vertices and edges')
    parser.add_argument('-time_steps', default=32, type=int, help='# Timesteps')
    parser.add_argument('-dev', default=dev, type=float, help='Target cost deviation')
    parser.add_argument('-instances', default='instances/test', help='Path for the test instances')
    parser.add_argument('-checkpoint', default='training/dev=%.2f/checkpoints/acc=%s' % (dev, checkpoint_version), help='Path for the checkpoint of the trained model')

    # Parse arguments from command line
    args = parser.parse_args()

    # Setup parameters
    d                       = vars(args)['d']
    time_steps              = vars(args)['time_steps']
    target_cost_dev         = vars(args)['dev']

    # Create instance loader
    loader = InstanceLoader(vars(args)['instances'])

    # Build model
    print('Building model ...', flush=True)
    GNN = build_network(d)

    # Disallow GPU use
    config = tf.ConfigProto( device_count = {'GPU':0})
    with tf.Session(config=config) as sess:

        # Initialize global variables
        # print("Initializing global variables ... ", flush=True)
        # sess.run( tf.global_variables_initializer() )

        # Restore saved weights
        load_weights(sess,vars(args)['checkpoint'])

        n_instances = len(loader.filenames)
        batch_size = 16
        n_batches = n_instances // batch_size
        stats = { k:np.zeros(n_batches) for k in ['loss','acc','sat','pred','TP','FP','TN','FN'] }

        for (batch_i, batch) in enumerate(loader.get_batches(batch_size, target_cost_dev)):
            stats['loss'][batch_i], stats['acc'][batch_i], stats['sat'][batch_i], stats['pred'][batch_i], stats['TP'][batch_i], stats['FP'][batch_i], stats['TN'][batch_i], stats['FN'][batch_i] = \
                run_batch(sess, GNN, batch, batch_i, 0, time_steps, train=False, verbose=True)

        summarize_epoch(0,stats['loss'],stats['acc'],stats['sat'],stats['pred'],train=False)
