#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors:
* Connor Dalby, University of Glasgow
* Damiano Ferrari, University of Brescia
* Michele Svanera, University of Glasgow
* Mattia Savardi, University of Brescia

Support function to manage multi-site dataset.
The goal of this file is to manage the dataset and return only the matrices (or filepaths),
needed for inference

"""

from logging import config
from pathlib import Path
from os.path import join as opj
import nibabel as nib
import numpy as np
from scipy import stats
import pandas as pd
import tensorflow as tf
from loguru import logger
from DeepThickness import python_utils

from DeepThickness.ImplicitNet.config import Config, InputOutputConfig, DataConfig
from LOD_Brain.src.deepthickness_tools import conform_image
from collections import Counter


def load_csv_with_fullpaths(config_io_dict: InputOutputConfig, config_data_dict: DataConfig):
    """
    Load the input CSV and build test-set input filepath arrays.

    :param config_io_dict: input/output configuration flags used to select
        input columns from the CSV
    :param config_data_dict: data configuration containing CSV directory and
        filename keys
    :return X_test: 2D numpy array of shape (n_test, n_input_channels) with
        filepath strings
    """

    def zip_arrays(accumulation:np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Just stacks an array next to an accumulation array. See its use in the
        construct_input_output() function.
        """
        return np.hstack((accumulation, np.expand_dims(b, axis=-1)))

    def construct_input_output(df):
        """
        Build network input filepath arrays from a dataframe according to the
        configured input channels. The output is a 2D array. E.g.:
        X = [[T1_a_path, segm_a_path], [T1_b_path, segm_b_path], ...]
        """
        # Input paths
        X_paths = np.empty(shape=(len(df),1), dtype=object)
        # zip together filenames based on the config.data flag
        if config_io_dict['in_T1']:
            X_paths = zip_arrays(X_paths, np.array(df["T1"]))
        if config_io_dict['in_gm_probability_map']:
            X_paths = zip_arrays(X_paths, np.array(df["segmentation_gm"]))
        if config_io_dict['in_wm_probability_map']:
            X_paths = zip_arrays(X_paths, np.array(df["segmentation_wm"]))
        if config_io_dict['in_segmentation']:
            X_paths = zip_arrays(X_paths, np.array(df["segmentation"]))
        X_paths = X_paths[:,1:] # Remove initial empty column

        return X_paths

    # Load csv
    csv = pd.read_csv(opj(config_data_dict['Path_in_csv'], config_data_dict['Filename_csv']))

    # Test set
    df_test = csv[csv['set'] == 'test']
    X_test = construct_input_output(df_test)    

    return X_test

def load_mesh_data_set_with_fullpath(config_data_dict: DataConfig):
    """
    Load mesh-related filepath columns from the input CSV.

    Missing columns are replaced with empty-string arrays of matching length.

    :param config_data_dict: data configuration containing CSV directory and
        filename keys
    :return: tuple of numpy arrays in this order:
        (lh_pial_surface, rh_pial_surface, lh_wm_surface, rh_wm_surface,
         lh_thickness, rh_thickness, orig, aparc, dataset_type,
         database_name, subject_name)
    """
    csv = pd.read_csv(opj(config_data_dict['Path_in_csv'], config_data_dict['Filename_csv']))
    
    # Get length of the dataframe for creating empty arrays
    n_rows = len(csv)
    
    # Define a helper function to safely get column values
    def get_column_if_exists(df, column_name):
        if column_name in df.columns.str.lower():
            return np.array(df[column_name])
        else:
            return np.array([""] * n_rows)
    
    # Get values for each column, handling missing columns
    subject_name = get_column_if_exists(csv, 'subject')
    dataset_type = get_column_if_exists(csv, 'dataset_type')
    database_name = get_column_if_exists(csv, 'database')
    lh_pial_surface = get_column_if_exists(csv, 'lh_pial_surface')
    rh_pial_surface = get_column_if_exists(csv, 'rh_pial_surface')
    lh_wm_surface = get_column_if_exists(csv, 'lh_wm_surface')
    rh_wm_surface = get_column_if_exists(csv, 'rh_wm_surface')
    lh_thickness = get_column_if_exists(csv, 'lh_thickness')
    rh_thickness = get_column_if_exists(csv, 'rh_thickness')
    orig = get_column_if_exists(csv, 'orig')
    aparc = get_column_if_exists(csv, 'aparc')

    return lh_pial_surface, rh_pial_surface, lh_wm_surface, rh_wm_surface, \
        lh_thickness, rh_thickness, orig, aparc, dataset_type, database_name, subject_name



def prepareDataset(config: Config):
    """
    Prepare dataset metadata and filepath arrays used for inference.

    :param config: Config object
    :return dataset: dict with loaded data (paths only) and few attributes
    """

    # Retrieve needed arguments and set up the input paths
    config_data_dict = config.data.dict()                       # dict with main paths on where data are

    # Import all the volume' filenames from the csv
    X_test_paths = load_csv_with_fullpaths(config.input_output.dict(), config_data_dict)
    
    input_shape, input_config, output_shape, output_config = python_utils.get_string_io_configuration(config.input_output)

    # Update 'experiment_dict' 
    experiment_dict = {}
    experiment_dict['data'] = dict()
  
    experiment_dict['data'].update({'len(X_test_paths)': len(X_test_paths)})
    experiment_dict['data'].update({'Input shape': input_shape})
    experiment_dict['data'].update({'Input configuration': input_config})
    experiment_dict['data'].update({'Output shape': output_shape})
    experiment_dict['data'].update({'Output configuration': output_config})


    # logger.info('Volume shapes: ' + str(data_dims))
    logger.info('All volumes number: ' + str(len(X_test_paths)))
    logger.info('Input shape: ' + input_shape)
    logger.info('Input configuration: ' + input_config)
    logger.info('Output shape: ' + output_shape)
    logger.info('Output configuration: ' + output_config)

    # Create a dict with all the material inside
    dataset = {}
    dataset['X_test_paths'] = X_test_paths

    lh_pial_surface, rh_pial_surface, lh_wm_surface, rh_wm_surface, \
    lh_thickness, rh_thickness, orig, aparc, dataset_type, database_name, subject_name = load_mesh_data_set_with_fullpath(config_data_dict)
    dataset["lh_pial_surface_test_paths"] = lh_pial_surface
    dataset["rh_pial_surface_test_paths"] = rh_pial_surface
    dataset["lh_wm_surface_test_paths"] = lh_wm_surface
    dataset["rh_wm_surface_test_paths"] = rh_wm_surface
    dataset["lh_thickness_test_paths"] = lh_thickness
    dataset["rh_thickness_test_paths"] = rh_thickness
    dataset["orig_test_paths"] = orig
    dataset["aparc_test_paths"] = aparc
    dataset["dataset_test_type"] = dataset_type
    dataset["database_test_name"] = database_name
    dataset["subject_test_names"] = subject_name
    
    return dataset

def load_input_volumes(paths, input_shape):
    """
    Load multi-channel input volumes into a single 4D array.

    :param paths: iterable of byte-string filepaths, one per input channel
    :param input_shape: target 4D shape (x, y, z, channels)
    :return X: float32 numpy array with shape input_shape; NaN and inf values
        are replaced with zeros
    """

    X = np.empty(input_shape, dtype=np.float32)
    for channel, path in enumerate(paths):
        X[:, :, :, channel] = np.nan_to_num(np.array(nib.load(path.decode('UTF-8')).dataobj, dtype=np.float32), nan=0, posinf=0, neginf=0)
    return X

def load_input_and_conform_volume(paths, input_shape):
    """
        Load input volumes into a single 4D array, conforming the first channel.

        The first channel is loaded via ``conform_image``; all remaining channels
        are loaded directly with nibabel.

        :param paths: iterable of byte-string filepaths, one per input channel
        :param input_shape: target 4D shape (x, y, z, channels)
        :return X: float32 numpy array with shape input_shape; NaN and inf values
                are replaced with zeros
    """

    X = np.empty(input_shape, dtype=np.float32)

    for channel, path in enumerate(paths):
        # decode from bytes to a Python string filename
        fname = path.decode('UTF-8')
        
        if channel == 0:
            # your preprocessing returns (nib_image, dict)
            img = conform_image(input_data=fname)

        else:
            # other channels, just load normally
            img = nib.load(fname)
        
        # extract the raw data, zero out nans & infs
        data = np.array(img.dataobj, dtype=np.float32)
        X[..., channel] = np.nan_to_num(data, nan=0, posinf=0, neginf=0)

    return X

def load_output_volumes(paths):
    """
    Load output volumes from filepath list.

    :param paths: iterable of byte-string output filepaths
    :return: tuple of float32 numpy arrays, one per path, each with NaN and
        inf values replaced with zeros
    """
    return tuple([np.nan_to_num(np.array(nib.load(path.decode('UTF-8')).dataobj, dtype=np.float32), nan=0, posinf=0, neginf=0) for path in paths])

def tf_zscore(x:np.ndarray, in_T1) -> np.ndarray:
    """ 
    Optionally z-score the first (T1) channel of an input volume.

    :param x: input volumes numpy array. Typically of shape (256,256,256, in_channel)
    :param in_T1: boolean-like flag indicating whether channel 0 is a T1 volume
    :return x_out: copy of x where channel 0 is z-scored when in_T1 is True
    """

    x_out = np.copy(x)
    if in_T1:
        x_out[:, :, :, 0] = stats.zscore(x_out[:, :, :, 0], axis=None)
    return x_out

def createDatasetTF(X_paths, config:Config):
    """
    Create a TensorFlow dataset that loads and shapes input volumes.

    :param X_paths: list with input data in the form of a matrix. E.g.:
                    [[file1_a, file1_b], [file2_a, file2b], ...]
    :param config: Config object used to infer number of input channels
    :return input_dataset: tf.data.Dataset yielding float32 tensors with shape
        (256, 256, 256, config.input_output.in_filters)
    """


    input_volume_size = (256,) * 3 # e.g. (256, 256, 256)
    input_channels = config.input_output.in_filters # e.g. 3
    input_shape = input_volume_size + (input_channels,) # (256,256,256,3)

    def input_generator():
        """
        This generator simply reads the input X_paths and yields them one at a time.
        """
        for X in X_paths:
            yield X

    def set_input_shape(x):
        """
        tensorflow is not able to infer the input shape automatically in this case,
        so this function set the input shape statically. This is **one** 4D array,
        typically with shape (256,256,256,config.input_output.in_filters).
        """
        x.set_shape(input_shape)
        return x

    # Define a generator thet yields the paths
    input_dataset = tf.data.Dataset.from_generator(input_generator, \
        output_signature=tf.TensorSpec(shape=(config.input_output.in_filters,), dtype=tf.string))


    # Load volumes from paths
    input_dataset = input_dataset.map(lambda x: tf.numpy_function(func=load_input_and_conform_volume, inp=[x, input_shape], Tout=(tf.float32)), num_parallel_calls=tf.data.AUTOTUNE)
    input_dataset = input_dataset.map(set_input_shape, num_parallel_calls=tf.data.AUTOTUNE)
    
    return input_dataset

def TFDatasetGenerator(config: Config, dataset: dict):
    """
    Build the batched inference dataset from loaded dataset paths.
    
    :param config: Config object
    :param dataset: dict with loaded data (paths only) and few attributes

    :return ds: repeated and batched tf.data.Dataset yielding a single-element
        tuple containing z-scored input tensors
    """

    ds_input = createDatasetTF(dataset['X_test_paths'], config)
    
    ds_input = ds_input.map(map_func=lambda x: tf.numpy_function(tf_zscore,
                                                                    inp=[x, config.input_output.in_T1],
                                                                    Tout=[tf.float32]),
                                        num_parallel_calls=tf.data.AUTOTUNE)
    
    ds = tf.data.Dataset.zip((ds_input,))
  
    ds = ds.repeat().batch(1)

    return ds
        
def inference_ds(dataset_paths, config):
    
    """
    Create the inference dataset according to ``config.data.inference_mode``.

    In ``'T1w'`` mode this returns only z-scored input tensors.
    In ``'all_inputs'`` mode this delegates to ``TFDatasetGenerator``.
    
    Args:
    - dataset_paths (dict): dictionary containing relevant input paths
    - config (Config): configuration object
    
    Returns:
    - ds_test: tf.data.Dataset
    """
    
    if config.data.inference_mode == 'T1w':
        # Create TF datasets 
        ds_input = createDatasetTF(dataset_paths['X_test_paths'], config)
        ds_test = ds_input.map(map_func=lambda x: tf.numpy_function(tf_zscore,
                                                                        inp=[x, config.input_output.in_T1],
                                                                        Tout=[tf.float32]),
                                            num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.repeat().batch(1)

    if config.data.inference_mode == 'all_inputs':
        ds_test = TFDatasetGenerator(config, dataset_paths)
            
    return ds_test
