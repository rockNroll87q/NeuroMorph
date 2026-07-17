#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors:
* Connor Dalby, University of Glasgow
* Michele Svanera, University of Glasgow
* Mattia Savardi, University of Brescia
* Damiano Ferrari, University of Brescia

General python utility functions.
"""

import os
import sys

sys.path.insert(0, "src/")

from typing import Tuple
import tensorflow as tf
from loguru import logger
from DeepThickness.ImplicitNet.config import Config, InputOutputConfig
from LOD_Brain.src.deepthickness_tools import findListOfAnatomical, generate_testing_csv, adapt_existing_csv
import nvidia_smi
from pathlib import Path
import yaml

def selectGPUsAvailability():
    """
    I suppose we are going to use deepnet4-8: 4 x RTX8000 (45GB)
    It also tests if GPUs have free memory on.
    :return
        int {0-2}:  which GPU to use
        False:      means no GPU available -> need to exit
    """

    # Find the first available
    for i_gpu in reversed(range(4)):
        try:
            nvidia_smi.nvmlInit()
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i_gpu)
            mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        except Exception:
            logger.warning('Warning: GPU not visible to nvidia-smi')
            break

        # If free mem is more than 35GB, then select this gpu
        if mem_res.free / (1024. ** 3) > 40.:  # free more than 35GB of VRAM -> found my GPU
            return i_gpu

    return False  # Means 'deepnet5/8' no GPU available

def findGPUtoUse():
    """
    Function that find the GPU available and set the others not visible.

    :return which_gpu: int {0-2}  which GPU to use
    """

    # Find which GPU to use
    which_gpu = selectGPUsAvailability()  # int {0-3}:  which GPU to use
    if type(which_gpu) is bool:  # False: try setting manually GPU 0      
        gpus = tf.config.experimental.list_physical_devices('GPU')

        if gpus:
            try:
                # Select GPU 0
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                
                # Set memory growth for GPU 0
                tf.config.experimental.set_memory_growth(gpus[0], True)
                
                # Print confirmation
                logger.info(f"Set GPU 0 ({gpus[0].name}) with memory growth enabled.")
                
            except RuntimeError as e:
                # Memory growth must be set at program startup
                logger.info(f"Error: {e}")
    else:                                
        os.environ["CUDA_VISIBLE_DEVICES"] = str(which_gpu)  # which GPU make visible
        logger.info('GPU used: ' + str(which_gpu))

        # Configure GPUs to prevent OOM errors
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

def configure_device(device: str, max_cpus: int = None) -> None:
    """Configure TensorFlow device visibility before model load."""
    
    if max_cpus is not None:
        tf.config.threading.set_intra_op_parallelism_threads(max_cpus)
        tf.config.threading.set_inter_op_parallelism_threads(max_cpus)
        logger.info(f"CPU threads limited to {max_cpus}.")
        
    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")
        logger.info("Device set to CPU (GPU disabled).")
    elif device == "gpu":
        gpus = tf.config.list_physical_devices("GPU")
        findGPUtoUse()
        if not gpus:
            logger.warning(
                "No GPU detected. Falling back to CPU. "
                "Check your CUDA/cuDNN installation or use --device cpu explicitly."
            )
        else:
            logger.info(f"GPU detected: {[g.name for g in gpus]}")
    elif device == "auto":
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            logger.info(f"Auto-selected GPU: {[g.name for g in gpus]}")
            findGPUtoUse()
        else:
            logger.info("No GPU detected. Running on CPU.")
    else:
        raise ValueError(f"Unknown device '{device}'. Use 'auto', 'cpu', or 'gpu'.")

def configure_cpu_threads(max_cpus: int) -> None:
    """
    Restrict TensorFlow's CPU thread pools to a fixed number of threads.

    Args:
        max_cpus: Maximum number of CPU threads TensorFlow is allowed to use.

    Returns:
        None
    """
    tf.config.threading.set_intra_op_parallelism_threads(max_cpus)
    tf.config.threading.set_inter_op_parallelism_threads(max_cpus)

def get_string_io_configuration(config:InputOutputConfig) -> Tuple[str, str, str, str]:

    """
    Returns a tuple of strings with 4 elements:
    - The input shape, e.g. "(256,256,256,2)"
    - The input config, e.g. "T1, segmentation"
    - The output shape, e.g. "(256,256,256,2)"
    - The output config, e.g. "Pial surface level set, Pial surface distance set"

    Useful for logging purpose.
    """

    def get_config_string(list_config:list) -> str:
        output_string = ""
        for option, name in list_config:
            if option:
                output_string += (name + ", ")
        return output_string[:-2] # removes trailing ", "

    input_config = [(config.in_T1, "T1"), (config.in_segmentation, "segmentation"), \
        (config.in_wm_probability_map, "WM probability map"), (config.in_gm_probability_map, "GM probability map")]
    output_config = [(config.out_pial_surface_level_set, "Pial surface level set"), \
        (config.out_wm_surface_level_set, "WM surface level set"), \
        (config.out_pial_surface_distance_set, "Pial surface distance set"), \
        (config.out_wm_surface_distance_set, "WM surface distance set")]
    
    in_filters = config.in_filters
    in_volume_size = 256
    out_cardinality = config.out_cardinality
    out_volume_size = 256
    input_shape = f"({in_volume_size},{in_volume_size},{in_volume_size},{in_filters})"
    output_shape = f"{out_cardinality} × ({out_volume_size},{out_volume_size},{out_volume_size})"

    return input_shape, get_config_string(input_config), output_shape, get_config_string(output_config)

def load_args_config(config_cls=Config):
    """
    Function to handle arguments passed by command line.
    """

    # Argument parsing
    all_args = {}

    for arg in sys.argv:
        if not arg.startswith(("-", "--")):
            continue

        # Use partition to split at the first '='. Should work for param values that include a '=' in the string
        i_arg, _, i_value = arg.partition('=') # ex. "[--training.filters, 4]"
        i_arg = i_arg.replace('-', '')  # ex. "training.filters"

        # Splitting the argument into root and parameter
        try:
            i_root_arg, i_param_arg = i_arg.split('.')  # ex. "[training, filters]"
        except ValueError:
            logger.warning(f'Error: argument -{i_arg}- is not properly formatted.')
            sys.exit()

        if i_root_arg not in all_args:
            all_args[i_root_arg] = {}

        if (i_root_arg not in config_cls().dict().keys()) or (i_param_arg not in config_cls().dict()[i_root_arg].keys()):
            logger.warning(f'Error: argument -{i_root_arg}.{i_param_arg}- not recognised.')
            sys.exit()

        all_args[i_root_arg][i_param_arg] = i_value
    return all_args

def load_config_from_yaml(config_yaml_path = Path("src/DeepThickness/weights/pial_config.yaml")) -> Config:
    
    """
    Function loads in a previous training config from yaml and overrides previous paremeters 
    with any terminal arguments provided e.g. python script.py --input_output.in_T1=False
    
    Args:
    - config_yaml_path (pathlib Path): a path to the target yaml file 
    
    Returns:
    config (dict): dictionary containing configuration parameters from loaded yaml file 
                   plus any overriding arguments from the command line
                   
    Logic:
    - Load in the target config yaml in safe way 
    - Temporaily convert any pathlib Path objects back to str 
    - Override and config parameters with input arguments from command line
    - Return the loaded/updated Config class
    
    """
    
    # Load with UnsafeLoader so we get Path objects
    with open(config_yaml_path, 'r') as f:
        raw = yaml.load(f, Loader=yaml.UnsafeLoader)

    # Convert any Path → str (so it matches config)
    def stringify_paths(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: stringify_paths(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [stringify_paths(v) for v in obj]
        return obj

    cleaned: dict = stringify_paths(raw)

    # Pull in config overrides (still a nested dict of strings)
    config_overrides: dict = load_args_config()

    # Merge config overrides into the YAML data
    for section, params in config_overrides.items():
        # if the section wasn’t in YAML at all, start a new sub-dict
        if section not in cleaned:
            cleaned[section] = {}
        # update keys in that section with their new string values
        cleaned[section].update(params)

    return Config(**cleaned)

def get_out_config(config):
    """
    Based on the config class settings, returns a string representing the desired output of the model.

    Args:
        config (Config): Configuration object

    Returns:
        str: String representing the output configuration
    """

    if config.input_output.out_cardinality == 1:
        if config.input_output.out_pial_surface_level_set:
            return "pial_surface_level_set"
        elif config.input_output.out_pial_surface_distance_set:
            return "pial_surface_distance_set"
        elif config.input_output.out_wm_surface_level_set:
            return "wm_surface_level_set"
        elif config.input_output.out_wm_surface_distance_set:
            return "wm_surface_distance_set"
    
    elif config.input_output.out_cardinality == 2:
        if config.input_output.out_pial_surface_level_set and config.input_output.out_pial_surface_distance_set:
            return "both_pial"
        elif config.input_output.out_wm_surface_level_set and config.input_output.out_wm_surface_distance_set:
            return "both_wm"
    
    return "other"

def update_config_for_inference(config, out_folder):
    """
    A function that will update the config based on whether the inference
    is starting from an input of T1w only or includes the pre-generated
    LODBrain WM, GM and Segmentation masks. The function also creates a csv of
    Tw1 paths to be used for later data preperation. The function returns
    a string for model_type to inform later processing whether to build a
    combined model (LODBrain + Implicitnet for T1w inputs) or Implicitnet model
    (for T1w + WM + GM + Segmentation inputs)

    Args:
        - config (dict): dictionary containing configuration parameters
        - out_folder (str-path): directory to save the csv if generated

    Returns:
        - model_type (str): what kind of model to build
        - config (dict): updated config dictionary based on inference mode

    Logic:
        - Set any config parameters needed for inference
        - If inference mode is T1w:
            The user is providing only T1w image(s) as input
            - Set config parameters for T1w input
            - If vol_in path is dir, recursively search and create a
              list of all the T1w images present from vol_in dir
            - If vol_in path is a path, use that T1w path only
            - Create a csv of the T1w images (and other columns needed)
              for dataset creation
            - Set the model type to 'Combined_Model'
        - If inference mode is all inputs:
            The user is providing T1w, WM mask, GM mask and Segmentation Mask as input
            - Set config parameters for all inputs
            - A csv must be provided for this inference mode - we presume one is given
            - Set the model type to 'Implicitnet'
        - Return the model type and updated config class

    """
    model_type = None
    all_anat = None

    config.data.output_dir = Path(config.data.output_dir)

    if config.data.inference_mode == "T1w":

        config.input_output.in_T1 = True
        config.input_output.in_gm_probability_map = False
        config.input_output.in_wm_probability_map = False
        config.input_output.in_segmentation = False

        # From single T1w volume or directory of T1w volumes
        if config.data.vol_in is not None:
            volume_path = Path(config.data.vol_in)
            if volume_path.is_dir():
                all_anat = findListOfAnatomical(path_in=volume_path, identifier=config.data.file_identifier)

            elif volume_path.is_file():
                all_anat = [volume_path]

            else:
                raise FileNotFoundError(f"Path: {volume_path} does not exist")

            config.data.Path_in_csv = out_folder
            config.data.Filename_csv = Path(
                generate_testing_csv(all_anat=all_anat, csv_output_dir=out_folder)
            ).name


        # From CSV with T1 paths
        if config.data.Path_in_csv is not None and config.data.Filename_csv is not None:
            config.data.Filename_csv = Path(
                adapt_existing_csv(existing_csv_path=f'{config.data.Path_in_csv}/{config.data.Filename_csv}', 
                                     csv_output_dir=out_folder)).name
            config.data.Path_in_csv = out_folder

        model_type = "Combined_Model"

    elif config.data.inference_mode == "all_inputs":
        
        assert config.input_output.out_segmentation is not True, "Segmentation can only be generated from T1w volume only as input"
        
        config.input_output.in_T1 = True
        config.input_output.in_gm_probability_map = True
        config.input_output.in_wm_probability_map = True
        config.input_output.in_segmentation = True
        config.input_output.out_segmentation = False

        model_type = "DeepThickness"

    return model_type, config

def flatten_dict(d, parent_key='', sep='_'):
    items = {}
    for k, v in d.items():
        # if this is the top‑level “results” chunk, don’t include “results” in the name
        new_key = (parent_key + sep + k.replace(' ', '_')) if parent_key and parent_key != 'results' else k.replace(' ', '_')

        if isinstance(v, dict):
            # if we're descending into the “results” dict, reset parent_key to ''
            next_parent = new_key if new_key != 'results' else ''
            items.update(flatten_dict(v, next_parent, sep=sep))
        else:
            items[new_key] = v
    return items
