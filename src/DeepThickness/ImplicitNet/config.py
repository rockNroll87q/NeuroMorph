#!/usr/bin/env python3
"""
@authors:
* Connor Dalby, University of Glasgow
* Damiano Ferrari, Univeristy of Brescia
* Michele Svanera, University of Glasgow
* Mattia Savardi, University of Brescia

A series of configuration classes for the ImplicitNet model.training and testing
"""

#from typing import List
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt, validator


class ExperimentConfig(BaseModel):
    name: str = Field("NeuroMorph", title="Experiment base name")
    seed: PositiveInt = Field(28, title="random seed")

class DataConfig(BaseModel):
    output_dir: str = Field('/NeuroMorph/out/', title="Dir to save the outputs")
    inference_mode: str = Field('T1w', title="What type of inference to perform. \
        If input is T1w only (e.g. file path , dir path or csv of paths), use 'T1w'. \
        If input is a csv of paths to T1w, GM, WM and Seg, use 'all_inputs'")
    vol_in: Optional[str]  = Field(None, title="Path to a single T1w volume or a folder containing T1w volumes")
    file_identifier : str = Field(".nii.gz", title="A substring for recursive T1w filename search in data.vol_in dir")
    Path_in_csv: str = Field('/NeuroMorph/csv/', title="csv path")
    Filename_csv: str = Field('LOD_Brain_dataset_valid_external.csv', title="csv filename")

class InputOutputConfig(BaseModel):
    # Possible input combination
    in_T1: bool = Field(True, title="Whether to include the T1 as an input of the network")
    in_gm_probability_map: bool = Field(False, title="Whether to include the GM probab map as an input of the network")
    in_wm_probability_map: bool = Field(False, title="Whether to include the WM probab map as an input of the network")
    in_segmentation: bool = Field(False, title="Whether to include the segmentation as an input of the network")
    
    # Method outputs
    no_of_save_outputs: Optional[int] = Field(1, title = "The number of subject to save outputs in testing.")
    out_T1: bool = Field(True, title="Whether to output the conformed T1w volume used for the model.")
    out_segmentation: bool = Field(True, title="Whether to output the 8 mask segmentation volume.")
    out_surface_type: str = Field("both", title="Whether to output pial/WM surface meshes. Options:'pial','wm','both'")
    out_surface_mesh: bool = Field(True, title="Whether to output the selected type surface mesh (with CTh overlay).")
    out_curvature_mesh: bool = Field(False, title="Whether to output the curvature surface mesh.")
    out_t1w_overlay_mesh: bool = Field(False, title="Whether to output the T1w intensity overlay surface mesh.")
    out_surface_fs: bool = Field(True, title="Whether to output the FreeSurfer surface files.")
    out_level_set: bool = Field(False, title="Whether to output the selected type level set volumes.")
    out_distance_set: bool = Field(False, title="Whether to output the selected type distance set volumes.")
    
    # Network outputs for training and testing
    out_wm_surface_distance_set: bool = Field(True, title="Whether the network outputs the WM surface distance set.")
    out_pial_surface_level_set: bool = Field(True, title="Whether the network outputs the pial surface level set.")
    out_wm_surface_level_set: bool = Field(True, title="Whether the network outputs the WM surface level set.")
    out_pial_surface_distance_set: bool = Field(True, title="Whether the network outputs the pial surface distance set")

    # in_filters : PositiveInt = Field(4, title="Number of filters in the first convolutional layer")
    @property
    def in_filters(cls):
        return get_filters_from_list([cls.in_T1, cls.in_gm_probability_map, cls.in_wm_probability_map,\
            cls.in_segmentation])
    
    @property
    def out_cardinality(cls):
        return get_filters_from_list([cls.out_pial_surface_level_set, cls.out_wm_surface_level_set, \
            cls.out_pial_surface_distance_set, cls.out_wm_surface_distance_set])


class TestConfig(BaseModel):
    pial_mc_level: float = Field(1.0, title="The level values to pass marching cubes algorithm.")
    pial_norm_multiplier: float = Field(1.05, title="The value to multiply the vertices norms by.") 
    wm_mc_level: float = Field(0.25, title="The level values to pass marching cubes algorithm.")
    wm_norm_multiplier: float = Field(0.15, title="The value to multiply the vertices norms by.") 
    mesh_decimation_target: PositiveInt = Field(75000, title="Number of vertices in the mesh")
    apply_mesh_post_processing: bool = Field(True, title="Whether to apply mesh post processing. See mesh_manager.py.")
    keep_mesh_in_memory: bool = Field(False, title="Whether to keep the mesh in memory for further processing.")
    limit_cpu_count: int = Field(None, title="Limit the number of CPU cores. Default is max-1")
    device: str = Field('auto', title="Device to use for inference. Options are 'auto', 'cpu' or 'gpu'.")

class NetConfig(BaseModel):
    num_initial_filter: PositiveInt = Field(4, title="number of filters in the first block")
    num_blocks_per_level: PositiveInt = Field(3, title="number of blocks per level")
    conv_block: str = Field("BottleNeck", title="conv block (either Plain or BottleNeck)")
    dropout_rate: NonNegativeFloat = Field(0.1, title="used dropout rate. A 0 value means no dropout")
    activation_enc: str = Field("elu", title="a registered tf2 activation function for the encoder")
    activation_dec: str = Field("elu", title="a registered tf2 activation function for the decoder")
    stride: PositiveInt = Field(2, title="applied downsampling and upsampling stride")
    bn: str = Field("GN", title="whether use batch norm (BN) or GroupNorm (GN) or None")
    kernel_initializer: str = Field("he_normal", title="kernel initializer")
    kernel_regularizer: PositiveFloat = Field(1e-2, title="l2 penalty")
    skip_connection_type: str = Field("Concatenate", title="How to merge skip to the decoder. 'Add' or 'Concatenate'")
    n_identity_layers: PositiveInt = Field(3, title = "The number of repetitions for each conv block up and down")
    squeeze_excite: bool = Field(False, title="Squeeze + excitation blocks in network to optimise features activation")
    split_decoder_level: NonNegativeInt = Field(4, title="Number of split to use in the encoder and decoder")
    saved_weights_path : str = Field(None, title="Path of initial weights for continuing a previous training.")    

    @validator('skip_connection_type')
    def norm_check(cls, v):
        if v not in ['Add', 'Concatenate']:
            raise ValueError('skip_connection_type must be in [Add, Concatenate]')
        return v

    @property
    def num_filters(cls):
        return [int(np.clip(cls.num_initial_filter * (2 ** i), 1, 128)) for i in range(cls.num_blocks_per_level)]

class LossConfig(BaseModel):
    level_set_0_loss: str = Field('ranged_weighted_mae', title="Loss for level_set_0 output.")
    distance_set_0_loss: str = Field('weighted_mae_0', title="Loss for distance_set_0 output.")
    target_max: PositiveFloat = Field(1.0, title='Maximum value of the target range to weight. Min = (Max-1).')
    target_gradient: str = Field('step', title='Slope vs step of weighting for non-capped values outside target range')
    range_weight: PositiveFloat = Field(10.0, title='weighting applied to values inside the target range')
    kl_weight: NonNegativeFloat = Field(0.1, title='weighting applied to kl divergence loss value')

class Config(BaseModel):
    experiment: ExperimentConfig = Field(ExperimentConfig(), title="Experiment configuration")
    network: NetConfig = Field(NetConfig(), title="Network configurations")
    input_output: InputOutputConfig = Field(InputOutputConfig(), title="Input/output network configuration")
    data: DataConfig = Field(DataConfig(), title="Data configurations")
    testing: TestConfig = Field(TestConfig(), title="Training configurations")
    losses: LossConfig = Field(LossConfig(), title="Loss configurations")


def get_help(schema, mother=''):
    """
    Create the help structure given a pydantic scheme
    :param schema: pydantic schema
    :param mother: the option group
    :return:
    """
    for c in schema["properties"]:
        if "allOf" in schema["properties"][c]:
            print(f'\n    [{c}]')
            sec_key = schema["properties"][c]["allOf"][0]["$ref"].split('/')[-1]
            get_help(schema["definitions"][sec_key], mother=f'{c}.')
        else:
            title = schema["properties"][c]["title"]
            arg_type = schema["properties"][c]["type"]
            default = schema["properties"][c].get("default", "---")
            print(f'      {title}\n      --{mother if mother else ""}{c} [{arg_type}] = {default}')


def show_help():
    """
    Shows an help message
    """
    print("""Usage: python main.py [OPTIONS]

        Script to train DeepThickness. The help is in the form --argument [type] = default

    Options:""")
    get_help(Config.schema())


def config_flattened(schema, mother=''):
    """
    Return a list of permitted config parameters
    :param schema: pydantic schema
    :param mother: the option group (used for recursion)
    :return:
    """
    out = []
    for c in schema["properties"]:
        if "allOf" in schema["properties"][c]:
            sec_key = schema["properties"][c]["allOf"][0]["$ref"].split('/')[-1]
            out += list(config_flattened(schema["definitions"][sec_key], mother=f'{c}.'))
        else:
            out += [f'{mother if mother else ""}{c}']
    return out

def validate_config_arg(arg):
    """
    Check if an argument is in the configuration scheme
    :param arg: argument
    :return: True if arg is present, False otherwise
    """
    config_list = config_flattened(Config.schema())
    return arg in config_list

def get_filters_from_list(list:list) -> int:
    """
    Giving a list of booleans it returns the number of values that are setted to
    True
    """
    counter = 0
    for i in list:
        if i:
            counter += 1
    return counter
