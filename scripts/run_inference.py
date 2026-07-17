#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday - October 10 2022, 15:36:49

@authors:
* Connor Dalby, University of Glasgow
* Michele Svanera, University of Glasgow
* Mattia Savardi, University of Brescia
* Damiano Ferrari, University of Brescia

Main Script for conducting inference or model testing.

"""


import os
import time
from os.path import join as opj
import sys
from pathlib import Path
from loguru import logger
import tensorflow as tf
import numpy as np
import matplotlib


sys.path.insert(0, "src/")
from DeepThickness.ImplicitNet.config import Config
from DeepThickness.ImplicitNet.data.dataset_manager import prepareDataset, inference_ds
from DeepThickness.ImplicitNet.model.network import build_inference_model
from DeepThickness.ImplicitNet.outputs.generate import inference_process
from DeepThickness import python_utils

matplotlib.use("Agg")



def main_inference():
    """
    Run inference/evaluation on model
    """

    # Load previous trained config then update with any terminal commands
    config = Config(**python_utils.load_args_config())
    path_out_folder = Path(f"{config.data.output_dir}")

    # Check if input is T1w only or also pre-generated LOD-Brain outputs
    model_type, config = python_utils.update_config_for_inference(
        config=config, out_folder=path_out_folder
    )
    
    tf.random.set_seed(config.experiment.seed)
    np.random.seed(config.experiment.seed)
    logger.add(opj(path_out_folder.as_posix(), config.experiment.name + ".log"))
    logger.info(f"Config: {config}")

    # Find a GPU available and set others not visible
    logger.info("\n\n\n******** Run started ********")
    logger.info(f"Running on {os.uname()[1]}")
    logger.info(f"Command line:{sys.executable}{sys.argv}")
    python_utils.configure_device(device=config.testing.device, 
                                  max_cpus=config.testing.limit_cpu_count)

    # Build the dataset for Model Testing
    logger.info("\n\n\n******** Dataset Preparation ********")
    dataset = prepareDataset(config)
    ds_test = inference_ds(dataset_paths=dataset, config=config)

    # Build Model for Model Testing
    logger.info("\n\n\n******** Build Model ********")
    model = build_inference_model(model_type=model_type, config=config)

    # Model Testing
    logger.info("\n\n\n******** Model Testing ********")
    inference_process(
        model=model,
        ds_test=ds_test,
        dataset=dataset,
        config=config,
        path_out_folder=path_out_folder,
        limit_cpu_count=config.testing.limit_cpu_count
    )

if __name__ == "__main__":
    main_inference()
