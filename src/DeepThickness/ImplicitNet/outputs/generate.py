#!/usr/bin/env python3
"""
Created on Monday - October 10 2022, 15:36:49

@authors:
* Connor Dalby, University of Glasgow
* Michele Svanera, University of Glasgow
* Mattia Savardi, University of Brescia
* Damiano Ferrari, University of Brescia

Main Script for testing after training. Also contains modular functions used in other scripts
"""

import os
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from os.path import join as opj

import numpy as np
import tensorflow as tf
from loguru import logger
from tqdm import tqdm

sys.path.append("/NeuroMorph/src/")

from DeepThickness.ImplicitNet.config import Config
from DeepThickness.ImplicitNet.outputs.export import (
    save_metadata,
    save_predicted_meshes,
    save_subject_volumes,
    write_results_csv,
)
from DeepThickness.mesh_manager import CorticalSurfaceMap


def merge_Cth(subject_info):
    """
    Given a subject_info dictionary containing results for pial and wm surfaces,
    compute the mean cortical thickness (CTh) for each subject.

    Logic:
        - If both Pial and WM Mean_Predicted_CTh exist, take their average and add
          'Mean_Predicted_CTh' at the subject's top level.
        - If only one or none exists, leave subject data unchanged.
    
    Args:
        subject_info (dict): Dictionary with subject data and results.
    
    Returns:
        dict: Updated dictionary with 'Mean_Predicted_CTh' added only when both exist.
    """
    for _, subject_data in subject_info.items():
        if "results" not in subject_data:
            continue
            
        results = subject_data["results"]
        
        pial_cth = results.get("Pial", {}).get("Mean_Predicted_CTh")
        wm_cth = results.get("Wm", {}).get("Mean_Predicted_CTh")
        
        # Only compute mean if both exist
        if pial_cth is not None and wm_cth is not None:
            subject_data["Mean_Predicted_CTh"] = (pial_cth + wm_cth) / 2.0
    
    return subject_info



def subject_mesh_processing(
    pred_level_set,
    pred_distance_set,
    config,
    quality_mapper=None,
    subject_name=None,
    subj_dir=None,
    surface_type='pial',
    pred_surface_mesh=None
):
    """
    Starting from the predicted level and distance set, process everything to
    generate all meshes for an individual subject. We first generate a predicted mesh
    and create a dict with the average CTh. 

    Args:
        pred_level_set (np.ndarray): Predicted level set from model
        pred_distance_set (np.ndarray): Predicted distance set from model
        config (Config): configuration parameters
        quality_mapper (str, optional): Colour transfer function for mesh saving.
        subject_name (str, optional): Name of subject.
        subj_dir (str, optional): Subject output directory. If ``None``, no mesh
            files are saved.
        surface_type (str): the type of mesh surface being processed. Processing of the mesh will differ
            depending on the surface (e.g. different mc_level and norm_mult). Str component
            will be used as prefix for results and save paths
        pred_surface_mesh (CorticalSurfaceMap, optional): Precomputed mesh to use
            directly. If provided, mesh generation from volumes is skipped.

    Returns:
        dict: Dictionary containing:
            - ``pred_{surface_type}_ct_map`` (optional, only when
              ``config.testing.keep_mesh_in_memory`` is ``True``)
            - ``subject_name``
            - ``surface_type`` (capitalized)
            - ``subject_results``

    Logic:
        1. Generate (or use a precomputed) predicted mesh.
        2. Extract mesh metrics from the predicted mesh.
        3. Return the subject name, surface label, subject metrics, and optionally the mesh if
           desired for downstream processing

    """
    
    if pred_surface_mesh is not None:
        pred_ct_map = pred_surface_mesh

    else:
        pred_ct_map = subject_generate_mesh(
            pred_level_set=pred_level_set,
            pred_distance_set=pred_distance_set,
            config=config,
            subj_dir=subj_dir,
            quality_mapper=quality_mapper,
            subject_name=subject_name,
            surface_type=surface_type
        )

    subject_results = pred_ct_map.extract_mesh_metrics(surface_type=surface_type)

    outputs = {
        **({f"pred_{surface_type}_ct_map": pred_ct_map}
           if config.testing.keep_mesh_in_memory else {}),
        "subject_name": subject_name,                
        "surface_type": surface_type.capitalize(),                
        "subject_results": subject_results,          
        }
    

    return outputs

def extract_volumetric_features(segmentation_mask):
    """
    Given the segmentation mask, extract the number of voxels (volume) for target segmentations
    
    Args:
        - segmentation_mask (ndarray): a 256^3 volume with each voxel being an int value that flags 
          what region that volume belongs
    
    Returns:
        volumetrics (dict): Dictionary containing normalized regional volumes
            (divided by intracranial volume) and raw intracranial volume voxel
            count.
        
    Logic:
        1. Builds boolean masks for each of the target regions
        2. Counts the non-zero voxels in each mask.
        3. Normalizes regional volumes by intracranial volume (except
           ``Intracranial Volume``, which is returned as raw voxel count).
        4. Returns a dict mapping region names to normalized volumes and
           intracranial volume.
        
    """
    
    gm_mask = (segmentation_mask == 1)
    basal_ganglia_mask = (segmentation_mask == 2)
    wm_mask = (segmentation_mask == 3)
    ventricles_mask = (segmentation_mask == 4)
    cerebellum_mask = (segmentation_mask == 5)
    brainstem_mask = (segmentation_mask == 6)
    csf_mask = (segmentation_mask == 7)
    cerebrum_mask = (segmentation_mask != 0)
    
    # Count the number of non-zero voxels in each mask
    intracranial_volume = np.count_nonzero(cerebrum_mask)
    gm_volume = np.count_nonzero(gm_mask)
    basal_ganglia_volume = np.count_nonzero(basal_ganglia_mask)
    wm_volume = np.count_nonzero(wm_mask)
    ventricular_volume = np.count_nonzero(ventricles_mask)
    cerebellum_volume = np.count_nonzero(cerebellum_mask)
    brainstem_volume = np.count_nonzero(brainstem_mask)
    csf_volume = np.count_nonzero(csf_mask)

    # Divide each volume by ICV (cerebrum volume) 
    gm_volume = gm_volume / intracranial_volume
    basal_ganglia_volume = basal_ganglia_volume / intracranial_volume
    wm_volume = wm_volume / intracranial_volume
    ventricular_volume = ventricular_volume / intracranial_volume
    cerebellum_volume = cerebellum_volume / intracranial_volume
    brainstem_volume = brainstem_volume / intracranial_volume
    csf_volume = csf_volume / intracranial_volume
    
    volumetrics = {
        'GM Volume' : gm_volume,
        'WM Volume' : wm_volume,
        'Ventricular Volume' : ventricular_volume,
        'Intracranial Volume' : intracranial_volume,
        'Basal Ganglia Volume': basal_ganglia_volume,
        'Cerebellum Volume': cerebellum_volume,
        'Brainstem Volume': brainstem_volume,
        'CSF Volume': csf_volume,
    }
    
    # Scale volumetrics to be in mm3
    normalised_volumetrics = ['GM Volume', 'WM Volume', 'Ventricular Volume', 'Basal Ganglia Volume',
                'Cerebellum Volume', 'Brainstem Volume', 'CSF Volume']
    for vol in normalised_volumetrics:
        volumetrics[vol] *= 100
    volumetrics['Intracranial Volume'] *= 0.0001
    
    return volumetrics
    
def subject_generate_mesh(
    pred_level_set,
    pred_distance_set,
    config,
    subj_dir,
    quality_mapper=None,
    subject_name=None,
    surface_type='pial'
):
    """
    Generates the mesh surface from the predicted level set and distance set using marching cubes,
    and optional post-processing (recommended), for a single subject. Save the outputs to subject
    directory if provided

    Args:
        pred_level_set (np.ndarray): Predicted level set from model
        pred_distance_set (np.ndarray): Predicted distance set from model
        config (Config): configuration parameters
        subj_dir (str / optional): path of subject directory in which to save the outputs
        quality_mapper (str, optional): what colour to apply to the vertices values when saving the ply mesh
        subject_name (str, optional): Name of subject
        surface_type (str): the type of mesh surface being processed. Processing of the mesh will differ
            depending on the surface (e.g. different mc_level and norm_mult). Str component
            will be used as prefix for results and save paths
       

    Returns:
        CorticalSurfaceMap: Predicted cortical surface map with thickness values.

    Logic:
        1. Generate the predicted mesh
            a. Build the `pred_ct_map` from predicted level and distance set.
            b. Save the predicted mesh if a subject directory is provided
        2. Return the `pred_ct_map` object.

    """

    if surface_type == 'pial':
        mc_level = config.testing.pial_mc_level
        norm_mult = config.testing.pial_norm_multiplier
        
    elif surface_type == 'wm':
        mc_level = config.testing.wm_mc_level
        norm_mult = config.testing.wm_norm_multiplier
    
    # If pred LS == GT LS, level would be 0.5
    pred_ct_map = CorticalSurfaceMap.from_volumetric_data(
        pred_level_set,
        pred_distance_set,
        mc_level=mc_level,
        norm_mult=norm_mult,
        apply_post_processing=config.testing.apply_mesh_post_processing,
        mesh_decimation_target=config.testing.mesh_decimation_target,
        subject_name=subject_name,
    )

    if subj_dir is not None:
        save_predicted_meshes(
            pred_ct_map, quality_mapper, config, subj_dir, subject_name, surface_type
        )

    return pred_ct_map

def single_prediction(model, X_test, config=None, check_dir=None):
    """
    Make a single model prediction for one input batch.

    Args:
        model (keras.Model): trained model to be used for inference and evaluation
        X_test: Input data (e.g., numpy array or tensor) to feed into the model for prediction.
        config (dict, optional): Unused; kept for API compatibility.
        check_dir (str or None, optional): Unused; kept for API compatibility.

    Returns:
        pred_dict (dict): ``model.predict_on_batch`` outputs, with tensors whose
            key ends in ``'_set'`` squeezed to remove singleton dimensions.


    Logic:
        1. Run ``model.predict_on_batch(X_test)``.
        2. Squeeze predictions for keys ending in ``'_set'``.
        3. Return the resulting prediction dictionary.
         
    """
    pred_dict = {}

    pred_dict = model.predict_on_batch(X_test)
    pred_dict = {
        key: val.squeeze() if key.endswith('_set') else val
        for key, val in pred_dict.items()
    }

    return pred_dict




class SequentialExecutor:
    """
    Logic Summary:
        Drop-in replacement for ProcessPoolExecutor that executes submitted
        callables immediately in the current process. Wraps results in a
        concurrent.futures.Future so downstream code using submit() and
        as_completed() works identically regardless of execution mode.

    Args:
        None

    Returns:
        None
    """

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def submit(self, fn, *args, **kwargs):
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as e:
            future.set_exception(e)
        return future


def parallel_process_ds(
    ds_test,
    model,
    subject_names,
    subject_save_data,
    config,
    quality_mapper,
    subjects_out_dir,
    device="gpu",
):
    """
    This function completes the following:
    - Load each sample in the ds_test dataset
    - Generate the level set and distance set (either from model prediction or load previously saved)
    - Generate model predictions for each sample
    - Generate the predicted cortical mesh (including CTh overlay)
    - The operations are designed to ensure that mesh generation (CPU-based) for multiple samples
      are processed in parallel with sequential model predictions (GPU-based), or run entirely
      sequentially depending on the parallel flag

    Args:
        ds_test: tf.data.Dataset
        model (keras.Model): trained model to be used for inference and evaluation
        subject_names: A list of all subject names in the dataset
        subject_save_data (dict): dictionary containing each subject's save information
        config (Config): configuration parameters
        quality_mapper (str): what colour to apply to the vertices values when saving the ply mesh
        num_cpu_workers (int): The number of CPU workers allocated to mesh generation
        subjects_out_dir (str): out dir to store the subject folders and results in
        parallel (bool): if True use ProcessPoolExecutor for mesh generation, if False
            run mesh generation sequentially in the current process

    Returns:
        subject_info (dict): Per-subject results dictionary containing subject
            index/dir metadata, optional volumetric features, per-surface mesh
            metrics, and merged ``Mean_Predicted_CTh`` when both pial and WM are
            available.

    """

    subject_info = {}
    mesh_futures = {}
    subject_idx = 0
    segmentation_mask_i = None

    executor = ProcessPoolExecutor(max_workers=config.testing.limit_cpu_count) if device == "gpu" \
        else SequentialExecutor()

    with executor as mesh_executor:
        for X_test in tqdm(ds_test,
            desc="Predicting subjects",
            total=len(ds_test),
            unit="subj",
        ):

            # --- Phase 1: Model Predictions ---
            subject_name = subject_names[subject_idx]
            subj_dir = subject_save_data[subject_name]["subj_dir"]
            subject_info[subject_name] = {"index": subject_idx, "subj_dir_i": subj_dir}
            subject_save = subject_idx < config.input_output.no_of_save_outputs or \
                config.input_output.no_of_save_outputs == -1

            predictions = single_prediction(
                model, X_test, config, check_dir=os.path.join(subjects_out_dir, subject_name)
            )

            if config.input_output.out_segmentation is True:
                segmentation_mask_i = predictions['segmentation_mask']
                subject_info[subject_name].update(extract_volumetric_features(segmentation_mask_i))

            for surface in ('pial', 'wm'):

                if any(surface in key for key in predictions):

                    pred_surface_mesh = predictions[f'{surface}_ct_map'] \
                        if any("ct_map" in key for key in predictions) else None

                    pred_level_set_i = predictions[f'{surface}_level_set']
                    pred_distance_set_i = predictions[f'{surface}_distance_set']

                    if subject_save:
                        save_subject_volumes(
                            pred_level_set=pred_level_set_i,
                            pred_distance_set=pred_distance_set_i,
                            segmentation_mask=segmentation_mask_i,
                            subject_i_save_data=subject_save_data[subject_name],
                            config=config,
                            surface_type=surface
                        )

                    # --- Phase 2: Mesh Generation ---
                    future = mesh_executor.submit(
                        subject_mesh_processing,
                        pred_level_set=pred_level_set_i,
                        pred_distance_set=pred_distance_set_i,
                        config=config,
                        quality_mapper=quality_mapper,
                        subject_name=subject_name,
                        subj_dir=subj_dir,
                        surface_type=surface,
                        pred_surface_mesh=pred_surface_mesh
                    )

                    mesh_futures[future] = subject_name
            subject_idx += 1

        for future in tqdm(
            as_completed(mesh_futures),
            total=len(mesh_futures),
            desc="Generating Meshes",
            unit="subj mesh",
        ):
            subject_name = mesh_futures[future]
            try:
                mesh_result_dict = future.result()
                surface = mesh_result_dict["surface_type"]
                results = mesh_result_dict["subject_results"]
                subject_idx = subject_info[subject_name]["index"]
                subject_info[subject_name].setdefault("results", {})[surface] = results

            except Exception as e:
                logger.warning(f"{subject_name} (idx={subject_idx}) failed: {e!r}")
                continue

    assert (all(name in subject_info for name in subject_names)), \
        f"There are {len(subject_names) - len(list(subject_info.keys()))} missing subjects"

    subject_info = merge_Cth(subject_info)

    return subject_info


def inference_process(model, ds_test: tf.data.Dataset, dataset: dict, config: Config, 
                      path_out_folder, device: str = "gpu"):
    """
    Function to run inference across the test dataset, generate meshes,
    and write per-subject results.

    Args:
        - model (keras.Model): trained model to be used for inference
        - ds_test: tf.data.Dataset
        - dataset (dict): dictionary containing the paths of the test dataset
        - config (Config): configuration parameters
        - path_out_folder (str): out folder where the output dir is

    Returns:
        None

    Logic:
        - Set definitions and variables for function
        - Prepare per-subject metadata and output folders
        - Create a list of subject names to process
        - Submit all predictions and mesh generation to a parallel processing pipeline which
          returns a dictionary of results
        - Save a csv of subjects' results
        - Log time taken for testing

    """

    # Setup and definitions
    subject_save_data = None
    start_time = time.time()
    subjects_out_dir = opj(path_out_folder, "Subjects/")
    if not os.path.exists(subjects_out_dir):
        os.mkdir(subjects_out_dir)
    
    dataset_size = int(len(dataset["X_test_paths"]))
    assert dataset_size > 0, "No subjects found for testing"
    ds_test = ds_test.take(dataset_size)

    # Get subject names and volume info for saving
    subject_save_data = save_metadata(
        subject_indexes=dataset_size,
        dataset=dataset,
        out_dir=subjects_out_dir,
        config=config,
    )

    subject_names = dataset["subject_test_names"]

    # Get the subject results
    subject_info = parallel_process_ds(
        ds_test=ds_test,
        model=model,
        subject_names=subject_names,
        subject_save_data=subject_save_data,
        config=config,
        quality_mapper="BR",
        subjects_out_dir=subjects_out_dir,
        device=device
    )
    
    write_results_csv(subject_info=subject_info, dataset=dataset, 
                         path_out_folder=path_out_folder)
    
    logger.info(f'Finished Model Testing on {dataset_size} subjects, time needed (hh:mm:ss): \
        {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}')
    
    return

