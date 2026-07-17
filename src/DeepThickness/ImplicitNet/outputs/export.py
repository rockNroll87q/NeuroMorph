#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday - October 10 2022, 15:36:49

@authors:
* Connor Dalby, University of Glasgow
* Michele Svanera, University of Glasgow
* Mattia Savardi, University of Brescia
* Damiano Ferrari, University of Brescia

Script for functions related to saving outputs.
"""

import os

import nibabel as nib
import numpy as np
import pandas as pd

from DeepThickness.python_utils import flatten_dict


def save_metadata(subject_indexes, dataset, out_dir, config):
    """
    Build per-subject save metadata used by downstream output writers.

    For subjects selected for saving, this loads the input T1 image and stores
    its spatial metadata (affine/header/shape, orientation, resolution), and
    creates that subject's output directory.

    Args:
       subject_indexes (int): Number of subjects to iterate over.
       dataset (dict): Dataset dictionary containing at least
           ``subject_test_names`` and ``X_test_paths``.
       out_dir (str): Path to the parent output directory.
       config (dict): Configuration object containing output options.

    Returns:
        dict: Mapping ``subject_name -> metadata`` with keys:
            ``T1_orientation``, ``T1_resolution``, ``T1_shape``,
            ``T1_affine``, ``T1_header``, and ``subj_dir``.
            If a subject is not selected for saving, metadata values remain
            ``None`` and ``subj_dir`` is ``None``.

    Logic:
    1. Loop over all subjects, extracting their name and input file paths.
    2. If that subject index is within the range of number of subjects to save, create the subject’s output
       directory and save the T1 image, capturing its affine and header.
    3. Record each subject’s metadata — affine, header, level set and distance set input paths, 
       and output directory — in a central dict.
    4. Return the dict mapping subject names to their saved metadata.

    """

    subject_save_data = {}

    for subject_index in range(subject_indexes):
        
        subject_name = dataset["subject_test_names"][subject_index]
        
        T1_path = dataset["X_test_paths"][subject_index][0]
        
        
        resolution = None
        orientation = None
        subj_dir = None
        affine = None
        header = None
        shape = None
        
        if (config.input_output.no_of_save_outputs == -1
            or subject_index < config.input_output.no_of_save_outputs
        ):
            subj_dir = os.path.join(out_dir, subject_name)
            os.makedirs(subj_dir, exist_ok=True)

            T1 = nib.load(T1_path)
            affine = T1.affine.copy()
            header = T1.header.copy()
            shape = T1.shape
            orientation = nib.aff2axcodes(T1.affine)
            resolution = header['pixdim'][1:4]
            if config.input_output.out_T1:
                nib.save(T1, f"{subj_dir}/{subject_name}.nii.gz")

        subject_save_data[subject_name] = {
            'T1_orientation': orientation,
            'T1_resolution': resolution,
            'T1_shape': shape,
            "T1_affine": affine,
            "T1_header": header,
            "subj_dir": subj_dir,
        }
        
    return subject_save_data

def save_predicted_meshes(pred_ct_map, quality_mapper, config, subj_dir, subject_name, surface_type = 'pial'):
    """
    Save selected mesh-based outputs for one predicted surface.

    Depending on output flags and selected surface type, this can save:
    cortical-thickness mesh, FreeSurfer surface/thickness files, curvature
    mesh, and a T1w-overlay mesh.

    Args:
       pred_ct_map (CorticalSurfaceMap): Predicted mesh with vertex values.
       quality_mapper (str): Transfer-function name used for PLY coloring.
       config (dict): Configuration object containing output flags.
       subj_dir (str): Subject output directory.
       subject_name (str): Subject identifier.
       surface_type (str): Surface label used in filenames (``'pial'`` or
           ``'wm'``).

    Returns:
       None

    Logic:
        1. Save predicted mesh with cortical thickness overlay to subj_dir.
        2. Save a version of predicted mesh and overlay as FreeSurfer surface and thickness file.
        3. Genetare a curvature overlay and save the pred mesh with curvature overlay as seperate mesh.
        4. Load T1 volume, interpolate intensities onto mesh, and save as T1w‐overlay mesh.

    """
    selected_type = surface_type == config.input_output.out_surface_type or \
                    config.input_output.out_surface_type == 'both'
                    
    pred_ct_map_path = f"{subj_dir}/pred_{surface_type}_ct_map.ply"
    pred_surface_path = f"{subj_dir}/pred_{surface_type}_surface.surf"
    pred_thickness_path = f"{subj_dir}/pred_{surface_type}_surface.thickness"
    pred_curvature_path = f"{subj_dir}/pred_curvature_{surface_type}_map.ply"
    pred_T1w_path = f"{subj_dir}/pred_T1w_{surface_type}_mesh.ply"
    
    # save predicted ply mesh with CTh overlay
    if selected_type: 
        if config.input_output.out_surface_mesh:
            pred_ct_map.save(pred_ct_map_path, quality_mapper, minval=1.5, maxval=5.0, quality_mapper_path=None)

        if config.input_output.out_surface_fs:
            pred_ct_map.save_as_freesurfer(pred_surface_path, pred_thickness_path)

        # save predicted ply mesh with curvature overlay
        if config.input_output.out_curvature_mesh:
            # save predicted ply mesh with curvature
            curvature_mesh = pred_ct_map.copy()
            curvature_mesh.get_curvature_mesh(save_path=pred_curvature_path)
        
        if config.input_output.out_t1w_overlay_mesh:   
            # save predicted ply mesh with T1w intensity overlay
            T1_path = f"{subj_dir}/{subject_name}.nii.gz"
            T1 = nib.load(T1_path).get_fdata()
            T1_mesh = pred_ct_map.copy()
            T1_mesh.apply_from_distance_set(distance_set=T1, smooth_thickness=False, 
                                            interpolation_method="linear")
            T1_mesh.save(pred_T1w_path, quality_mapper,
                        minval=min(T1_mesh.vertices_values),maxval=max(T1_mesh.vertices_values),
                        quality_mapper_path=None)
                                    
def compute_conform_affine(orig_affine, orig_shape, orig_zooms,
                           target_shape=(256,256,256),
                           target_zooms=(1.0,1.0,1.0),
                           target_axes=('L','I','A')):
    """
    Compute the exact affine that maps voxels in a would-be-conformed
    image (new shape, zooms, axes) directly into world space—
    without ever touching image data.

    Args:
        orig_affine (np.ndarray): Original 4x4 voxel-to-world affine.
        orig_shape (tuple): Original volume shape (x, y, z).
        orig_zooms (tuple): Original voxel sizes (x, y, z).
        target_shape (tuple, optional): Target volume shape.
        target_zooms (tuple, optional): Target voxel sizes.
        target_axes (tuple, optional): Target axis codes.

    Returns:
        np.ndarray: 4x4 affine for the target conformed grid.
    """
    # 1) figure out how the original affine is oriented
    orig_ornt = nib.orientations.io_orientation(orig_affine)
    # 2) turn target axis-codes into an orientation array
    targ_ornt = nib.orientations.axcodes2ornt(target_axes)
    # 3) the permutation+flip that takes you from orig→target
    ornt_trans = nib.orientations.ornt_transform(orig_ornt, targ_ornt)

    # 4) build the 4×4 index-space affine 
    inv_aff = nib.orientations.inv_ornt_aff(ornt_trans, orig_shape)

    # 5) compute the shape of image *after* that reorientation
    oriented_shape = tuple(orig_shape[int(axis)] for axis, _ in ornt_trans)

    # 6) figure out how many voxels of pad (or crop) on each side
    #    to go from `oriented_shape` → `target_shape`, and
    #    *center* it:
    pad = (np.array(target_shape) - np.array(oriented_shape)) / 2.0

    # 7) fold that pad into index-space transform:
    #    we want
    #       v_old = inv_aff @ (v_new - pad)
    #    which is the same as
    #       inv_aff_pad @ v_new   with
    #       inv_aff_pad[:3,3] = inv_aff[:3,3] - inv_aff[:3,:3] @ pad
    inv_aff_pad = inv_aff.copy()
    inv_aff_pad[:3, 3] = inv_aff[:3, 3] - (inv_aff[:3, :3] @ pad)

    # 8) now build final affine in one go:
    #    new_voxel → old_index → world
    new_affine = orig_affine @ inv_aff_pad

    # 9) finally, strip out the old zooms and re-apply new zooms,
    #    so that the rotational “directions” stay exactly the same,
    #    but the voxel size becomes `target_zooms`:
    #       take the 3×3 from new_affine, divide by orig_zooms, *then*
    #       multiply by target_zooms
    dircos = new_affine[:3, :3] / np.array(orig_zooms)[None, :]
    new_affine[:3, :3] = dircos * np.array(target_zooms)[None, :]

    return new_affine

def save_subject_volumes(
    subject_i_save_data, config, 
    pred_level_set=None, pred_distance_set=None, segmentation_mask=None, 
    surface_type='pial'
):
    """
    Save a copy of the segmentation mask, and optionally predicted level and distance set if flagged

    Args:
        subject_i_save_data (dict): Subject-specific metadata used for saving
            (native affine/header/shape/resolution and subject output dir).
        config (dict): Configuration object containing output flags.
        pred_level_set (numpy array): the predicted level set from DeepThickness 
        pred_distance_set (numpy array): the predicted distance set from DeepThickness
        segmentation_mask (numpy array): the segmentation_mask from LODBrain
        surface_type (str): Surface label associated with level/distance sets
            (``'pial'`` or ``'wm'``).

    Returns:
        None


    Logic:
        1. Retrieve the subject’s output directory, input level‐set &
           distance‐set paths, and T1 spatial metadata.
        2. If level‐set output is a flagged output:
            - Copy the existing level‐set file if present and save the pred_level_set
        3. If distance‐set output is a flagged output:
            - Copy the existing distance‐set file if present and save the pred_distance_set
        4. If the segmentation output is generated:
            - save the predicted segmentation volume

    """

    subj_dir = subject_i_save_data["subj_dir"]
    native_resolution = subject_i_save_data["T1_resolution"]
    native_shape = subject_i_save_data["T1_shape"]
    native_affine = subject_i_save_data["T1_affine"]
    native_header = subject_i_save_data["T1_header"]
    
    if subj_dir is not None:
        
        # Compute the conform affine to save the volume back into native T1w space
        conform_affine = compute_conform_affine(orig_affine=native_affine,
                                                orig_shape=native_shape, 
                                                orig_zooms=native_resolution)    

        selected_type = surface_type == config.input_output.out_surface_type or \
                config.input_output.out_surface_type == 'both'
        
        if selected_type:
            # Level set
            if config.input_output.out_level_set: 
                assert ((config.input_output.out_pial_surface_level_set 
                or config.input_output.out_wm_surface_level_set)), "Level set output is not flagged in config"
                
                if isinstance(pred_level_set, np.ndarray):
                    pred_nifti = nib.Nifti1Image(pred_level_set, conform_affine, native_header)
                    nib.save(pred_nifti, f"{subj_dir}/pred_{surface_type}_level_set.nii.gz")

            # Distance set
            if config.input_output.out_distance_set: 
                assert ((config.input_output.out_pial_surface_distance_set 
                or config.input_output.out_wm_surface_distance_set)), "Distance set output is not flagged in config"
                
                if isinstance(pred_distance_set, np.ndarray):
                    nib.save(
                        nib.Nifti1Image(pred_distance_set, conform_affine, native_header),
                        f"{subj_dir}/pred_{surface_type}_distance_set.nii.gz",
                    )
                
        # Segmentation Mask
        if config.input_output.out_segmentation is True:
            segmentation_mask_path = f"{subj_dir}/pred_segmentation_mask.nii.gz"
            if isinstance(segmentation_mask, np.ndarray) and not os.path.exists(segmentation_mask_path):
                nib.save(
                    nib.Nifti1Image(segmentation_mask, conform_affine, native_header),
                    segmentation_mask_path,
                )



def write_results_csv(subject_info, dataset, path_out_folder):
    """
    A function to save results for CTh and mesh metrics.

    Args:
        subject_info (dict): Dictionary of per-subject results.
        dataset (dict): Dataset dictionary used to extract dataset type.
        path_out_folder (str): The path to create a plotting dir

    Returns:
        None
    """
    # Save the subject's results to a csv
    records = [{"Subject": subj, **flatten_dict(info)} for subj, info in subject_info.items()]
    ds_key = "dataset_test_type"
    ds = dataset.get(ds_key, "unknown")
    try: 
        ds = ds[0]  
    except Exception: 
        pass  # if it's already a scalar, do nothing

    all_results_df = pd.DataFrame(records).assign(dataset_type=ds)
    all_results_df.to_csv(f"{path_out_folder}/subjects_morphology.csv", index=False)
