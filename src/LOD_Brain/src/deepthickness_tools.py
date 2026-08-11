import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
from loguru import logger
from nibabel.processing import conform
from tensorflow.keras.models import (
    Model,
    load_model,  # type: ignore
)

from LOD_Brain.src.LOD_Brain.model import layers, losses


class GaussianSmoothing3D(tf.keras.layers.Layer):
    def __init__(self, sigma=0.5, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.kernel_size = kernel_size

    def build(self, input_shape):
        channels = input_shape[-1]

        # Build Gaussian kernel
        coords = (
            tf.range(self.kernel_size, dtype=tf.float32) - (self.kernel_size - 1) / 2.0
        )
        x, y, z = tf.meshgrid(coords, coords, coords, indexing="ij")
        squared_distance = x**2 + y**2 + z**2

        kernel = tf.exp(-squared_distance / (2.0 * self.sigma**2))
        kernel = kernel / tf.reduce_sum(kernel)

        # Reshape kernel to (k, k, k, in_channels_per_group=1, out_channels=channels)
        kernel = tf.reshape(
            kernel, [self.kernel_size, self.kernel_size, self.kernel_size, 1, 1]
        )
        kernel = tf.tile(kernel, [1, 1, 1, 1, channels])  # 👈 correct order!

        # Convert to numpy only now
        self.kernel = tf.constant(kernel, dtype=self.dtype)

        # Build Conv3D
        self.conv = tf.keras.layers.Conv3D(
            filters=channels,
            kernel_size=self.kernel_size,
            padding="same",
            groups=channels,
            use_bias=False,
            trainable=False,
        )
        self.conv.build(input_shape)
        self.conv.set_weights([self.kernel.numpy()])

        super().build(input_shape)

    def call(self, inputs):
        return self.conv(inputs)

def remap_classes(x):
    # I want to create a new model for DeepThickness that outputs a different #classes
    background_classes = [0, 5, 6, 7]  # bakground, cerebellum, brainstem, CSF          -> 0
    gm_classes = [1]  # gray matter                               -> 1
    wm_classes = [2, 3, 4]  # basic ganglia, white matter, ventricles   -> 2
    # Sum probabilities for each new class
    background = tf.reduce_sum(
        tf.gather(x, background_classes, axis=-1), axis=-1, keepdims=True
    )
    gray_matter = tf.reduce_sum(
        tf.gather(x, gm_classes, axis=-1), axis=-1, keepdims=True
    )
    white_matter = tf.reduce_sum(
        tf.gather(x, wm_classes, axis=-1), axis=-1, keepdims=True
    )

    # Concatenate them in the desired order
    return tf.concat([background, gray_matter, white_matter], axis=-1)

def load_lodbrain_model(config):
    """
    Search through the folders in 'Path_in_models' and returns the model path.

    :param model_keyword: keywork of the experiment
    :return model
    """

    # Restore the (best) model weights
    # (nomenclature: "model.{epoch:02d}-{val_compute_per_channel_dice:.2f}.h5")
    last_weights_file = Path(__file__).resolve().parents[1] / 'trained_model' / 'LODBrain_plus_weights.h5'
    assert os.path.exists(
        last_weights_file
    ), f"Model {last_weights_file} not found! Please check."

    custom_objects = {
        "BottleNeck": layers.BottleNeck,
        "UpBottleNeck": layers.UpBottleNeck,
        "Plain": layers.Plain,
        "UpPlain": layers.UpPlain,
    }
    custom_objects.update(losses.losses_dict)
    custom_objects.update(losses.metrics_dict)
    custom_objects["tversky_metric"] = custom_objects.pop("tversky_custom_metric")
    
    logger.info("Building LODBrain Model")
    model = load_model(
        last_weights_file, custom_objects=custom_objects
    )  # , compile=False)

    # Build new model
    inputs = model.input
    outputs = tf.keras.layers.Lambda(remap_classes, name="8_to_3_classes")(model.output)
    smoothed_output = GaussianSmoothing3D(
        sigma=0.5, kernel_size=5, name="gaussian_smoothing"
    )(outputs)
    model_outputs = smoothed_output
    
    if config.input_output.out_segmentation is True:
        y_test_prob_map = tf.squeeze(model.output)  
        y_test_pred = tf.argmax(y_test_prob_map, axis=-1)
        y_test_pred = tf.cast(y_test_pred, dtype=tf.uint8) 
        model_outputs = (smoothed_output, y_test_pred)


    model_n_classes = Model(inputs=inputs, outputs=model_outputs)

    return model_n_classes

def get_largest_connected_component(mask, structure=None):
    """Function to get the largest connected component for a given input.
    :param mask: a 2d or 3d label map of boolean type.
    :param structure: numpy array defining the connectivity.
    """
    components, n_components = scipy.ndimage.label(mask, structure)
    return (
        components == np.argmax(np.bincount(components.flat)[1:]) + 1
        if n_components > 0
        else mask.copy()
    )


def seg_keep_only_biggest_components(seg_in, labels=None):
    """
    Keep only the biggest component of each class
    (no 'background' and 'ventricles') and the mask as a whole.
    """

    if labels is None:
        labels = np.unique(seg_in)

    # Keep only the biggest component of the seg mask
    seg_out = seg_in.copy()
    seg_out[~get_largest_connected_component(seg_out > 0)] = 0

    # For each label, keep only the biggest component
    for i_label in labels:
        if i_label == 0 or i_label == 4:  # 'background' and 'ventricles' aren't to scan
            continue
        seg_out[
            (~get_largest_connected_component(seg_in == i_label)) & (seg_in == i_label)
        ] = 0

    return seg_out


def prob_keep_only_biggest_components(prob_in, threshold=0.25):
    """
    Keep only the biggest component of each class (no 'background' and 'ventricles') and the mask as a whole.
    :param prob_in: probability map, shape = (x,y,z,n_classes)
    :param threshold: for the biggest map as a whole
    :return prob_out probability map with only biggest components, shape = (x,y,z,n_classes)
    """

    prob_out = prob_in.copy()

    # For each label, keep only the biggest component
    for i_label in range(1, prob_in.shape[-1]):
        if i_label == 4:  # 'ventricles' isn't to scan
            continue
        mask_out = prob_out[..., i_label] > threshold
        prob_out[(~get_largest_connected_component(mask_out)) & (mask_out), i_label] = 0

    # Keep only the biggest component of the output seg mask
    mask_out = np.argmax(prob_out, axis=-1)
    mask_out = seg_keep_only_biggest_components(mask_out, labels=[1, 3, 5, 6]) > 0
    prob_out[~(mask_out > 0), :] = 0

    # Copy back the probabilities for 'background'
    prob_out[..., 0] = prob_in[..., 0]

    return prob_out


def findListOfAnatomical(path_in, identifier="nii.gz"):
    "Conduct a recursive search from path_in to get a list of files ending with identifier"
    all_anat = []
    for root, _, files in os.walk(path_in):
        for i_file in files:
            if i_file.endswith(identifier):
                all_anat.append(root + "/" + i_file)

    all_anat = sorted(list(np.unique(all_anat)))
    all_anat = [i for i in all_anat if "/._" not in i]

    # Ensure at least one matching file was found
    assert all_anat, f"No files with identifier '{identifier}' found in '{path_in}'."

    return all_anat



def pad_volume(vol_in, pad_size=(256, 256, 256)):
    """Pad the input volume 'vol_in' of 'pad_size'."""

    padding_needed = tuple(
        [(pad_size[i] - vol_in.shape[i]) for i in range(len(vol_in.shape))]
    )
    padding_needed = tuple(
        [tuple([int(np.floor(i / 2)), int(np.ceil(i / 2))]) for i in padding_needed]
    )
    vol_out = np.pad(vol_in, padding_needed, "constant", constant_values=0)

    return vol_out, padding_needed

def conform_image(input_data, out_shape=None, out_orientation=None, out_resolution=None):

    i_X = input_data if isinstance(input_data, nib.nifti1.Nifti1Image) else nib.load(input_data)

    out_shape = (256,256,256) if out_shape is None else out_shape
    out_orientation = ("L","I","A") if out_orientation is None else out_orientation
    out_resolution = [1.,1.,1.] if out_resolution is None else out_resolution

    vol_shape = i_X.shape
    vol_orientation = nib.aff2axcodes(i_X.affine)
    vol_resolution = i_X.header['pixdim'][1:4]
    

    if not (vol_shape == out_shape and vol_orientation == out_orientation \
        and np.all(vol_resolution == out_resolution)):
        i_X = conform(i_X, out_shape=out_shape, voxel_size=out_resolution, orientation=out_orientation)
    
    return i_X


def lodbrain_model_post_processing(prob_maps_i):
    """Prepare the output of LODBrain prediction to be suitable for input to ImplicitNet model

    Args:
        prob_maps_i (ndarray): Per-voxel probability maps for segmentation classes

    Returns:
        ndarray: A volume where the three channels correspond to GM prob map, WM prob map and segmentation map
    """

    # Keep only 1 component per class
    prob_maps_i = prob_keep_only_biggest_components(prob_maps_i)  # SAME AS BEFORE

    # compute the argmag (hard-thresholding) of the proabability map to obtain the labelled volume
    segmentation = np.argmax(prob_maps_i, axis=-1).astype(dtype="uint8")

    # Keep only: gm_map, wm_map, segmented_volume (3 labels)
    gm_map = prob_maps_i[..., 1]
    wm_map = prob_maps_i[..., 2]

    # Adjust the wm_map to be coherent with segmented_volume
    wm_map[~scipy.ndimage.binary_dilation(segmentation == 2)] = 0

    lodbrain_input_i = np.stack([gm_map, wm_map, segmentation], axis=-1)

    return lodbrain_input_i


def tf_lod_post_processing(prob_tensor):
    """
    Wrap the native lodbrain_model_post_processing so it runs in the TF graph with NumPy arrays automatically.
    """
    # Use numpy_function instead of py_function:
    processed = tf.numpy_function(
        func=lodbrain_model_post_processing,  # this takes/returns NumPy arrays
        inp=[prob_tensor],  # list of Tensors → NumPy arrays
        Tout=tf.float32,  # output dtype
    )
    # restore the static shape info so TF knows it’s still (batch,256,256,256,3)
    processed.set_shape([None, 256, 256, 256, 3])
    return processed

def find_delineating_subjects(all_paths):
    """
    Given a list of file‐paths, try first to use each file’s basename
    (minus .nii.gz) as the subject. If those are unique, just return them.
    Otherwise roll up the directory tree until you find a level where
    each path lives in a uniquely‐named folder.
    """
    paths = [Path(p) for p in all_paths]

    # 1) try filenames themselves (minus double extensions)  
    base_names = [Path(p.stem).stem for p in paths]
    if len(base_names) == len(set(base_names)):
        return base_names

    # 2) otherwise search parent folders
    max_depth = max(len(p.parents) for p in paths)
    for depth in range(max_depth):
        candidates = [p.parents[depth].name for p in paths]
        if len(candidates) == len(set(candidates)):
            return candidates

    raise AssertionError(
        "Could not find a directory level where all subjects are unique."
    )

def generate_testing_csv(all_anat, csv_output_dir=None):
    """
    To create a ds_test we rely on the function prepareDataset()
    That function requires us to provide a csv with expected column names
    (file_map) even if the files themselves don't exist. Therefore,
    we take the list of T1w file paths and create a csv that includes all the T1w paths
    and paths for the other relevant data that could be loaded as part of the ds_test.

    Args:
        - all_anat (list): A list of all the T1w image paths
        - csv_out_dir (str): A path to save the output csv

    Returns:
        - csv_file_path: A path to the csv which can be used to assign to
                         config.data.Path_in_csv / config.data.Filename_csv

    Logic:
        Concisely build a CSV of T1 paths + derived paths by iterating over a
        mapping of column names to filename patterns. The file_map
        columns are set in the output csv but many can be ignored not needed
        for the inference type. E.g. aparc paths are needed for regional_predictions
        and the FS filepaths are needed for model_evaluation_type and/or mesh_evaluation.


    """
    
    # first find unique subject names    
    subject_names = find_delineating_subjects(all_anat)

    # map each output‐column to its filename under ct_dataset/<subject>/
    file_map = {
        "segmentation_gm": "GM_mask_LOD_Brain.nii.gz",
        "segmentation_wm": "WM_mask_LOD_Brain.nii.gz",
        "segmentation": "segmentation_LOD_Brain.nii.gz",
        "pial_surface_level_set": "FS_dpial.ribbon_NS.mgz",
        "wm_surface_level_set": "FS_dwhite.ribbon_NS.mgz",
        "pial_surface_distance_set": "pial_distance_set_NS.nii.gz",
        "wm_surface_distance_set": "wm_distance_set_NS.nii.gz",
        "lh_pial_surface": "lh.pial.native",
        "rh_pial_surface": "rh.pial.native",
        "lh_wm_surface": "lh.white.native",
        "rh_wm_surface": "rh.white.native",
        "lh_thickness": "lh.thickness",
        "rh_thickness": "rh.thickness",
        "orig": "orig_NS.mgz",
        "aparc": "FS_aparc+aseg_NS.mgz",
    }


    rows = []
    for t1_path_str, subject in zip(all_anat, subject_names):
        t1_path = Path(t1_path_str)
        # assume your ct_dataset mirror lives in a folder named after the subject
        ct_root = t1_path.parent / subject

        row = {
            "T1": str(t1_path),
            "subject": subject,
            "set": "test",
            "database": subject[:8],
            "dataset_type": "unknown",
        }
        row.update({col: str(ct_root / fname) for col, fname in file_map.items()})
        rows.append(row)

    # build DataFrame; pandas aligns columns by their keys
    df = pd.DataFrame(rows)

    # ensure we get the exact column order you want
    desired_order = (
        ["T1"] + list(file_map.keys()) + ["subject", "set", "database", "dataset_type"]
    )
    df = df[desired_order]

    # prepare output directory
    out_dir = Path(csv_output_dir) if csv_output_dir else Path(all_anat[0]).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # write CSV and return its path
    csv_file_path = out_dir / "subjects_input.csv"
    df.to_csv(csv_file_path, index=False)
    logger.info(f"Inference CSV file created at {csv_file_path}")

    return str(csv_file_path)



def adapt_existing_csv(existing_csv_path, csv_output_dir=None):
    """
    Build DeepThickness_testing.csv using an existing CSV. 
    Ensures 'T1' and 'subject' columns exist, and only adds missing columns.

    Args:
        existing_csv_path (str|Path): Path to existing CSV containing at least 'T1' and 'Subject'
        csv_output_dir (str|Path|None): Directory to save output CSV (defaults to directory of input CSV)

    Returns:
        str: Path to the generated CSV
    """
    file_map = {
        "segmentation_gm": "GM_mask_LOD_Brain.nii.gz",
        "segmentation_wm": "WM_mask_LOD_Brain.nii.gz",
        "segmentation": "segmentation_LOD_Brain.nii.gz",
        "pial_surface_level_set": "FS_dpial.ribbon_NS.mgz",
        "wm_surface_level_set": "FS_dwhite.ribbon_NS.mgz",
        "pial_surface_distance_set": "pial_distance_set_NS.nii.gz",
        "wm_surface_distance_set": "wm_distance_set_NS.nii.gz",
        "lh_pial_surface": "lh.pial.native",
        "rh_pial_surface": "rh.pial.native",
        "lh_wm_surface": "lh.white.native",
        "rh_wm_surface": "rh.white.native",
        "lh_thickness": "lh.thickness",
        "rh_thickness": "rh.thickness",
        "orig": "orig_NS.mgz",
        "aparc": "FS_aparc+aseg_NS.mgz",
    }
    desired_order = ["T1"] + list(file_map.keys()) + ["subject", "set", "database", "dataset_type"]

    # Load CSV
    df = pd.read_csv(existing_csv_path)

    # Normalize colnames (case-insensitive check)
    col_map = {c.lower(): c for c in df.columns}
    assert "t1" in col_map, "Existing CSV must contain 'T1' column"
    assert "subject" in col_map, "Existing CSV must contain 'Subject' column"

    # Standardize column names
    df = df.rename(columns={col_map["t1"]: "T1", col_map["subject"]: "subject"})

    # Fill missing required columns
    for idx, row in df.iterrows():
        t1_path = Path(str(row["T1"]))
        subject = row["subject"]

        # Normalize subject: if numeric → subj_<number>
        if isinstance(subject, (int, float)) and not pd.isna(subject):
            subject = f"subj_{int(subject)}"
            df.at[idx, "subject"] = subject

        ct_root = t1_path.parent / subject

        # Add file_map paths only if missing
        for col, fname in file_map.items():
            if col not in df.columns or pd.isna(row.get(col)):
                df.at[idx, col] = str(ct_root / fname)

        # Add metadata columns if missing
        if "set" not in df.columns:
            df["set"] = "test"
        if "database" not in df.columns:
            df["database"] = df["subject"].astype(str).str[:8]
        if "dataset_type" not in df.columns:
            df["dataset_type"] = "unknown"

    # Ensure desired order but keep extra user columns at the end
    for col in desired_order:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[[c for c in desired_order if c in df.columns] + [c for c in df.columns if c not in desired_order]]

    # Prepare output directory
    out_dir = Path(csv_output_dir) if csv_output_dir else Path(existing_csv_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save
    csv_file_path = out_dir / "subjects_input.csv"
    df.to_csv(csv_file_path, index=False)
    logger.info(f"Inference CSV file created at {csv_file_path}")
    return str(csv_file_path)
