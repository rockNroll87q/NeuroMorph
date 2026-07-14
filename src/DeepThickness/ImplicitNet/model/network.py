"""
@authors:
- Connor Dalby, University of Glasgow
- Damiano Ferrari, Univeristy of Brescia

Script to build the native Implicitnet U-net model or shallow U-net model.

"""

import sys
sys.path.insert(0, "src/")
from os.path import join as opj
from pathlib import Path
import tensorflow as tf
import keras
from loguru import logger
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Input, Concatenate, Lambda  # type: ignore
from DeepThickness.ImplicitNet.config import Config
from DeepThickness.ImplicitNet.model.layers import (
    BottleNeck,
    UpBottleNeck,
    Plain,
    UpPlain,
    SqueezeExcitation,
)
from DeepThickness.ImplicitNet.model import losses
from LOD_Brain.src.deepthickness_tools import (
    load_lodbrain_model,
    tf_lod_post_processing
)
from DeepThickness.python_utils import load_config_from_yaml
import DeepThickness.ImplicitNet.model.activations as activations


def build_unet_model(config: Config) -> Model:
    """
    Build the 3D segmentation model.

    :param config: pydantic config object with the parameters.
    :return: keras model.
    """

    # with tf.python.keras.backend.get_graph().as_default():  # bugfix for tf 2.2.0.
    # see also https://github.com/tensorflow/tensorflow/issues/27298

    input_shape = (256,) * 3 + (
        4,
    )  # (config.input_output.in_filters,)
    inputs = tf.keras.layers.Input(input_shape)
    inputs.set_shape(
        (1,) + input_shape
    )  # Explicitly set the batch size to avoid OOM problems

    inputs_resized = inputs
    conv_blocks = {
        "BottleNeck": {"Encoder": BottleNeck, "Decoder": UpBottleNeck},
        "Plain": {"Encoder": Plain, "Decoder": UpPlain},
    }
    conv_block = conv_blocks[config.network.conv_block]

    num_filters = config.network.num_filters

    skip_x = []

    x = inputs_resized

    # Encoder
    with tf.name_scope("Encoder"):
        for r in range(
            config.network.n_identity_layers
        ):  # repeat each Convblock r times
            x = tf.keras.layers.Conv3D(
                filters=4 * num_filters[0],
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                padding="same",
                kernel_regularizer=tf.keras.regularizers.L2(
                    config.network.kernel_regularizer
                ),
                name=f"enc_{4 * num_filters[0]}_{r}",
            )(x)
        skip_x.append(x)
        for i, f in enumerate(num_filters[1:-1]):
            for r in range(config.network.n_identity_layers):
                x = conv_block["Encoder"](
                    filter_num=f,
                    dropout_rate=config.network.dropout_rate,
                    stride=(
                        1
                        if r < config.network.n_identity_layers - 1
                        else config.network.stride
                    ),
                    activation=config.network.activation_enc,
                    bn=config.network.bn,
                    groups=min(8, f),
                    kernel_regularizer=config.network.kernel_regularizer,
                    name=f"enc_cb_{f}_n_identity_{r}",
                )(
                    x
                )  # appply the stride changes
                if config.network.squeeze_excite is True:
                    x = SqueezeExcitation(
                        in_filters=f * 4,
                        out_filters=f * 4,
                        use_3d_input=True,
                        se_ratio=1,
                    )(x)
            skip_x.append(x)

        # Bridge
        x = conv_block["Encoder"](
            filter_num=num_filters[-1],
            dropout_rate=config.network.dropout_rate,
            stride=config.network.stride,
            activation=config.network.activation_enc,
            bn=config.network.bn,
            groups=min(8, num_filters[-1]),
            kernel_regularizer=config.network.kernel_regularizer,
            name=f"enc_cb_{num_filters[-1]}",
        )(x)

    num_filters.reverse()
    skip_x.reverse()
    num_layers_before_split = len(num_filters) - config.network.split_decoder_level

    # Decoder
    with tf.name_scope("Decoder"):
        for i, f in enumerate(
            num_filters[1:num_layers_before_split]
        ):  # (Conv + Skip) for number of layers before splitting into 2 outputs
            for r in range(
                config.network.n_identity_layers
            ):  # repeat each Convblock r times
                x = conv_block["Decoder"](
                    filter_num=f,
                    dropout_rate=config.network.dropout_rate,
                    stride=(
                        1
                        if r < config.network.n_identity_layers - 1
                        else config.network.stride
                    ),
                    activation=config.network.activation_dec,
                    bn=config.network.bn,
                    groups=min(8, f),
                    kernel_regularizer=config.network.kernel_regularizer,
                    name=f"dec_cb_{f}_n_identity_{r}",
                )(x)
                if config.network.squeeze_excite is True:
                    x = SqueezeExcitation(
                        in_filters=f * 4,
                        out_filters=f * 4,
                        use_3d_input=True,
                        se_ratio=1,
                    )(x)
            if config.network.skip_connection_type == "Add":
                x = tf.keras.layers.Add()([x, skip_x[i]])
            elif config.network.skip_connection_type == "Concatenate":
                x = tf.keras.layers.Concatenate()([x, skip_x[i]])

    outputs = []
    # Each output has this shape, typically (256,256,256,1)
    single_output_shape = (256,) * 3 + (1,)

    # Define output configuration names
    out_configuration = []
    count = 0
    for is_included in [
        config.input_output.out_pial_surface_level_set,
        config.input_output.out_wm_surface_level_set,
    ]:
        if is_included:
            out_configuration.append(f"level_set_{count}")
            count += 1
    count = 0
    for is_included in [
        config.input_output.out_pial_surface_distance_set,
        config.input_output.out_wm_surface_distance_set,
    ]:
        if is_included:
            out_configuration.append(f"distance_set_{count}")
            count += 1

    for out_conf in out_configuration:
        out_layer = x
        for f in num_filters[
            num_layers_before_split:
        ]:  # (ConvBlock + Skip) for per output for the rest of the Upsampling
            for r in range(
                config.network.n_identity_layers
            ):  # repeat each Convblock r times
                out_layer = conv_block["Decoder"](
                    filter_num=f,
                    dropout_rate=config.network.dropout_rate,
                    stride=(
                        1
                        if r < config.network.n_identity_layers - 1
                        else config.network.stride
                    ),
                    activation=config.network.activation_dec,
                    bn=config.network.bn,
                    groups=min(8, num_filters[-1]),
                    kernel_regularizer=config.network.kernel_regularizer,
                    name=f"dec_cb_{f}_{out_conf}_{r}",
                )(out_layer)

            matching_index = next(
                (
                    i
                    for i, tensor in enumerate(skip_x)
                    if tensor.shape[-1] == out_layer.shape[-1]
                ),
                None,
            )  # find index of corresponding skip_layer

            if config.network.skip_connection_type == "Add":
                out_layer = tf.keras.layers.Add()([out_layer, skip_x[matching_index]])
            elif config.network.skip_connection_type == "Concatenate":
                out_layer = tf.keras.layers.Concatenate()(
                    [out_layer, skip_x[matching_index]]
                )

        out_layer = tf.keras.layers.Conv3D(
            filters=4 * num_filters[-1],
            kernel_size=(3, 3, 3),
            padding="same",
            kernel_initializer=config.network.kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.L2(
                config.network.kernel_regularizer
            ),
            name=f"out_cb_{4 * num_filters[-1]}_{out_conf}",
        )(out_layer)

        activation = (
            activations.leaky_clip(leaky_min=-5.0, leaky_max=5.0)
            if out_conf.startswith("level_set_")
            else activations.leaky_clip(leaky_min=1.0, leaky_max=5.0)
        )
        out_layer = tf.keras.layers.Conv3D(
            filters=1,
            kernel_size=(1, 1, 1),
            activation=activation,
            kernel_initializer=config.network.kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.L2(
                config.network.kernel_regularizer
            ),
            name=out_conf,
        )(out_layer)
        out_layer.set_shape(
            (1,) + single_output_shape
        )  # Explicitly set the batch size to avoid OOM problems
        outputs.append(out_layer)
    
    model_name = 'pial_surface_model' if config.input_output.out_pial_surface_level_set is True else 'wm_surface_model'
    
    return Model(inputs, outputs, name=model_name)

def build_implicitnet_model(config):
    """
    Build and compile the Implicitnet model.
    This model takes T1w, WM Mask, GM Mask and Segmentation Mask as input and generates a
    predicted level set and distance set as output.

    Args:
    - config (dict): dictionary containing configuration parameters

    Returns:
    - model (keras.Model): a compiled model with previous training weights loaded (optional)

    Logic:
    - Build the model
    - Create a lossess and metrics dictionary for level set, distance set and combined
    - Compile the model with the desired losses, optimiser and metrics
    - If provided, load in the pre-trained weights
    - Return the model

    """

    # Build architecture
    model = build_unet_model(config)

    # Define Model Metrics and Losses
    metrics = {}
    loss = {}

    if (
        config.input_output.out_pial_surface_level_set
        ^ config.input_output.out_wm_surface_level_set
    ):
        metrics["level_set_0"] = list(losses.metrics_level_set.values())
        loss["level_set_0"] = losses.create_level_set_loss(config)
    if (
        config.input_output.out_pial_surface_distance_set
        ^ config.input_output.out_wm_surface_distance_set
    ):
        metrics["distance_set_0"] = list(losses.metrics_distance_set.values())
        loss["distance_set_0"] = losses.losses_distance_set[
            config.losses.distance_set_0_loss
        ]

    # Compile Model
    model.compile(loss=loss, metrics=metrics, weighted_metrics=[])

    # load initial weights if testing only
    initial_weights_path = config.network.saved_weights_path
    if initial_weights_path is not None:
        model.load_weights(initial_weights_path)
        logger.info(f"Loaded weights from path: {initial_weights_path}")

    return model

def load_pretrained_model(model_surface_type):

    """
    Build an Implicitnet Model based on a previously trained model's configuration 
    and model training weights.
    
    Args:
    - model_surface_typ (str): which kind of surface the model is predicting

    Returns:
    - model (keras.Model): a compiled model that incorporates pretrained weights

    Logic:
        - Load the config yaml from a previous training to ensure same model architecture 
          and hyperparameters 
        - Build the Implicinet model, compile and load pretrained weights based on the config
        - Return model 
        
    """
    
    
    if model_surface_type == 'pial':
        
        pial_config = load_config_from_yaml(Path("src/DeepThickness/weights/pial_config.yaml"))
        pial_config.input_output.out_wm_surface_distance_set=False
        pial_config.input_output.out_wm_surface_level_set=False
        pial_config.network.saved_weights_path = (
            "src/DeepThickness/weights/pial_weights_unet.h5"
        )
        logger.info("Building Pial Surface Model")
        pial_surface_model = build_implicitnet_model(pial_config)
        return pial_surface_model
    
    if model_surface_type == 'wm':
        
        wm_config = load_config_from_yaml(Path("src/DeepThickness/weights/wm_config.yaml"))
        wm_config.input_output.out_pial_surface_distance_set=False
        wm_config.input_output.out_pial_surface_level_set=False
        wm_config.network.saved_weights_path = (
            "src/DeepThickness/weights/wm_weights_unet.h5"
        )
        logger.info("Building WM Surface Model")
        wm_surface_model = build_implicitnet_model(wm_config)
        return wm_surface_model

def build_deepthickness_model(config, training_model=None, evaluate_model=None):
    
    """
    Build the DeepThickness Model which can generate a level set and distance set for 
    either the pial surface or wm surface or both
    
    Args:
    - config (dict): dictionary containing configuration parameters
    - training_model (keras.Model): A model that has just finished training. 
                                     This function then acts as a wrapper to store the model tuple
                                     outputs into a dict for 'parralel_process_ds()'  
    - evaluate_model (str / optional): parameter to build only a 'wm' or 'pial' surface
                                       type model, bypassing config 

    Returns:
    - model (keras.Model): a model that incorporates two smaller pretrained Implicitnet models

    Logic:
        - Define input as the output of LODBrain and T1w (4 channels)
        - Seperately build, compile and load weights for the pial and wm surface model.
        - When evaluating the model we want to only build the pial or wm model seperately. 
          To leave the experiment config intact, we use this parameter to specifically build the
          wm or pial surface model (bypassing the other) when building a model for evaluation.
        - Make a prediction for each model 
        - The output of each model is stored in an overall outputs dictionary
        - Return the DeepThickness Model

    """
    
    deepthickness_outputs = {}
    deepthickness_inputs = Input(shape=(256, 256, 256, 4), name="deepthickness_input")   

    assert (
        (config.input_output.out_pial_surface_level_set
        and config.input_output.out_pial_surface_distance_set)
        or (config.input_output.out_wm_surface_level_set
            and config.input_output.out_wm_surface_distance_set)
    ), (
        "config.input_output.out_surface_level_set and "
        "config.input_output.out_surface_distance_set must be set "
        "for either the wm or pial surface or both"
    )

    if (config.input_output.out_pial_surface_level_set and config.input_output.out_pial_surface_distance_set) \
        and evaluate_model != "wm":
        
        if training_model is not None:
            pial_surface_model = training_model
        else:
            pial_surface_model = load_pretrained_model(model_surface_type='pial')
            
        pial_level_set, pial_distance_set = pial_surface_model(deepthickness_inputs)
        deepthickness_outputs['pial_level_set'] = pial_level_set
        deepthickness_outputs['pial_distance_set'] = pial_distance_set
        
    if config.input_output.out_wm_surface_level_set and config.input_output.out_wm_surface_distance_set \
        and evaluate_model != "pial":

        if training_model is not None:
            wm_surface_model = training_model
        else:
            wm_surface_model = load_pretrained_model(model_surface_type='wm')
        
        wm_level_set, wm_distance_set = wm_surface_model(deepthickness_inputs)
        deepthickness_outputs['wm_level_set'] = wm_level_set
        deepthickness_outputs['wm_distance_set'] = wm_distance_set

    deepthickness_model = Model(
    inputs=deepthickness_inputs, outputs=deepthickness_outputs, name="deepthickness_model")

    return deepthickness_model
    
def build_combined_model(config):
    """
    Build a tensorflow model that encapsulates the ImplicitNet model and LODBrain model.
    The combined model can be used to make a multi-model prediction with a single
    combined_model.predict() call.

    Args:
    - config (dict): dictionary containing configuration parameters

    Returns:
    - model (keras.Model): a model that incorporates two smaller models

    Logic:
        - Seperately build, compile and load weights for the LODBrain and DeepThickness Model.
        - Define input as T1w image
        - Make a prediction using LODBrain from input
        - LODBrain output is the WM, GM, and Segmentation Probability maps
        - If flagged, LODBrain output will also include a segmentation mask 
        - Apply post-processing on LODBrain outputs for suitable input into ImplicitNet
        - Create the DeepThickness input (T1w + LODBrain Outputs)
        - Make a prediction using DeepThickness from input
        - The output of DeepThickness will be a level and distance set for pial surface
          and/or wm pial surface 
        - Store all the outputs of the model as a dictionary output
        
    """

    model_outputs = {}
    
    # Build, compile and load weights of preivous models
    lodbrain_model = load_lodbrain_model(config)
    deepthickness_model = build_deepthickness_model(config)

    # Input to model
    t1w_input = Input(shape=(256, 256, 256, 1), name="t1w_input")

    # LODBrain prediciton + post_processing
    lod_output = lodbrain_model(t1w_input)
    
    if config.input_output.out_segmentation is True:
        lod_output, segmentation_mask = lod_output
    
    lod_output_pp = Lambda(tf_lod_post_processing, name="lod_postproc")(lod_output)

    # DeepThickness Input
    deepthickness_inputs = Concatenate(axis=-1, name="t1w_plus_lod")(
        [t1w_input, lod_output_pp]
    )
    
    # DeepThickness prediction
    model_outputs = deepthickness_model(deepthickness_inputs)

    # Add segmentation to outputs if flagged
    if config.input_output.out_segmentation is True:
        model_outputs['segmentation_mask'] = segmentation_mask
    
    # Building the overall combined model
    combined_model = Model(
        inputs=t1w_input, outputs=model_outputs, name="combined_model"
    )

    return combined_model

def build_inference_model(model_type, config):
    """
    A wrapper function to build and return the target model based on the desired model type.

    Args:
    - config (dict): dictionary containing configuration parameters used in building models

    Returns:
    - model (keras.Model): a tf keras model with inputs/outputs based on model_type parameter.

    """

    if model_type == "Combined_Model":
        
        logger.info("Building Combined Model (LODBrain + DeepThickness)")
        model = build_combined_model(config)

    elif model_type == "DeepThickness":

        logger.info("Building DeepThickness Model")
        model = build_deepthickness_model(config)

    return model

