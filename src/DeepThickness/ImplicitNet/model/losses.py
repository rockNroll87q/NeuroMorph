#!/usr/bin/env python3
"""
Authors:
* Connor Dalby, University of Glasgow
* Damiano Ferrari

Various loss and metrics functions to be passed to the model for training and evaluation.
"""

import tensorflow as tf
from tensorflow.keras import backend as K


def non_capped_mae(cap_high = 5, cap_low = -5):
    """
    This metric compute the mean absolute error just in voxel where y_true is
    not equal to cap_high or cap_low. This is useful expecially for distance set
    that have 90%+ voxel which value is zero.
    """
    def non_cap_mae(y_true, y_pred):
        # Flatten just for simplification
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        indices = K.flatten(tf.where(tf.logical_and(tf.not_equal(y_true, tf.ones_like(y_true)*cap_high), \
            tf.not_equal(y_true, tf.ones_like(y_true)*cap_low))))
        return tf.keras.metrics.mean_absolute_error(tf.gather(y_true, indices), tf.gather(y_pred, indices))
    return non_cap_mae

def weighted_mae(cap_values:list=None, cap_weight = 0.03, non_cap_weight = 1.0):
    """
    This loss function gives `non_cap_weight` to errors where the voxels in
    `y_true` are different then cap_values and `cap_weight` to errors of the
    capped voxels. This is useful to balance the high number of zero values in
    distance set (at about 97% of the values) that are also useless. The loss
    is a mean absolute error. 

    ## Parameters
    - cap_values:list
        Contains a list of the values you identify as capped. Typically is [0.]
        for distance set and [-5., 5.] for level set.
    - cap_weight:float
        Is the weight you want to assign to the capped values in the weighted
        loss.
    - non_cap_weight:float
        Is the weight you want to assign to the non capped values in the
        weighted loss. Typically non_cap_weight >> cap_weight
    """
    if cap_values is None:
        cap_values = [0.]
    def weighted_mae(y_true, y_pred):
        # Cast y_true and y_pred to float32
        y_true = tf.cast(K.flatten(y_true), tf.float32)
        y_pred = tf.cast(K.flatten(y_pred), tf.float32)

        # y_true_capped is a mask where it is True where there is a cap value, False otherwise
        y_true_capped = tf.zeros_like(y_true, dtype=bool)
        for cap_value in cap_values:
            y_true_capped = tf.math.logical_or(y_true_capped, y_true == \
                tf.cast(tf.ones_like(y_true)*cap_value, tf.float32))

        size = tf.cast(tf.size(y_true), dtype=tf.float32)
        number_of_capped_values = tf.math.count_nonzero(y_true_capped, dtype=tf.float32)
        number_of_non_capped_values = size - number_of_capped_values

        # Ensure weights are in float32
        weights = K.switch(y_true_capped, tf.cast(tf.ones_like(y_true)*cap_weight, tf.float32),
                           tf.cast(tf.ones_like(y_true)*non_cap_weight, tf.float32))

        loss = tf.keras.losses.MeanAbsoluteError()
        # 'scale' is used to have as loss the typical weighted average formula
        scale = size / (number_of_capped_values * cap_weight + number_of_non_capped_values * non_cap_weight)
        return scale * loss(tf.reshape(y_true, (-1, 1)), tf.reshape(y_pred, (-1, 1)), sample_weight=weights)
    return weighted_mae

def weighted_mse(cap_values=None, cap_weight = 0.03, non_cap_weight = 1.):
    """
    This loss function gives `non_cap_weight` to errors where the voxels in
    `y_true` are different then cap_values and `cap_weight` to errors of the
    capped voxels. This is useful to balance the high number of zero values in
    distance set (at about 97% of the values) that are also useless. The loss
    is a mean squared error. 

    ## Parameters
    - cap_values:list
        Contains a list of the values you identify as capped. Typically is [0.]
        for distance set and [-5., 5.] for level set.
    - cap_weight:float
        Is the weight you want to assign to the capped values in the weighted
        loss.
    - non_cap_weight:float
        Is the weight you want to assign to the non capped values in the
        weighted loss. Typically non_cap_weight >> cap_weight
    """
    if cap_values is None:
        cap_values = [0.]
    def weighted_mse(y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        # y_true_capped is a mask where it is True where there is a cap value, False otherwsie
        y_true_capped = tf.zeros_like(y_true, dtype=bool)
        for cap_value in cap_values:
            y_true_capped = tf.math.logical_or(y_true_capped, y_true == tf.ones_like(y_true)*cap_value)
        size = tf.cast(tf.size(y_true), dtype=tf.float32)
        number_of_capped_values = tf.math.count_nonzero(y_true_capped, dtype=tf.float32)
        number_of_non_capped_values = size - number_of_capped_values
        weights = K.switch(y_true_capped, tf.ones_like(y_true)*cap_weight, tf.ones_like(y_true)*non_cap_weight)
        loss = tf.keras.losses.MeanSquaredError()
        # `scale` is used to have as loss the typical weighted average formula
        scale = size / (number_of_capped_values * cap_weight + number_of_non_capped_values * non_cap_weight)
        return scale * loss(tf.reshape(y_true, (-1, 1)), tf.reshape(y_pred, (-1, 1)), sample_weight=weights)
    return weighted_mse

def weighted_huber(cap_values=None, cap_weight = 0.03, non_cap_weight = 1.):
    """
    This loss function gives `non_cap_weight` to errors where the voxels in
    `y_true` are different then cap_values and `cap_weight` to errors of the
    capped voxels. This is useful to balance the high number of zero values in
    distance set (at about 97% of the values) that are also useless. The loss
    is huber.

    ## Parameters
    - cap_values:list
        Contains a list of the values you identify as capped. Typically is [0.]
        for distance set and [-5., 5.] for level set.
    - cap_weight:float
        Is the weight you want to assign to the capped values in the weighted
        loss.
    - non_cap_weight:float
        Is the weight you want to assign to the non capped values in the
        weighted loss. Typically non_cap_weight >> cap_weight
    """
    if cap_values is None:
        cap_values = [0.]
        
    def weighted_huber(y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        # y_true_capped is a mask where it is True where there is a cap value, False otherwsie
        y_true_capped = tf.zeros_like(y_true, dtype=bool)
        for cap_value in cap_values:
            y_true_capped = tf.math.logical_or(y_true_capped, y_true == tf.ones_like(y_true)*cap_value)
        size = tf.cast(tf.size(y_true), dtype=tf.float32)
        number_of_capped_values = tf.math.count_nonzero(y_true_capped, dtype=tf.float32)
        number_of_non_capped_values = size - number_of_capped_values
        weights = K.switch(y_true_capped, tf.ones_like(y_true)*cap_weight, tf.ones_like(y_true)*non_cap_weight)
        loss = tf.keras.losses.Huber()
        # `scale` is used to have as loss the typical weighted average formula
        scale = size / (number_of_capped_values * cap_weight + number_of_non_capped_values * non_cap_weight)
        return scale * loss(tf.reshape(y_true, (-1, 1)), tf.reshape(y_pred, (-1, 1)), sample_weight=weights)
    return weighted_huber

def bounded_mae(cap_high=1, cap_low=0):
    """
    This metric computes the mean absolute error (MAE) only for values in `y_true` and `y_pred`
    that are within the range `[cap_low, cap_high]`.
    """
    def filtered_mae(y_true, y_pred):

        y_true = tf.cast(K.flatten(y_true), tf.float32)
        y_pred = tf.cast(K.flatten(y_pred), tf.float32)

        # Create a mask for values within the specified range
        mask = tf.logical_and(y_true >= cap_low, y_true <= cap_high)

        # Get the indices of the values that satisfy the mask
        indices = tf.where(mask)

        # Gather the filtered values for y_true and y_pred
        y_true_filtered = tf.gather(y_true, indices)
        y_pred_filtered = tf.gather(y_pred, indices)

        # Compute MAE only on the filtered values
        return tf.keras.metrics.mean_absolute_error(y_true_filtered, y_pred_filtered)
    
    return filtered_mae
 
def ranged_weighted_mae(focus_range=None, gradient_type='slope', max_weight=1.0, kl_weight=0.0):
    """
    Computes a weighted MAE with a custom weighting scheme:
    - Base weights are determined by 1 - |y_true|/max(|y_true|)
    - We enforce a minimum weight of 0.01.
    - Voxels within focus_range are given a weight of 1.0.
    - If gradient_type == 'step', certain voxels get a fixed weight of 0.5.
    """
    if focus_range is None:
        focus_range = [0.0, 1.0]

    def weighted_mae(y_true, y_pred):
        # Flatten and cast to float32
        y_true = tf.cast(K.flatten(y_true), tf.float32)
        y_pred = tf.cast(K.flatten(y_pred), tf.float32)

        # Compute |y_true| and its max
        abs_y_true = tf.abs(y_true)
        max_abs_y_true = tf.reduce_max(abs_y_true)
        max_abs_y_true = tf.maximum(max_abs_y_true, 1e-10)

        # Base weights: 1 - (|y_true| / max_abs_y_true)
        weights = 1.0 - (abs_y_true / max_abs_y_true)

        # Enforce minimum base weight of 0.01
        base_weight = 0.01
        non_cap_weight = tf.maximum(weights, tf.fill(tf.shape(y_true), base_weight))

        # Set weights to 1.0 for voxels within the focus range
        condition_focus = (y_true >= focus_range[0]) & (y_true <= focus_range[1])
        non_cap_weight = tf.where(condition_focus, max_weight, non_cap_weight)

        # If gradient_type == 'step', override certain weights to 0.5
        if gradient_type == 'step':
            # If weights are in [non_cap_weight, 1], set them to 0.5
            # Note: This condition as stated might need clarification.
            # We'll just apply it if the original weights (not the modified ones)
            # are between non_cap_weight and 1.
            condition_step = (weights >= non_cap_weight) & (weights <= 1.0)
            non_cap_weight = tf.where(condition_step, 0.5, non_cap_weight)

        # Compute MeanAbsoluteError with sample_weight
        loss_fn = tf.keras.losses.MeanAbsoluteError()
        loss_value = loss_fn(tf.reshape(y_true, (-1, 1)),
                             tf.reshape(y_pred, (-1, 1)),
                             sample_weight=tf.reshape(non_cap_weight, [-1]))

        size = tf.cast(tf.size(y_true), dtype=tf.float32)
        number_of_focus_values = tf.reduce_sum(tf.cast(condition_focus, tf.float32))
        number_of_non_focus_values = size - number_of_focus_values

        scale = size / (number_of_focus_values * max_weight + number_of_non_focus_values * base_weight)

        voxel_loss = scale * loss_value

        # Compute KL divergence
        """        
        KL divergence between predicted and ground truth voxel distributions.
        """

        def compute_histogram(values):
            """
            Computes normalized histogram of voxel values.
            """
            # Compute the histogram and ensure it is cast to float32
            hist = tf.histogram_fixed_width(values, focus_range, nbins=50)
            hist = tf.cast(hist, tf.float32)  # Explicitly cast to float32

            # Normalize histogram to create a probability distribution
            hist_normalized = hist / tf.reduce_sum(hist + 1e-8)  # Avoid division by zero
            return hist_normalized

        # Compute normalized histograms for y_true and y_pred
        true_hist = compute_histogram(y_true)
        pred_hist = compute_histogram(y_pred)

        # Compute KL divergence: D_KL(P || Q) = sum(P * log(P / Q))
        kl_raw = tf.reduce_sum(true_hist * tf.math.log((true_hist + 1e-8) / (pred_hist + 1e-8)))
        kl_loss = kl_raw * kl_weight 
       
        # Combine voxel loss and KL loss
        total_loss = voxel_loss + kl_loss

        return total_loss

    return weighted_mae


def create_level_set_loss(config):
    if config.losses.level_set_0_loss == 'ranged_weighted_mae':
        loss = ranged_weighted_mae(focus_range = (float(config.losses.target_max)-1, float(config.losses.target_max)),
                                   gradient_type = config.losses.target_gradient,
                                   max_weight = config.losses.range_weight,
                                   kl_weight=config.losses.kl_weight)
        
    else:
        raise KeyError(f"Unknown loss: {config.losses.level_set_0_loss}")
    
    return loss

losses_distance_set = {'weighted_mae_0': weighted_mae(cap_values=[0.], cap_weight = 0.)}
metrics_level_set = {
                     'mean_squared_error': tf.keras.metrics.mean_squared_error,
                     'mean_absolute_error': tf.keras.metrics.mean_absolute_error,
                     'non_capped_mae': non_capped_mae(cap_high=5, cap_low=-5),
                     'bounded_mae': bounded_mae(cap_low=0., cap_high = 1.),
}

metrics_distance_set = {
                        'mean_squared_error': tf.keras.metrics.mean_squared_error,
                        'mean_absolute_error': tf.keras.metrics.mean_absolute_error,
                        'non_capped_mae': non_capped_mae(cap_high=0, cap_low=0),
}
