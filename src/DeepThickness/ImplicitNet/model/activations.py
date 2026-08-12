#!/usr/bin/env python3
"""
@authors:
* Damiano Ferrari, University of Brescia

Custom activation functions
"""

from tensorflow.keras import backend as K


def leaky_clip(leaky_min = -5., leaky_max = 5., alpha=0.1):
    """
    Returns an activation function that is:
    * (alpha * x + leaky_min - alpha * leaky_min) if x < leaky_min
    * x if leaky_min < x < leaky_max
    * (alpha * x + leaky_max - alpha * leaky_max) if x > leaky_max
    
    -5 and 5. is the best configuration for the level set, while 1 and 5 is the
    best for the distance set.
    """
    def leaky_clip(x):
        return K.switch(x < leaky_min, alpha * x + leaky_min - alpha * leaky_min, \
            K.switch(x > leaky_max, alpha * x + leaky_max - alpha * leaky_max, x))

    return leaky_clip
