#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
from sklearn.cluster import MiniBatchKMeans
import numpy as np

__version__   = '1.0.0'
__author__    = 'Goat'
__email__     = 'goat@ridiculousglitch.com'
__license__   = 'BSD-2'
__copyright__ = 'Copyright (c) 2017, Ridiculous Glitch'

def process_frame(im, width, height, n_clusters=16, prescale_size=None):
    h, w = im.shape[:2]

    # Apply optional prescale to source image
    if prescale_size is not None:
        im = cv2.resize(im, prescale_size, interpolation=cv2.INTER_LINEAR)

    # Quantize (cluster colors) and resample (subsample+supersample)
    return im_resample(im_quantize(im, n_clusters), width, height, w, h)

def im_resample(im, subsample_width, subsample_height, resample_width, resample_height):
    # Subsample to pixelate
    im_subsample = cv2.resize(im, (subsample_width, subsample_height), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(im_subsample, (resample_width, resample_height), interpolation=cv2.INTER_NEAREST)

def im_quantize(im, n_clusters):
    h, w = im.shape[:2]

    # Convert the image from the BGR color space to the L*a*b* color space.
    # Since we will be clustering using k-means which is based on the euclidean
    # distance, we'll use the L*a*b* color space where the euclidean distance
    # implies perceptual meaning
    im_colorspace = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)

    # - Reshape the image into a feature vector so that k-means can be applied
    # - Apply k-means using the specified number of clusters and then
    #   create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters=n_clusters)
    labels = clt.fit_predict(im_colorspace.reshape((w * h, 3)))
    im_quant_colorspace = clt.cluster_centers_.astype(np.uint8)[labels].reshape((h, w, 3))

    # Convert from L*a*b* back to BGR
    return cv2.cvtColor(im_quant_colorspace, cv2.COLOR_LAB2BGR)
