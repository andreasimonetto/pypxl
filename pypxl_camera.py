#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, cv2
from pypxl import process_frame
import numpy as np
from time import time

def camera_read(stream, n=2):
    """Video stream (cv2.VideoCapture) iterator, able to read n frames a time."""

    while True:
        frames = []
        for _ in range(n):
            stream_isopen, frame = stream.read()
            frames.append(np.expand_dims(frame, 0) if stream_isopen else None)

        if any([ frame is None for frame in frames ]):
            return

        yield np.mean(np.concatenate(frames), axis=0).astype(np.uint8)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='pypxl_image.py', description='Pixelate an image.', formatter_class=lambda *args, **kwargs: argparse.HelpFormatter(*args, max_help_position=80, **kwargs))
    parser.add_argument('-k', '--clusters-num', metavar='K', type=int, required=False, default=16, help='Number of clusters in K-Means (default: 16)')
    parser.add_argument('-p', '--prescale-size', type=int, required=False, nargs=2, metavar=('W', 'H'), default=None, help='Prescale size (width and height)')
    parser.add_argument('-s', '--subsample-size', type=int, required=True, nargs=2, metavar=('W', 'H'), help='Subsample size (width and height)')
    parser.add_argument('-d', '--device-no', type=str, required=False, default='/dev/video0', help='Camera device to use')
    #parser.add_argument('path_out', metavar='output', type=str, nargs='?', default=None, help='Output image file path')

    # Parse command line arguments
    args = parser.parse_args()

    camera_in = cv2.VideoCapture(args.device_no)
    for frame_in in camera_read(camera_in):
        #cv2.fastNlMeansDenoisingColored(frame_in, frame_in, 3, 3, 7, 7)

        # Process image with given arguments
        frame_out = process_frame(frame_in, args.subsample_size[0], args.subsample_size[1], args.clusters_num, None if args.prescale_size is None else tuple(args.prescale_size))

        # Display output frame
        cv2.imshow('Output', frame_out)
        if cv2.waitKey(1) & 0xff == 0x1b:
            break
