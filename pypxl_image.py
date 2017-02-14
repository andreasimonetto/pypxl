#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, cv2
from pypxl import process_frame

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='pypxl_image.py', description='Pixelate an image.', formatter_class=lambda *args, **kwargs: argparse.HelpFormatter(*args, max_help_position=80, **kwargs))
    parser.add_argument('-k', '--clusters-num', metavar='K', type=int, required=False, default=16, help='Number of clusters in K-Means (default: 16)')
    parser.add_argument('-p', '--prescale-size', type=int, required=False, nargs=2, metavar=('W', 'H'), default=None, help='Prescale size (width and height)')
    parser.add_argument('-s', '--subsample-size', type=int, required=True, nargs=2, metavar=('W', 'H'), help='Subsample size (width and height)')
    parser.add_argument('path_in', metavar='input', type=str, help='Input image file path')
    parser.add_argument('path_out', metavar='output', type=str, nargs='?', default=None, help='Output image file path')

    # Parse command line arguments
    args = parser.parse_args()

    # Open input image in BGR space
    im_in = cv2.imread(args.path_in, cv2.IMREAD_COLOR)

    # Process image with given arguments
    im_out = process_frame(im_in, args.subsample_size[0], args.subsample_size[1], args.clusters_num, None if args.prescale_size is None else tuple(args.prescale_size))

    # Write output (or display output image, depending on usage)
    if args.path_out is None:
        cv2.imshow('Output', im_out)
        cv2.waitKey(0)
    else:
        cv2.imwrite(args.path_out, im_out)
