#!/usr/bin/env python
# -*- coding: utf-8 -*-

import signal, argparse, cv2
from multiprocessing import Pool
from pypxl import process_frame

def process_frame_tuple(args):
    """One argument version of process_frame(), taking arguments as a tuple."""

    return process_frame(*args)

def stream_multiread(stream, n=1):
    """Video stream (cv2.VideoCapture) iterator, able to read n frames a time."""

    while True:
        frames = []
        for _ in range(n):
            stream_isopen, frame = stream.read()
            frames.append(frame if stream_isopen else None)

        if all([ frame is None for frame in frames ]):
            return

        yield frames

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='pypxl_video.py', description='Pixelate a video.', formatter_class=lambda *args, **kwargs: argparse.HelpFormatter(*args, max_help_position=80, **kwargs))
    parser.add_argument('-np', '--processes-num', metavar='NP', type=int, required=False, default=1, help='Number of processes')
    parser.add_argument('-c', '--codec', type=str, required=False, default=None, help='Codec to use (default: same as source). See: http://www.fourcc.org/codecs.php')
    parser.add_argument('-k', '--clusters-num', metavar='K', type=int, required=False, default=16, help='Number of clusters in K-Means (default: 16)')
    parser.add_argument('-p', '--prescale-size', type=int, required=False, nargs=2, metavar=('W', 'H'), default=None, help='Prescale size (width and height)')
    parser.add_argument('-s', '--subsample-size', type=int, required=True, nargs=2, metavar=('W', 'H'), help='Subsample size (width and height)')
    parser.add_argument('path_in', metavar='input', type=str, help='Input video file path')
    parser.add_argument('path_out', metavar='output', type=str, help='Output video file path')

    # Parse command line arguments
    args = parser.parse_args()

    # Default process_frame() arguments
    process_args = (args.subsample_size[0], args.subsample_size[1], args.clusters_num, None if args.prescale_size is None else tuple(args.prescale_size))

    # Open input video
    video_in = cv2.VideoCapture(args.path_in)

    # Open output video (initially empty)
    fourcc_out = int(video_in.get(cv2.CAP_PROP_FOURCC)) if args.codec is None else cv2.VideoWriter_fourcc(*(args.codec.upper()))
    fps_out = int(video_in.get(cv2.CAP_PROP_FPS))
    width_out = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_out = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_out = cv2.VideoWriter(args.path_out, fourcc_out, fps_out, (width_out, height_out)) # Fixed Xvid 640x480@20fps

    if args.processes_num <= 1:
        # Single process
        for (frame_in,) in stream_multiread(video_in):
            # Write the output video, frame by frame
            video_out.write(process_frame(*((frame_in,) + process_args)))
    else:
        # Multiprocessing: open a pool of processes (ignoring Ctrl-C from keyboard)
        pool = Pool(args.processes_num, lambda *args: signal.signal(signal.SIGINT, signal.SIG_IGN))

        try:
            # Read NP frames per iteration
            for frames_in in stream_multiread(video_in, args.processes_num):
                # Map process_frame() function to processes in pool, each
                # with a different input frame. Use process_frame_tuple()
                # because pool.map() wants a function with only one argument
                for frame_out in pool.map(process_frame_tuple, [ ((frame_in,) + process_args) for frame_in in frames_in ]):
                    # Write processed frames to output stream.
                    # frame_out is None if all the input stream is consumed
                    if frame_out is not None:
                        video_out.write(frame_out)
        except KeyboardInterrupt:
            # If Ctrl-C were pressed, close process pool and exit
            pool.terminate()
            pool.join()
