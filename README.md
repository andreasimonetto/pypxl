PyPXL
=====

Python script to pixelate images and video using K-Means clustering in the
Lab colorspace. Video pixelating support multi-processing to achieve
better performance.

Dependencies
------------

	:::sh
	# Mandatory
	sudo apt-get install python-numpy python-sklearn python-opencv

	# Optional
	sudo apt-get install libav-tools

Sources
-------

- pypxl.py: Pixelate library.
- pypxl_image.py: Pixelate an image from command line.
- pypxl_video.py: Pixelate a video from command line.

Images
------

	:::sh
	# Pixelate and display an simage
	./pypxl_image.py -k 6 -s 4 4 test.png

![Test image](/goat/pypxl/raw/master/test.png)
![Pixelate result](/goat/pypxl/raw/master/test_pxl.png)

Videos
------

K-Means is an expensive task, so you may consider to use multi-processing in
pixelating videos (-np option). This may not be efficient enough: in that case
you can apply a pre-scale to each frame to reduce the time spent on K-Means
(-p option).

pypxl_video.py can process only video tracks. The audio can be extracted and
re-added to the pixelated video with avconv:

	:::sh
	# Extract audio from source video
	avconv -i src.avi -c:a copy src_audio.mp3

	# Pixelate source video using 4 parallel processes
	./pypxl_video.py -np 4 -s 8 8 src.avi out_noaudio.avi

	# Add audio to pixelate video
	avconv -i out_noaudio.avi -i src_audio.mp3 out.avi
