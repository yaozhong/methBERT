.. _installation:

Running Enviroment
=======================

Envoriment
-----------------------
* ubuntu 16.04.10
* Python 3.5.2
* Pytorch 1.5.1+cu101

Docker file
-----------------------
We provide a docker image for running this source code::

	# get docker file
	docker pull yaozhong/ont_methylation:0.6
	# get into the docker enviroment
	nvidia-docker run -it --shm-size=64G -v LOCAL_DATA_PATH:MOUNT_DATA_PATH yaozhong/ont_methylation:0.6

