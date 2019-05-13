#!/bin/bash

nvidia-docker run \
	-v $HOME/sp19-6s897-colon/basic_classification:/basic_classification \
	-e CUDA_VISIBLE_DEVICES \
	elaine \
	$*