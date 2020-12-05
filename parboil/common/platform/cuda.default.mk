# (c) 2007 The Board of Trustees of the University of Illinois.

# Cuda-related definitions common to all benchmarks

########################################
# Variables
########################################

# c.default is the base along with CUDA configuration in this setting
include $(PARBOIL_ROOT)/common/platform/c.default.mk

# Paths
CUDAHOME=/usr/local/cuda-10.1

# Programs
CUDACC=$(CUDAHOME)/bin/nvcc
CUDALINK=$(CUDAHOME)/bin/nvcc

# Flags
#PLATFORM_CUDACFLAGS=-O3
HIGHEST_SM=70
PLATFORM_CUDACFLAGS +=-gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
PLATFORM_CUDALDFLAGS=-lm -lpthread


