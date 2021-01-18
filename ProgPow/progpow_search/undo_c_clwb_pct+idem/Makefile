#
# Copyright (c) 2019 Solar Designer <solar at openwall.com>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted.
#
# There's ABSOLUTELY NO WARRANTY, express or implied.
#
TARGET_ARCH := x86_64
TARGET_SIZE := 64
HOST_COMPILER := gcc

# Location of the CUDA Toolkit
CUDA_PATH := "/usr/local/cuda-10.1"
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m${TARGET_SIZE}
CCFLAGS     :=
LDFLAGS     := -s -lm
# build flags
ifeq ($(TARGET_OS),darwin)
    LDFLAGS += -rpath $(CUDA_PATH)/lib
    CCFLAGS += -arch $(HOST_ARCH)
else ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_OS),x86_64-armv7l-linux)
    LDFLAGS += --dynamic-linker=/lib/ld-linux-armhf.so.3
    CCFLAGS += -mfloat-abi=hard
else ifeq ($(TARGET_OS),android)
    LDFLAGS += -pie
    CCFLAGS += -fpie -fpic -fexceptions
endif

ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-linux)
        ifneq ($(TARGET_FS),)
            GCCVERSIONLTEQ46 := $(shell expr `$(HOST_COMPILER) -dumpversion` \<= 4.6)
            ifeq ($(GCCVERSIONLTEQ46),1)
                CCFLAGS += --sysroot=$(TARGET_FS)
            endif
            LDFLAGS += --sysroot=$(TARGET_FS)
            LDFLAGS += -rpath-link=$(TARGET_FS)/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/arm-linux-gnueabihf
        endif
    endif
endif

ifeq ($(TARGET_OS),qnx)
    CCFLAGS += -DWIN_INTERFACE_CUSTOM
    LDFLAGS += -lsocket
endif

# Install directory of different arch
CUDA_INSTALL_TARGET_DIR :=
ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-linux)
    CUDA_INSTALL_TARGET_DIR = targets/armv7-linux-gnueabihf/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-linux)
    CUDA_INSTALL_TARGET_DIR = targets/aarch64-linux/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-android)
    CUDA_INSTALL_TARGET_DIR = targets/armv7-linux-androideabi/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-android)
    CUDA_INSTALL_TARGET_DIR = targets/aarch64-linux-androideabi/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-qnx)
    CUDA_INSTALL_TARGET_DIR = targets/ARMv7-linux-QNX/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-qnx)
    CUDA_INSTALL_TARGET_DIR = targets/aarch64-qnx/
else ifeq ($(TARGET_ARCH),ppc64le)
    CUDA_INSTALL_TARGET_DIR = targets/ppc64le-linux/
endif

# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      BUILD_TYPE := debug
else
      BUILD_TYPE := release
endif
EXTRA_NVCCFLAGS := --cudart=shared
ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

SAMPLE_ENABLED := 1

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Common includes and paths for CUDA
#INCLUDES  := -I../../common/inc
INCLUDES  :=
LIBRARIES :=

################################################################################

# Gencode arguments
#SMS ?= 30 35 37 50 52 60 61 70
SMS ?=70
ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

ifeq ($(SAMPLE_ENABLED),0)
EXEC ?= @echo "[@]"
endif

####################################################
# Common includes and paths for CUDA
CUDA_INC_PATH   = $(CUDA_PATH)/include
NVCCINCLUDES      := -I$(CUDA_INC_PATH) -I. -I/usr/local/cuda-10.1/samples/common/inc/

LIBS = -lrt -lpthread 

CC_SRC_FILES = $(wildcard *.c)
OBJECT_FILE = $(CC_SRC_FILES:.c=.o)

####################################################
CC = gcc
LD = $(CC)
RM = rm -f
CFLAGS = -std=c99 -Wall -O2 #-DPROGPOW_VERSION=93
#LDFLAGS = -s -lm

PROJ = main
OBJS = main.o \
	libethcore/EthashAux.o libethcore/Miner.o \
	libethash/internal.o libethash/sha3.o libethash/io.o libethash/io_posix.o \
	libdevcore/SHA3.o \
	libethash-cuda/CUDAMiner_kernel.o

#OBJS = main.o \
	progpow.o \
	libethash/internal.o libethash/sha3.o libethash/io.o libethash/io_posix.o \
	libethash-cuda/CUDAMiner_cuda.o ibethash-cuda/CUDAMiner_kernel.o 
	
#libdevcore/CommonData.o libdevcore/FixedHash.o libdevcore/Log.o libdevcore/RLP.o libdevcore/SHA3.o libdevcore/Worker.o 
all: $(PROJ)

main: $(OBJS)
	g++   $(LDFLAGS) $(NVCCINCLUDES) $(OBJS) -L$(CUDA_PATH)/lib64 -o main -lcuda -lcudart

.cpp.o:
	g++ -c -std=c++11  $(NVCCINCLUDES) $< -o $@
.c.o:
	$(CC) -c $(CFLAGS)  $(NVCCINCLUDES) $< -o $@

libethash-cuda/CUDAMiner_cuda.o : libethash-cuda/CUDAMiner_cuda.cu
	$(EXEC) $(NVCC) $(NVCCINCLUDES) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

libethash-cuda/CUDAMiner_kernel.o : libethash-cuda/CUDAMiner_kernel.cu
	$(EXEC) $(NVCC) $(NVCCINCLUDES) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<


clean:
	$(RM) $(PROJ)
	$(RM) $(OBJS)
