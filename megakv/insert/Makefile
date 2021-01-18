TARGET_ARCH := x86_64
TARGET_SIZE := 64
HOST_COMPILER := gcc

# Location of the CUDA Toolkit
CUDA_PATH := "/usr/local/cuda-10.1"
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m${TARGET_SIZE}
CCFLAGS     :=
LDFLAGS     :=
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
SMS ?= 30 35 37 50 52 60 70

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
all: build

build: run

check.deps:
ifeq ($(SAMPLE_ENABLED),0)
	@echo "Sample will be waived due to the above missing dependencies"
else
	@echo "Sample is ready - all dependencies have been met"
endif

run: $(OBJECT_FILE) gpu_hash.o
	$(EXEC) $(NVCC) $(NVCCINCLUDES) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)$(LIBS)

$(OBJECT_FILE): $(CC_SRC_FILES)
	$(EXEC) $(HOST_COMPILER) $(NVCCINCLUDES) $(INCLUDES) $(CCFLAGS) $(EXTRA_CCFLAGS) -o $@ -c $<

gpu_hash.o : gpu_hash.cu
	$(EXEC) $(NVCC) $(NVCCINCLUDES) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

.PHONY : clean

clean:
	rm -f *.o run 


