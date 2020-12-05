# (c) 2010 The Board of Trustees of the University of Illinois.

SRCDIR_OBJS=main.o 
LANGUAGE=cuda
APP_LDFLAGS=-lm -lstdc++

# To use slow (but precise) math routines
ifeq ($(slowmath), 1)
	APP_CFLAGS += -fp-model precise
else
	APP_CFLAGS += -ffast-math -g
	APP_CXXFLAGS += -ffast-math -g
	APP_CUDACFLAGS += -Xptxas -dlcm=cg --use_fast_math -g -Xcompiler -O2
endif

# To use the MKL library
ifeq ($(mkl), 1)
	APP_LDFLAGS += -L/opt/intel/mkl/9.1.023/lib/32 \
		-lmkl -lvml -lguide -lpthread
	INCLUDES += -I/opt/intel/mkl/9.1.023/include
	APP_CFLAGS += -DMKL
endif

