/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#include "nvm_util.h"

/* Integer ceiling division.  This computes ceil(x / y) */
#define CEIL(x,y) (((x) + ((y) - 1)) / (y))

/* Fast multiplication by 33 */
#define TIMES_DIM_POS(x) (((x) << 5) + (x))

/* Amount of dynamically allocated local storage
 * measured in bytes, 2-byte words, and 8-byte words */
#define SAD_LOC_SIZE_ELEMS (THREADS_W * THREADS_H * MAX_POS_PADDED)
#define SAD_LOC_SIZE_BYTES (SAD_LOC_SIZE_ELEMS * sizeof(unsigned short))
#define SAD_LOC_SIZE_8B (SAD_LOC_SIZE_BYTES / sizeof(vec8b))

/* The search position index space is distributed across threads
 * and across time. */
/* This many search positions are calculated by each thread.
 * Note: the optimized kernel requires that this number is
 * divisible by 3. */
#define POS_PER_THREAD 18

/* The width and height (in number of 4x4 blocks) of a tile from the
 * current frame that is computed in a single thread block. */
#define THREADS_W 1
#define THREADS_H 1

// #define TIMES_THREADS_W(x) (((x) << 1) + (x))
#define TIMES_THREADS_W(x) ((x) * THREADS_W)

/* This structure is used for vector load/store operations. */
struct vec8b {
  int fst;
  int snd;
} __align__(8);

typedef struct vec8b vec8b;

/* 4-by-4 SAD computation on the device. */
__global__ void mb_sad_calc(unsigned short*,
			    unsigned short*,
			    int, int);
__global__ void mb_sad_calc_nvmb(unsigned short*,
			    unsigned short*,
			    int, int);
__global__ void mb_sad_calc_nvmd(unsigned short*,
			    unsigned short*,
			    int, int);
__global__ void mb_sad_calc_nvmg(unsigned short*,
			    unsigned short*,
			    int, int);
__global__ void mb_sad_calc_nvmi(unsigned short*,
			    unsigned short*,
			    int, int);
__global__ void mb_sad_calc_nvmq(unsigned short*,
			    unsigned short*,
			    int, int);
__global__ void mb_sad_calc_nvmo(unsigned short*,
			    unsigned short*,
			    int, int);
__global__ void mb_sad_calc_nvmj(unsigned short*,
			    unsigned short*,
			    int, int);
__global__ void mb_sad_calc_nvml(unsigned short*,
			    unsigned short*,
			    int, int);
__global__ void mb_sad_calc_nvmu(unsigned short*,
			    unsigned short*,
			    int, int);
__global__ void mb_sad_calc_nvmw(unsigned short*,
			    unsigned short*,
			    int, int);
__global__ void mb_sad_calc_nvm1(unsigned short*,
			    unsigned short*,
			    int, int);
__global__ void mb_sad_calc_nvm2(unsigned short*,
			    unsigned short*,
			    int, int);
__global__ void mb_sad_calc_nvm3(unsigned short*,
			    unsigned short*,
			    int, int);
__global__ void mb_sad_calc_nvm4(unsigned short*,
			    unsigned short*,
			    int, int);
__global__ void mb_sad_calc_nvm5(unsigned short*,
			    unsigned short*,
			    int, int);
__global__ void mb_sad_calc_nvm6(unsigned short*,
			    unsigned short*,
			    int, int);

/* A function to get a reference to the "ref" texture, because sharing
 * of textures between files isn't really supported. */
texture<unsigned short, 2, cudaReadModeElementType> &get_ref(void);

