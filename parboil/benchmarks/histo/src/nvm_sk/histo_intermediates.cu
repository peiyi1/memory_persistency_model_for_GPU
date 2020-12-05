#include <stdio.h>
#include <cuda.h>

#include "util.h"
#include "nvm_util.h"

__device__ void calculateBin (
			      const unsigned int bin,
			      uchar4 *sm_mapping)
{
  unsigned char offset  =  bin        %   4;
  unsigned char indexlo = (bin >>  2) % 256;
  unsigned char indexhi = (bin >> 10) %  KB;
  unsigned char block   =  bin / BINS_PER_BLOCK;

  offset *= 8;

  uchar4 sm;
  sm.x = block;
  sm.y = indexhi;
  sm.z = indexlo;
  sm.w = offset;

  *sm_mapping = sm;
}

__global__ void histo_intermediates_kernel (
					    uint2 *input,
					    unsigned int height,
					    unsigned int width,
					    unsigned int input_pitch,
					    uchar4 *sm_mappings)
{
  unsigned int line = UNROLL * blockIdx.x;// 16 is the unroll factor;

  uint2 *load_bin = input + line * input_pitch + threadIdx.x;

  unsigned int store = line * width + threadIdx.x;
  bool skip = (width % 2) && (threadIdx.x == (blockDim.x - 1));

  //#pragma unroll
  for (int i = 0; i < UNROLL; i++)
    {
      uint2 bin_value = *load_bin;

      calculateBin (
		    bin_value.x,
		    &sm_mappings[store]
		    );

      if (!skip) calculateBin (
			       bin_value.y,
			       &sm_mappings[store + blockDim.x]
			       );

      load_bin += input_pitch;
      store += width;
    }
}


__device__ void calculateBin_nvma (
				   const unsigned int bin,
				   unsigned *sm_mapping)
{
  unsigned char offset  =  bin        %   4;
  unsigned char indexlo = (bin >>  2) % 256;
  unsigned char indexhi = (bin >> 10) %  KB;
  unsigned char block   =  bin / BINS_PER_BLOCK;

  offset *= 8;

  union {
    uchar4 sm;
    unsigned int smi;
  } cvt;
  cvt.sm.x = block;
  cvt.sm.y = indexhi;
  cvt.sm.z = indexlo;
  cvt.sm.w = offset;

  *sm_mapping = cvt.smi;
  //ST_WT_INT(sm_mapping, cvt.smi);
  //MEM_FENCE;
}

__global__ void histo_intermediates_kernel_nvma (
						 uint2 *input,
						 unsigned int height,
						 unsigned int width,
						 unsigned int input_pitch,
						 unsigned *sm_mappings)
{
  unsigned int line = UNROLL * blockIdx.x;// 16 is the unroll factor;

  uint2 *load_bin = input + line * input_pitch + threadIdx.x;

  unsigned int store = line * width + threadIdx.x;
  bool skip = (width % 2) && (threadIdx.x == (blockDim.x - 1));

#pragma unroll
  for (int i = 0; i < UNROLL; i++)
    {
      uint2 bin_value = *load_bin;

      calculateBin_nvma (
			 bin_value.x,
			 &sm_mappings[store]
			 );

      if (!skip) calculateBin_nvma (
				    bin_value.y,
				    &sm_mappings[store + blockDim.x]
				    );

      load_bin += input_pitch;
      store += width;
    }
}


__device__ void calculateBin_nvmb (
				   const unsigned int bin,
				   unsigned *sm_mapping)
{
  unsigned char offset  =  bin        %   4;
  unsigned char indexlo = (bin >>  2) % 256;
  unsigned char indexhi = (bin >> 10) %  KB;
  unsigned char block   =  bin / BINS_PER_BLOCK;

  offset *= 8;

  union {
    uchar4 sm;
    unsigned int smi;
  } cvt;
  cvt.sm.x = block;
  cvt.sm.y = indexhi;
  cvt.sm.z = indexlo;
  cvt.sm.w = offset;

  //*sm_mapping = sm;
  ST_WT_INT(sm_mapping, cvt.smi);
  MEM_FENCE;
}

__global__ void histo_intermediates_kernel_nvmb (
						 uint2 *input,
						 unsigned int height,
						 unsigned int width,
						 unsigned int input_pitch,
						 unsigned *sm_mappings)
{
  unsigned int line = UNROLL * blockIdx.x;// 16 is the unroll factor;

  uint2 *load_bin = input + line * input_pitch + threadIdx.x;

  unsigned int store = line * width + threadIdx.x;
  bool skip = (width % 2) && (threadIdx.x == (blockDim.x - 1));

#pragma unroll
  for (int i = 0; i < UNROLL; i++)
    {
      uint2 bin_value = *load_bin;

      calculateBin_nvmb (
			 bin_value.x,
			 &sm_mappings[store]
			 );

      if (!skip) calculateBin_nvmb (
				    bin_value.y,
				    &sm_mappings[store + blockDim.x]
				    );

      load_bin += input_pitch;
      store += width;
    }
}

__device__ void calculateBin_nvmo (
				   const unsigned int bin,
				   unsigned *sm_mapping)
{
  unsigned char offset  =  bin        %   4;
  unsigned char indexlo = (bin >>  2) % 256;
  unsigned char indexhi = (bin >> 10) %  KB;
  unsigned char block   =  bin / BINS_PER_BLOCK;

  offset *= 8;

  union {
    uchar4 sm;
    unsigned int smi;
  } cvt;
  cvt.sm.x = block;
  cvt.sm.y = indexhi;
  cvt.sm.z = indexlo;
  cvt.sm.w = offset;

  *sm_mapping = cvt.smi;
  //ST_WT_INT(sm_mapping, cvt.smi);
  CLWB(sm_mapping); SFENCE;
}

__global__ void histo_intermediates_kernel_nvmo (
						 uint2 *input,
						 unsigned int height,
						 unsigned int width,
						 unsigned int input_pitch,
						 unsigned *sm_mappings)
{
  unsigned int line = UNROLL * blockIdx.x;// 16 is the unroll factor;

  uint2 *load_bin = input + line * input_pitch + threadIdx.x;

  unsigned int store = line * width + threadIdx.x;
  bool skip = (width % 2) && (threadIdx.x == (blockDim.x - 1));

#pragma unroll
  for (int i = 0; i < UNROLL; i++)
    {
      uint2 bin_value = *load_bin;

      calculateBin_nvmo (
			 bin_value.x,
			 &sm_mappings[store]
			 );

      if (!skip) calculateBin_nvmo (
				    bin_value.y,
				    &sm_mappings[store + blockDim.x]
				    );

      load_bin += input_pitch;
      store += width;
    }
}

__device__ void calculateBin_nvmu (
				   const unsigned int bin,
				   unsigned *sm_mapping)
{
  unsigned char offset  =  bin        %   4;
  unsigned char indexlo = (bin >>  2) % 256;
  unsigned char indexhi = (bin >> 10) %  KB;
  unsigned char block   =  bin / BINS_PER_BLOCK;

  offset *= 8;

  union {
    uchar4 sm;
    unsigned int smi;
  } cvt;
  cvt.sm.x = block;
  cvt.sm.y = indexhi;
  cvt.sm.z = indexlo;
  cvt.sm.w = offset;

  *sm_mapping = cvt.smi;
  //ST_WT_INT(sm_mapping, cvt.smi);
  CLWB(sm_mapping); SFENCE;
  PCOMMIT; SFENCE;
}

__global__ void histo_intermediates_kernel_nvmu (
						 uint2 *input,
						 unsigned int height,
						 unsigned int width,
						 unsigned int input_pitch,
						 unsigned *sm_mappings)
{
  unsigned int line = UNROLL * blockIdx.x;// 16 is the unroll factor;

  uint2 *load_bin = input + line * input_pitch + threadIdx.x;

  unsigned int store = line * width + threadIdx.x;
  bool skip = (width % 2) && (threadIdx.x == (blockDim.x - 1));

#pragma unroll
  for (int i = 0; i < UNROLL; i++)
    {
      uint2 bin_value = *load_bin;

      calculateBin_nvmu (
			 bin_value.x,
			 &sm_mappings[store]
			 );

      if (!skip) calculateBin_nvmu (
				    bin_value.y,
				    &sm_mappings[store + blockDim.x]
				    );

      load_bin += input_pitch;
      store += width;
    }
}

__device__ void calculateBin_nvmj (
				   const unsigned int bin,
				   unsigned *sm_mapping)
{
  unsigned char offset  =  bin        %   4;
  unsigned char indexlo = (bin >>  2) % 256;
  unsigned char indexhi = (bin >> 10) %  KB;
  unsigned char block   =  bin / BINS_PER_BLOCK;

  offset *= 8;

  union {
    uchar4 sm;
    unsigned int smi;
  } cvt;
  cvt.sm.x = block;
  cvt.sm.y = indexhi;
  cvt.sm.z = indexlo;
  cvt.sm.w = offset;

  //*sm_mapping = cvt.sm;
  ST_WT_INT(sm_mapping, cvt.smi);
  SFENCE; PCOMMIT; SFENCE;
}

__global__ void histo_intermediates_kernel_nvmj (
						 uint2 *input,
						 unsigned int height,
						 unsigned int width,
						 unsigned int input_pitch,
						 unsigned *sm_mappings)
{
  unsigned int line = UNROLL * blockIdx.x;// 16 is the unroll factor;

  uint2 *load_bin = input + line * input_pitch + threadIdx.x;

  unsigned int store = line * width + threadIdx.x;
  bool skip = (width % 2) && (threadIdx.x == (blockDim.x - 1));

#pragma unroll
  for (int i = 0; i < UNROLL; i++)
    {
      uint2 bin_value = *load_bin;

      calculateBin_nvmj (
			 bin_value.x,
			 &sm_mappings[store]
			 );

      if (!skip) calculateBin_nvmj (
				    bin_value.y,
				    &sm_mappings[store + blockDim.x]
				    );

      load_bin += input_pitch;
      store += width;
    }
}
