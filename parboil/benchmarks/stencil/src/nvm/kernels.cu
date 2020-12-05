/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include "common.h"


__global__ void naive_kernel(float c0,float c1,float *A0,float *Anext, int nx, int ny, int nz)
{
  int i = threadIdx.x;
  int j = blockIdx.x+1;
  int k = blockIdx.y+1;
  if(i>0)
    {
      Anext[Index3D (nx, ny, i, j, k)] = 
	(A0[Index3D (nx, ny, i, j, k + 1)] +
	 A0[Index3D (nx, ny, i, j, k - 1)] +
	 A0[Index3D (nx, ny, i, j + 1, k)] +
	 A0[Index3D (nx, ny, i, j - 1, k)] +
	 A0[Index3D (nx, ny, i + 1, j, k)] +
	 A0[Index3D (nx, ny, i - 1, j, k)])*c1
	- A0[Index3D (nx, ny, i, j, k)]*c0;
    }
}

__global__ void naive_kernel_nvmb(float c0,float c1,float *A0,float *Anext, int nx, int ny, int nz)
{
  int i = threadIdx.x;
  int j = blockIdx.x+1;
  int k = blockIdx.y+1;
  if(i>0)
    {
      float result = 
	(A0[Index3D (nx, ny, i, j, k + 1)] +
	 A0[Index3D (nx, ny, i, j, k - 1)] +
	 A0[Index3D (nx, ny, i, j + 1, k)] +
	 A0[Index3D (nx, ny, i, j - 1, k)] +
	 A0[Index3D (nx, ny, i + 1, j, k)] +
	 A0[Index3D (nx, ny, i - 1, j, k)])*c1
	- A0[Index3D (nx, ny, i, j, k)]*c0;

      //Anext[Index3D (nx, ny, i, j, k)] = result;
      ST_WT_FLOAT(&Anext[Index3D (nx, ny, i, j, k)], result);
      MEM_FENCE;
    }
}

__global__ void naive_kernel_nvmo(float c0,float c1,float *A0,float *Anext, int nx, int ny, int nz)
{
  int i = threadIdx.x;
  int j = blockIdx.x+1;
  int k = blockIdx.y+1;
  if(i>0)
    {
      float result = 
	(A0[Index3D (nx, ny, i, j, k + 1)] +
	 A0[Index3D (nx, ny, i, j, k - 1)] +
	 A0[Index3D (nx, ny, i, j + 1, k)] +
	 A0[Index3D (nx, ny, i, j - 1, k)] +
	 A0[Index3D (nx, ny, i + 1, j, k)] +
	 A0[Index3D (nx, ny, i - 1, j, k)])*c1
	- A0[Index3D (nx, ny, i, j, k)]*c0;

      Anext[Index3D (nx, ny, i, j, k)] = result;
      CLWB(&Anext[Index3D (nx, ny, i, j, k)]);
      SFENCE;
    }
}

__global__ void naive_kernel_nvmu(float c0,float c1,float *A0,float *Anext, int nx, int ny, int nz)
{
  int i = threadIdx.x;
  int j = blockIdx.x+1;
  int k = blockIdx.y+1;
  if(i>0)
    {
      float result = 
	(A0[Index3D (nx, ny, i, j, k + 1)] +
	 A0[Index3D (nx, ny, i, j, k - 1)] +
	 A0[Index3D (nx, ny, i, j + 1, k)] +
	 A0[Index3D (nx, ny, i, j - 1, k)] +
	 A0[Index3D (nx, ny, i + 1, j, k)] +
	 A0[Index3D (nx, ny, i - 1, j, k)])*c1
	- A0[Index3D (nx, ny, i, j, k)]*c0;

      Anext[Index3D (nx, ny, i, j, k)] = result;
      CLWB(&Anext[Index3D (nx, ny, i, j, k)]);
      SFENCE; PCOMMIT; SFENCE;
    }
}

__global__ void naive_kernel_nvm1(float c0,float c1,float *A0,float *Anext, int nx, int ny, int nz)
{
  int i = threadIdx.x;
  int j = blockIdx.x+1;
  int k = blockIdx.y+1;
  if(i>0)
    {
      float result = 
	(A0[Index3D (nx, ny, i, j, k + 1)] +
	 A0[Index3D (nx, ny, i, j, k - 1)] +
	 A0[Index3D (nx, ny, i, j + 1, k)] +
	 A0[Index3D (nx, ny, i, j - 1, k)] +
	 A0[Index3D (nx, ny, i + 1, j, k)] +
	 A0[Index3D (nx, ny, i - 1, j, k)])*c1
	- A0[Index3D (nx, ny, i, j, k)]*c0;

      Anext[Index3D (nx, ny, i, j, k)] = result;
      L2WB;
      SFENCE;
    }
}
__global__ void naive_kernel_nvm2(float c0,float c1,float *A0,float *Anext, int nx, int ny, int nz)
{
  int i = threadIdx.x;
  int j = blockIdx.x+1;
  int k = blockIdx.y+1;
  if(i>0)
    {
      float result = 
	(A0[Index3D (nx, ny, i, j, k + 1)] +
	 A0[Index3D (nx, ny, i, j, k - 1)] +
	 A0[Index3D (nx, ny, i, j + 1, k)] +
	 A0[Index3D (nx, ny, i, j - 1, k)] +
	 A0[Index3D (nx, ny, i + 1, j, k)] +
	 A0[Index3D (nx, ny, i - 1, j, k)])*c1
	- A0[Index3D (nx, ny, i, j, k)]*c0;

      Anext[Index3D (nx, ny, i, j, k)] = result;
      L2WB;
      SFENCE; PCOMMIT; SFENCE;
    }
}

__device__ float NVM_log[LOG_SIZE_16M];
__device__ float NVM_flag[FLAG_SIZE_1M];

__global__ void naive_kernel_nvmg(float c0,float c1,float *A0,float *Anext, int nx, int ny, int nz)
{
  int i = threadIdx.x;
  int j = blockIdx.x+1;
  int k = blockIdx.y+1;
  if(i>0)
    {
      ST_WT_FLOAT(&NVM_log[Index3D (nx, ny, i, j, k)], Anext[Index3D (nx, ny, i, j, k)]);
      MEM_FENCE; __syncthreads();
      SET_NVM_FLAG(1);

      float result = 
	(A0[Index3D (nx, ny, i, j, k + 1)] +
	 A0[Index3D (nx, ny, i, j, k - 1)] +
	 A0[Index3D (nx, ny, i, j + 1, k)] +
	 A0[Index3D (nx, ny, i, j - 1, k)] +
	 A0[Index3D (nx, ny, i + 1, j, k)] +
	 A0[Index3D (nx, ny, i - 1, j, k)])*c1
	- A0[Index3D (nx, ny, i, j, k)]*c0;

      //Anext[Index3D (nx, ny, i, j, k)] = result;
      ST_WT_FLOAT(&Anext[Index3D (nx, ny, i, j, k)], result);
    }
  MEM_FENCE; __syncthreads();
  SET_NVM_FLAG(2);
}

__global__ void naive_kernel_nvm3(float c0,float c1,float *A0,float *Anext, int nx, int ny, int nz)
{
  int i = threadIdx.x;
  int j = blockIdx.x+1;
  int k = blockIdx.y+1;
  if(i>0)
    {
      ST_WB_FLOAT(&NVM_log[Index3D (nx, ny, i, j, k)], Anext[Index3D (nx, ny, i, j, k)]);
      MEM_FENCE; __syncthreads();
      SET_NVM_FLAG_WB(1);

      float result = 
	(A0[Index3D (nx, ny, i, j, k + 1)] +
	 A0[Index3D (nx, ny, i, j, k - 1)] +
	 A0[Index3D (nx, ny, i, j + 1, k)] +
	 A0[Index3D (nx, ny, i, j - 1, k)] +
	 A0[Index3D (nx, ny, i + 1, j, k)] +
	 A0[Index3D (nx, ny, i - 1, j, k)])*c1
	- A0[Index3D (nx, ny, i, j, k)]*c0;

      Anext[Index3D (nx, ny, i, j, k)] = result;
      CLWB(&Anext[Index3D (nx, ny, i, j, k)]);
    }
  MEM_FENCE; __syncthreads();
  SET_NVM_FLAG_WB(2);
}

__global__ void naive_kernel_nvm4(float c0,float c1,float *A0,float *Anext, int nx, int ny, int nz)
{
  int i = threadIdx.x;
  int j = blockIdx.x+1;
  int k = blockIdx.y+1;
  if(i>0)
    {
      ST_WB_FLOAT(&NVM_log[Index3D (nx, ny, i, j, k)], Anext[Index3D (nx, ny, i, j, k)]);
      MEM_FENCE; __syncthreads();
      SET_NVM_FLAG_WB_PC(1);

      float result = 
	(A0[Index3D (nx, ny, i, j, k + 1)] +
	 A0[Index3D (nx, ny, i, j, k - 1)] +
	 A0[Index3D (nx, ny, i, j + 1, k)] +
	 A0[Index3D (nx, ny, i, j - 1, k)] +
	 A0[Index3D (nx, ny, i + 1, j, k)] +
	 A0[Index3D (nx, ny, i - 1, j, k)])*c1
	- A0[Index3D (nx, ny, i, j, k)]*c0;

      Anext[Index3D (nx, ny, i, j, k)] = result;
      CLWB(&Anext[Index3D (nx, ny, i, j, k)]);
    }
  MEM_FENCE; PCOMMIT; MEM_FENCE; __syncthreads();
  SET_NVM_FLAG_WB_PC(2);
}


__global__ void naive_kernel_nvmi(float c0,float c1,float *A0,float *Anext, int nx, int ny, int nz)
{
  int i = threadIdx.x;
  int j = blockIdx.x+1;
  int k = blockIdx.y+1;
  __syncthreads();
  SET_NVM_FLAG(1);
  if(i>0)
    {

      float result = 
	(A0[Index3D (nx, ny, i, j, k + 1)] +
	 A0[Index3D (nx, ny, i, j, k - 1)] +
	 A0[Index3D (nx, ny, i, j + 1, k)] +
	 A0[Index3D (nx, ny, i, j - 1, k)] +
	 A0[Index3D (nx, ny, i + 1, j, k)] +
	 A0[Index3D (nx, ny, i - 1, j, k)])*c1
	- A0[Index3D (nx, ny, i, j, k)]*c0;

      //Anext[Index3D (nx, ny, i, j, k)] = result;
      ST_WT_FLOAT(&Anext[Index3D (nx, ny, i, j, k)], result);
      MEM_FENCE;
    }
  MEM_FENCE; __syncthreads();
  SET_NVM_FLAG(2);
}

__global__ void naive_kernel_nvm6(float c0,float c1,float *A0,float *Anext, int nx, int ny, int nz)
{
  int i = threadIdx.x;
  int j = blockIdx.x+1;
  int k = blockIdx.y+1;
  __syncthreads();
  SET_NVM_FLAG_WB_PC(1);
  if(i>0)
    {

      float result = 
	(A0[Index3D (nx, ny, i, j, k + 1)] +
	 A0[Index3D (nx, ny, i, j, k - 1)] +
	 A0[Index3D (nx, ny, i, j + 1, k)] +
	 A0[Index3D (nx, ny, i, j - 1, k)] +
	 A0[Index3D (nx, ny, i + 1, j, k)] +
	 A0[Index3D (nx, ny, i - 1, j, k)])*c1
	- A0[Index3D (nx, ny, i, j, k)]*c0;

      Anext[Index3D (nx, ny, i, j, k)] = result;
      CLWB(&Anext[Index3D (nx, ny, i, j, k)]);
    }
  MEM_FENCE; PCOMMIT; MEM_FENCE; __syncthreads();
  SET_NVM_FLAG_WB_PC(2);
}

__global__ void naive_kernel_nvm5(float c0,float c1,float *A0,float *Anext, int nx, int ny, int nz)
{
  int i = threadIdx.x;
  int j = blockIdx.x+1;
  int k = blockIdx.y+1;
  __syncthreads();
  SET_NVM_FLAG_WB(1);
  if(i>0)
    {

      float result = 
	(A0[Index3D (nx, ny, i, j, k + 1)] +
	 A0[Index3D (nx, ny, i, j, k - 1)] +
	 A0[Index3D (nx, ny, i, j + 1, k)] +
	 A0[Index3D (nx, ny, i, j - 1, k)] +
	 A0[Index3D (nx, ny, i + 1, j, k)] +
	 A0[Index3D (nx, ny, i - 1, j, k)])*c1
	- A0[Index3D (nx, ny, i, j, k)]*c0;

      Anext[Index3D (nx, ny, i, j, k)] = result;
      CLWB(&Anext[Index3D (nx, ny, i, j, k)]);
    }
  MEM_FENCE; __syncthreads();
  SET_NVM_FLAG_WB(2);
}
