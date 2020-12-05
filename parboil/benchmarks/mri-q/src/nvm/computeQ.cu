/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include "nvm_util.h"

#define PI   3.1415926535897932384626433832795029f
#define PIx2 6.2831853071795864769252867665590058f

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define K_ELEMS_PER_GRID 2048

#define KERNEL_PHI_MAG_THREADS_PER_BLOCK 512
#define KERNEL_Q_THREADS_PER_BLOCK 256
#define KERNEL_Q_K_ELEMS_PER_GRID 1024

#define CUDA_ERRCK							\
  {cudaError_t err;							\
    if ((err = cudaGetLastError()) != cudaSuccess) {			\
      fprintf(stderr, "CUDA error on line %d: %s\n", __LINE__, cudaGetErrorString(err)); \
      exit(-1);								\
    }									\
  }

struct kValues {
  float Kx;
  float Ky;
  float Kz;
  float PhiMag;
};

/* Values in the k-space coordinate system are stored in constant memory
 * on the GPU */
//__constant__ __device__ kValues ck[KERNEL_Q_K_ELEMS_PER_GRID];
__device__ kValues ck[KERNEL_Q_K_ELEMS_PER_GRID];

#include "nvm_kernels.cu"
__global__ void kernel_l2wb(void){
        L2WB;
        MEM_FENCE;
}
__global__ void kernel_l2wb_pct(void){
        L2WB;
        MEM_FENCE;
         PCOMMIT; MEM_FENCE;
}

__global__ void
ComputePhiMag_GPU(float* phiR, float* phiI, float* phiMag, int numK) {
  int indexK = blockIdx.x*KERNEL_PHI_MAG_THREADS_PER_BLOCK + threadIdx.x;
  if (indexK < numK) {
    float real = phiR[indexK];
    float imag = phiI[indexK];
    phiMag[indexK] = real*real + imag*imag;
  }
}

__global__ void
ComputeQ_GPU(int numK, int kGlobalIndex,
	     float* x, float* y, float* z, float* Qr , float* Qi)
{
  float sX;
  float sY;
  float sZ;
  float sQr;
  float sQi;

  // Determine the element of the X arrays computed by this thread
  int xIndex = blockIdx.x*KERNEL_Q_THREADS_PER_BLOCK + threadIdx.x;

  // Read block's X values from global mem to shared mem
  sX = x[xIndex];
  sY = y[xIndex];
  sZ = z[xIndex];
  sQr = Qr[xIndex];
  sQi = Qi[xIndex];

  // Loop over all elements of K in constant mem to compute a partial value
  // for X.
  int kIndex = 0;
  if (numK % 2) {
    float expArg = PIx2 * (ck[0].Kx * sX + ck[0].Ky * sY + ck[0].Kz * sZ);
    sQr += ck[0].PhiMag * __cosf(expArg);
    sQi += ck[0].PhiMag * __sinf(expArg);
    kIndex++;
    kGlobalIndex++;
  }

  for (; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
       kIndex += 2, kGlobalIndex += 2) {
    float expArg = PIx2 * (ck[kIndex].Kx * sX +
			   ck[kIndex].Ky * sY +
			   ck[kIndex].Kz * sZ);
    sQr += ck[kIndex].PhiMag * __cosf(expArg);
    sQi += ck[kIndex].PhiMag * __sinf(expArg);

    int kIndex1 = kIndex + 1;
    float expArg1 = PIx2 * (ck[kIndex1].Kx * sX +
			    ck[kIndex1].Ky * sY +
			    ck[kIndex1].Kz * sZ);
    sQr += ck[kIndex1].PhiMag * __cosf(expArg1);
    sQi += ck[kIndex1].PhiMag * __sinf(expArg1);
  }

  Qr[xIndex] = sQr;
  Qi[xIndex] = sQi;
}

syncLapTimer st_pm, st_q;

void computePhiMag_GPU(int numK, float* phiR_d, float* phiI_d, float* phiMag_d)
{
  int phiMagBlocks = numK / KERNEL_PHI_MAG_THREADS_PER_BLOCK;
  if (numK % KERNEL_PHI_MAG_THREADS_PER_BLOCK)
    phiMagBlocks++;
  dim3 DimPhiMagBlock(KERNEL_PHI_MAG_THREADS_PER_BLOCK, 1);
  dim3 DimPhiMagGrid(phiMagBlocks, 1);
  //DimPhiMagBlock.x = 32;
  //DimPhiMagGrid.x = 1;
  float *NVM_klog;
  NVM_KLOG_ALLOC(&NVM_klog);
  for (int i = 0; i < 1; i++) {
    st_pm.lap_start();
    if (nvm_opt == 'a')
      ComputePhiMag_GPU_nvma <<< DimPhiMagGrid, DimPhiMagBlock >>> (phiR_d, phiI_d, phiMag_d, numK);
    else if (nvm_opt == 'b')
      ComputePhiMag_GPU_nvmb <<< DimPhiMagGrid, DimPhiMagBlock >>> (phiR_d, phiI_d, phiMag_d, numK);
    else if (nvm_opt == 'o')
      ComputePhiMag_GPU_nvmo <<< DimPhiMagGrid, DimPhiMagBlock >>> (phiR_d, phiI_d, phiMag_d, numK);
    else if (nvm_opt == 'u')
      ComputePhiMag_GPU_nvmu <<< DimPhiMagGrid, DimPhiMagBlock >>> (phiR_d, phiI_d, phiMag_d, numK);
    else if (nvm_opt == 'f') {
      NVM_KLOG_FILL(NVM_klog, phiMag_d, numK * sizeof(float));
      CHECK_BARRIER;
      ComputePhiMag_GPU <<< DimPhiMagGrid, DimPhiMagBlock >>> (phiR_d, phiI_d, phiMag_d, numK);
    }
    else if  (nvm_opt == 'k') {
      ComputePhiMag_GPU <<< DimPhiMagGrid, DimPhiMagBlock >>> (phiR_d, phiI_d, phiMag_d, numK);
	cudaDeviceSynchronize();
	kernel_l2wb <<< DimPhiMagGrid, DimPhiMagBlock >>>();
    } 
    else if  (nvm_opt == 'l') {
      ComputePhiMag_GPU <<< DimPhiMagGrid, DimPhiMagBlock >>> (phiR_d, phiI_d, phiMag_d, numK);
        cudaDeviceSynchronize();
	kernel_l2wb_pct <<< DimPhiMagGrid, DimPhiMagBlock >>>();
    }
    else if (nvm_opt == 'm') {
      NVM_KLOG_FILL(NVM_klog, phiMag_d, numK * sizeof(float));
      CHECK_BARRIER;
      ComputePhiMag_GPU <<< DimPhiMagGrid, DimPhiMagBlock >>> (phiR_d, phiI_d, phiMag_d, numK);
        cudaDeviceSynchronize();
        kernel_l2wb <<< DimPhiMagGrid, DimPhiMagBlock >>>();
    } 
    else {
      ComputePhiMag_GPU <<< DimPhiMagGrid, DimPhiMagBlock >>> (phiR_d, phiI_d, phiMag_d, numK);    }
    st_pm.lap_end();
  }
  st_pm.print_avg_usec("ComputePhiMag", DimPhiMagGrid);
}

void computeQ_GPU(int numK, int numX,
                  float* x_d, float* y_d, float* z_d,
                  kValues* kVals,
                  float* Qr_d, float* Qi_d)
{
  int QGrids = numK / KERNEL_Q_K_ELEMS_PER_GRID;
  if (numK % KERNEL_Q_K_ELEMS_PER_GRID)
    QGrids++;
  int QBlocks = numX / KERNEL_Q_THREADS_PER_BLOCK;
  if (numX % KERNEL_Q_THREADS_PER_BLOCK)
    QBlocks++;
  dim3 DimQBlock(KERNEL_Q_THREADS_PER_BLOCK, 1);
  dim3 DimQGrid(QBlocks, 1);

  for (int QGrid = 0; QGrid < QGrids; QGrid++) {
    // Put the tile of K values into constant mem
    int QGridBase = QGrid * KERNEL_Q_K_ELEMS_PER_GRID;
    kValues* kValsTile = kVals + QGridBase;
    int numElems = MIN(KERNEL_Q_K_ELEMS_PER_GRID, numK - QGridBase);

    //cudaMemcpyToSymbol(ck, kValsTile, numElems * sizeof(kValues), 0);
    for (int i = 0; i < 1; i++) {
      st_q.lap_start();
      if (nvm_opt == 'a')
	ComputeQ_GPU <<< DimQGrid, DimQBlock >>>(numK, QGridBase, x_d, y_d, z_d, Qr_d, Qi_d);
      else if (nvm_opt == 'b')
	ComputeQ_GPU_nvmb <<< DimQGrid, DimQBlock >>>(numK, QGridBase, x_d, y_d, z_d, Qr_d, Qi_d);
      else if (nvm_opt == 'g')
	ComputeQ_GPU_nvmg <<< DimQGrid, DimQBlock >>>(numK, QGridBase, x_d, y_d, z_d, Qr_d, Qi_d);
      else if (nvm_opt == 'o')
	ComputeQ_GPU_nvmo <<< DimQGrid, DimQBlock >>>(numK, QGridBase, x_d, y_d, z_d, Qr_d, Qi_d);
      else if (nvm_opt == 'u')
	ComputeQ_GPU_nvmu <<< DimQGrid, DimQBlock >>>(numK, QGridBase, x_d, y_d, z_d, Qr_d, Qi_d);
      else if (nvm_opt == '1')
	ComputeQ_GPU_nvm1 <<< DimQGrid, DimQBlock >>>(numK, QGridBase, x_d, y_d, z_d, Qr_d, Qi_d);
      else if (nvm_opt == '2')
	ComputeQ_GPU_nvm2 <<< DimQGrid, DimQBlock >>>(numK, QGridBase, x_d, y_d, z_d, Qr_d, Qi_d);
      else if (nvm_opt == '3')
	ComputeQ_GPU_nvm3 <<< DimQGrid, DimQBlock >>>(numK, QGridBase, x_d, y_d, z_d, Qr_d, Qi_d);
      else if (nvm_opt == '4')
	ComputeQ_GPU_nvm4 <<< DimQGrid, DimQBlock >>>(numK, QGridBase, x_d, y_d, z_d, Qr_d, Qi_d);
      st_q.lap_end();
    }
  }
  st_q.print_avg_usec("ComputeQ", DimQGrid);
}

void createDataStructsCPU(int numK, int numX, float** phiMag,
	 float** Qr, float** Qi)
{
  *phiMag = (float* ) memalign(16, numK * sizeof(float));
  *Qr = (float*) memalign(16, numX * sizeof (float));
  *Qi = (float*) memalign(16, numX * sizeof (float));
}

