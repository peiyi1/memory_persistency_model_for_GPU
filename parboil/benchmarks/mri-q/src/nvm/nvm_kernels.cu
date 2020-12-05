__global__ void
ComputePhiMag_GPU_nvma(float* phiR, float* phiI, float* phiMag, int numK) {
  int indexK = blockIdx.x*KERNEL_PHI_MAG_THREADS_PER_BLOCK + threadIdx.x;
  if (indexK < numK) {
    float real = phiR[indexK];
    float imag = phiI[indexK];
    phiMag[indexK] = real*real + imag*imag;
  }
}

__global__ void
ComputePhiMag_GPU_nvmb(float* phiR, float* phiI, float* phiMag, int numK) {
  int indexK = blockIdx.x*KERNEL_PHI_MAG_THREADS_PER_BLOCK + threadIdx.x;
  if (indexK < numK) {
    float real = phiR[indexK];
    float imag = phiI[indexK];
    ST_WT_FLOAT(&phiMag[indexK], real*real + imag*imag);
    MEM_FENCE;
  }
}



__global__ void
ComputePhiMag_GPU_nvmu(float* phiR, float* phiI, float* phiMag, int numK) {
  int indexK = blockIdx.x*KERNEL_PHI_MAG_THREADS_PER_BLOCK + threadIdx.x;
  if (indexK < numK) {
    float real = phiR[indexK];
    float imag = phiI[indexK];
    phiMag[indexK] = real*real + imag*imag;
    CLWB(&phiMag[indexK]);
    SFENCE; PCOMMIT; SFENCE;
  }
}

__global__ void
ComputePhiMag_GPU_nvmo(float* phiR, float* phiI, float* phiMag, int numK) {
  int indexK = blockIdx.x*KERNEL_PHI_MAG_THREADS_PER_BLOCK + threadIdx.x;
  if (indexK < numK) {
    float real = phiR[indexK];
    float imag = phiI[indexK];
    phiMag[indexK] = real*real + imag*imag;
    CLWB(&phiMag[indexK]);
    SFENCE;
  }
}

__global__ void
ComputeQ_GPU_nvmb(int numK, int kGlobalIndex,
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

  //Qr[xIndex] = sQr;
  //Qi[xIndex] = sQi;
  ST_WT_FLOAT(&Qr[xIndex], sQr);
  MEM_FENCE;
  ST_WT_FLOAT(&Qi[xIndex], sQi);
  MEM_FENCE;
}

__device__ float NVM_log[LOG_SIZE_16M];
__device__ float NVM_flag[FLAG_SIZE_1M];
__global__ void ComputeQ_GPU_nvmg(int numK, int kGlobalIndex,
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

  ST_WT_FLOAT(&NVM_log[xIndex+numK], Qr[xIndex]);
  ST_WT_FLOAT(&NVM_log[xIndex+2*numK], Qi[xIndex]);
  MEM_FENCE; __syncthreads();
  SET_NVM_FLAG(1);

  ST_WT_FLOAT(&Qr[xIndex], sQr);
  ST_WT_FLOAT(&Qi[xIndex], sQi);
  MEM_FENCE;

  MEM_FENCE; __syncthreads();
  SET_NVM_FLAG(2);

}

__global__ void ComputeQ_GPU_nvm3(int numK, int kGlobalIndex,
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

  ST_WB_FLOAT(&NVM_log[xIndex+numK], Qr[xIndex]);
  ST_WB_FLOAT(&NVM_log[xIndex+2*numK], Qi[xIndex]);
  CLWB(&NVM_log[xIndex+numK]);
  CLWB(&NVM_log[xIndex+2*numK]);
  MEM_FENCE; __syncthreads();
  SET_NVM_FLAG_WB(1);

  Qr[xIndex] = sQr;
  Qi[xIndex] = sQi;
  CLWB(&Qr[xIndex]);
  CLWB(&Qi[xIndex]);
  MEM_FENCE; __syncthreads();
  SET_NVM_FLAG_WB(2);

}

__global__ void ComputeQ_GPU_nvm4(int numK, int kGlobalIndex,
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

  ST_WB_FLOAT(&NVM_log[xIndex+numK], Qr[xIndex]);
  ST_WB_FLOAT(&NVM_log[xIndex+2*numK], Qi[xIndex]);
  CLWB(&NVM_log[xIndex+numK]);
  CLWB(&NVM_log[xIndex+2*numK]);
  MEM_FENCE; PCOMMIT; MEM_FENCE; __syncthreads();
  SET_NVM_FLAG_WB_PC(1);

  Qr[xIndex] = sQr;
  Qi[xIndex] = sQi;
  CLWB(&Qr[xIndex]);
  CLWB(&Qi[xIndex]);
  MEM_FENCE; PCOMMIT; MEM_FENCE;

  __syncthreads();
  SET_NVM_FLAG_WB_PC(2);

}


__global__ void
ComputeQ_GPU_nvmo(int numK, int kGlobalIndex,
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
  CLWB(&Qr[xIndex]); SFENCE;
  Qi[xIndex] = sQi;
  CLWB(&Qi[xIndex]); SFENCE;
  
}

__global__ void
ComputeQ_GPU_nvmu(int numK, int kGlobalIndex,
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
  CLWB(&Qr[xIndex]); SFENCE; PCOMMIT; SFENCE;
  Qi[xIndex] = sQi;
  CLWB(&Qi[xIndex]); SFENCE; PCOMMIT; SFENCE;
  
}


__global__ void
ComputeQ_GPU_nvm1(int numK, int kGlobalIndex,
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
  L2WB;
  SFENCE;
}
__global__ void
ComputeQ_GPU_nvm2(int numK, int kGlobalIndex,
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
  L2WB;
  SFENCE; PCOMMIT; SFENCE;
}
