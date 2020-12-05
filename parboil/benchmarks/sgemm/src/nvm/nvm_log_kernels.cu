__device__ float NVM_log[LOG_SIZE_16M];
__device__ float NVM_flag[FLAG_SIZE_1M];

__global__ void mysgemmNT_nvmg( const float *A, int lda, const float *B, int ldb, float* C, int ldc, int k, float alpha, float beta )
{

    // Partial results 
    float c[TILE_N];
    for (int i=0; i < TILE_N; i++)
	c[i] = 0.0f;
    int mid = threadIdx.y * blockDim.x + threadIdx.x; //flattened id
    int m = blockIdx.x * TILE_M + mid;
    int n = blockIdx.y * TILE_N + threadIdx.x;
    __shared__ float b_s[TILE_TB_HEIGHT][TILE_N];
    for (int i = 0; i < k; i+=TILE_TB_HEIGHT) {
	float a; 
	b_s[threadIdx.y][threadIdx.x]=B[n + (i+threadIdx.y)*ldb];
	__syncthreads();
	for (int j = 0; j < TILE_TB_HEIGHT; j++) {
	    a = A[m + (i+j)*lda];
	    for (int kk = 0; kk < TILE_N; kk++)
		c[kk] += a * b_s[j][kk];

	}
	__syncthreads();
    }
    int t = ldc*blockIdx.y * TILE_N + m;
    // Create undo log
    for (int i = 0; i < TILE_N; i++) {
      ST_WT_FLOAT(&NVM_log[t+i*ldc], C[t+i*ldc]);
      //NVM_log[t+i*ldc] = C[t+i*ldc];
    }
    SFENCE; __syncthreads();
    SET_NVM_FLAG(1);
    STC_WT(0); 
    STC_WT(1); 
    STC_WT(2); 
    STC_WT(3); 
    STC_WT(4); 
    STC_WT(5); 
    STC_WT(6); 
    STC_WT(7); 
    STC_WT(8); 
    STC_WT(9); 
    STC_WT(10); 
    STC_WT(11); 
    STC_WT(12); 
    STC_WT(13); 
    STC_WT(14); 
    STC_WT(15); 
    SFENCE; __syncthreads();
    SET_NVM_FLAG(2);
}

__global__ void mysgemmNT_nvm3( const float *A, int lda, const float *B, int ldb, float* C, int ldc, int k, float alpha, float beta )
{

    // Partial results 
    float c[TILE_N];
    for (int i=0; i < TILE_N; i++)
	c[i] = 0.0f;
    int mid = threadIdx.y * blockDim.x + threadIdx.x; //flattened id
    int m = blockIdx.x * TILE_M + mid;
    int n = blockIdx.y * TILE_N + threadIdx.x;
    __shared__ float b_s[TILE_TB_HEIGHT][TILE_N];
    for (int i = 0; i < k; i+=TILE_TB_HEIGHT) {
	float a; 
	b_s[threadIdx.y][threadIdx.x]=B[n + (i+threadIdx.y)*ldb];
	__syncthreads();
	for (int j = 0; j < TILE_TB_HEIGHT; j++) {
	    a = A[m + (i+j)*lda];
	    for (int kk = 0; kk < TILE_N; kk++)
		c[kk] += a * b_s[j][kk];

	}
	__syncthreads();
    }
    int t = ldc*blockIdx.y * TILE_N + m;
    // Create undo log
    for (int i = 0; i < TILE_N; i++) {
      NVM_log[t+i*ldc] = C[t+i*ldc];
    }
    for (int i = 0; i < TILE_N; i++) {
      CLWB(&NVM_log[t+i*ldc]);
    }
    SFENCE; __syncthreads();
    SET_NVM_FLAG_WB(1);
    for (int i = 0; i < TILE_N; i++) {
	C[t+i*ldc] = C[t+i*ldc] * beta + alpha * c[i];
    }
    for (int i = 0; i < TILE_N; i++) {
      CLWB(&C[t+i*ldc]);
    }
    SFENCE; __syncthreads();
    SET_NVM_FLAG_WB(2);
}

__global__ void mysgemmNT_nvm4( const float *A, int lda, const float *B, int ldb, float* C, int ldc, int k, float alpha, float beta )
{

    // Partial results 
    float c[TILE_N];
    for (int i=0; i < TILE_N; i++)
	c[i] = 0.0f;
    int mid = threadIdx.y * blockDim.x + threadIdx.x; //flattened id
    int m = blockIdx.x * TILE_M + mid;
    int n = blockIdx.y * TILE_N + threadIdx.x;
    __shared__ float b_s[TILE_TB_HEIGHT][TILE_N];
    for (int i = 0; i < k; i+=TILE_TB_HEIGHT) {
	float a; 
	b_s[threadIdx.y][threadIdx.x]=B[n + (i+threadIdx.y)*ldb];
	__syncthreads();
	for (int j = 0; j < TILE_TB_HEIGHT; j++) {
	    a = A[m + (i+j)*lda];
	    for (int kk = 0; kk < TILE_N; kk++)
		c[kk] += a * b_s[j][kk];

	}
	__syncthreads();
    }
    int t = ldc*blockIdx.y * TILE_N + m;
    // Create undo log
    for (int i = 0; i < TILE_N; i++) {
      NVM_log[t+i*ldc] = C[t+i*ldc];
    }
    for (int i = 0; i < TILE_N; i++) {
      CLWB(&NVM_log[t+i*ldc]);
    }
    SFENCE; PCOMMIT; SFENCE; __syncthreads();
    SET_NVM_FLAG_WB_PC(1);
    for (int i = 0; i < TILE_N; i++) {
	C[t+i*ldc] = C[t+i*ldc] * beta + alpha * c[i];
    }
    for (int i = 0; i < TILE_N; i++) {
      CLWB(&C[t+i*ldc]);
    }
    SFENCE; PCOMMIT; SFENCE; __syncthreads();
    SET_NVM_FLAG_WB_PC(2);
}
