#define STC_CLWB(i) C[t+(i)*ldc] = C[t+(i)*ldc] * beta + alpha * c[(i)]; CLWB(&C[t+(i)*ldc]);

__global__ void mysgemmNT_nvmo( const float *A, int lda, const float *B, int ldb, float* C, int ldc, int k, float alpha, float beta )
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
    STC_CLWB(0); SFENCE;
    STC_CLWB(1); SFENCE;
    STC_CLWB(2); SFENCE;
    STC_CLWB(3); SFENCE;
    STC_CLWB(4); SFENCE;
    STC_CLWB(5); SFENCE;
    STC_CLWB(6); SFENCE;
    STC_CLWB(7); SFENCE;
    STC_CLWB(8); SFENCE;
    STC_CLWB(9); SFENCE;
    STC_CLWB(10); SFENCE;
    STC_CLWB(11); SFENCE;
    STC_CLWB(12); SFENCE;
    STC_CLWB(13); SFENCE;
    STC_CLWB(14); SFENCE;
    STC_CLWB(15); SFENCE;
}

__global__ void mysgemmNT_nvmu( const float *A, int lda, const float *B, int ldb, float* C, int ldc, int k, float alpha, float beta )
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
    STC_CLWB(0); SFENCE; PCOMMIT; SFENCE;
    STC_CLWB(1); SFENCE; PCOMMIT; SFENCE;
    STC_CLWB(2); SFENCE; PCOMMIT; SFENCE;
    STC_CLWB(3); SFENCE; PCOMMIT; SFENCE;
    STC_CLWB(4); SFENCE; PCOMMIT; SFENCE;
    STC_CLWB(5); SFENCE; PCOMMIT; SFENCE;
    STC_CLWB(6); SFENCE; PCOMMIT; SFENCE;
    STC_CLWB(7); SFENCE; PCOMMIT; SFENCE;
    STC_CLWB(8); SFENCE; PCOMMIT; SFENCE;
    STC_CLWB(9); SFENCE; PCOMMIT; SFENCE;
    STC_CLWB(10); SFENCE; PCOMMIT; SFENCE;
    STC_CLWB(11); SFENCE; PCOMMIT; SFENCE;
    STC_CLWB(12); SFENCE; PCOMMIT; SFENCE;
    STC_CLWB(13); SFENCE; PCOMMIT; SFENCE;
    STC_CLWB(14); SFENCE; PCOMMIT; SFENCE;
    STC_CLWB(15); SFENCE; PCOMMIT; SFENCE;
}


__global__ void mysgemmNT_nvmq( const float *A, int lda, const float *B, int ldb, float* C, int ldc, int k, float alpha, float beta )
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
    for (int i = 0; i < TILE_N; i++) {
      float result = C[t+i*ldc] * beta + alpha * c[i];
      C[t+i*ldc] = result;
    }
    for (int i = 0; i < TILE_N; i++) {
      CLWB(&C[t+i*ldc]);
    }
    SFENCE;

}

__global__ void mysgemmNT_nvmw( const float *A, int lda, const float *B, int ldb, float* C, int ldc, int k, float alpha, float beta )
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
    for (int i = 0; i < TILE_N; i++) {
      float result = C[t+i*ldc] * beta + alpha * c[i];
      C[t+i*ldc] = result;
    }
    for (int i = 0; i < TILE_N; i++) {
      CLWB(&C[t+i*ldc]);
    }
    SFENCE; PCOMMIT; SFENCE;

}

__global__ void mysgemmNT_nvm1( const float *A, int lda, const float *B, int ldb, float* C, int ldc, int k, float alpha, float beta )
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
    for (int i = 0; i < TILE_N; i++) {
      float result = C[t+i*ldc] * beta + alpha * c[i];
      C[t+i*ldc] = result;
    }
    L2WB;
    SFENCE;

}
__global__ void mysgemmNT_nvm2( const float *A, int lda, const float *B, int ldb, float* C, int ldc, int k, float alpha, float beta )
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
    for (int i = 0; i < TILE_N; i++) {
      float result = C[t+i*ldc] * beta + alpha * c[i];
      C[t+i*ldc] = result;
    }
    L2WB;
    SFENCE; PCOMMIT; SFENCE;

}


__global__ void mysgemmNT_nvmj( const float *A, int lda, const float *B, int ldb, float* C, int ldc, int k, float alpha, float beta )
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
    for (int i = 0; i < TILE_N; i++) {
      //C[t+i*ldc] = C[t+i*ldc] * beta + alpha * c[i];
      float result = C[t+i*ldc] * beta + alpha * c[i];
      ST_WT_FLOAT(&C[t+i*ldc], result);
      SFENCE; PCOMMIT; SFENCE;
    }
}

__global__ void mysgemmNT_nvml( const float *A, int lda, const float *B, int ldb, float* C, int ldc, int k, float alpha, float beta )
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
    for (int i = 0; i < TILE_N; i++) {
      //C[t+i*ldc] = C[t+i*ldc] * beta + alpha * c[i];
      float result = C[t+i*ldc] * beta + alpha * c[i];
      ST_WT_FLOAT(&C[t+i*ldc], result);
    }
    SFENCE; PCOMMIT; SFENCE;
}
