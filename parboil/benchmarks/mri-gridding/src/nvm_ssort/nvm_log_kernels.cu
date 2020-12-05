__device__ unsigned int NVM_log1[LOG_SIZE_16M];
__device__ unsigned int NVM_log2[LOG_SIZE_16M];
__device__ unsigned int NVM_log3[LOG_SIZE_16M];
__device__ unsigned int NVM_flag[FLAG_SIZE_1M];

__global__ static void splitSort_nvmg(int numElems, int iter, unsigned int* keys, unsigned int* values, unsigned int* histo)
{
    __shared__ unsigned int flags[BLOCK_P_OFFSET];
    __shared__ unsigned int histo_s[1<<BITS];

    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x*4*SORT_BS+4*threadIdx.x;

    // Copy input to shared mem. Assumes input is always even numbered
    uint4 lkey = { UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
    uint4 lvalue;
    if (gid < numElems){
      lkey = *((uint4*)(keys+gid));
      lvalue = *((uint4*)(values+gid));
    }

    if(tid < (1<<BITS)){
      histo_s[tid] = 0;
    }
    __syncthreads();

    histo_s[((lkey.x&((1<<(BITS*(iter+1)))-1))>>(BITS*iter))] += 1;
    histo_s[((lkey.y&((1<<(BITS*(iter+1)))-1))>>(BITS*iter))] += 1;
    histo_s[((lkey.z&((1<<(BITS*(iter+1)))-1))>>(BITS*iter))] += 1;
    histo_s[((lkey.w&((1<<(BITS*(iter+1)))-1))>>(BITS*iter))] += 1;

    uint4 index = {4*tid, 4*tid+1, 4*tid+2, 4*tid+3};

    for (int i=BITS*iter; i<BITS*(iter+1);i++){
      const uint4 flag = {(lkey.x>>i)&0x1,(lkey.y>>i)&0x1,(lkey.z>>i)&0x1,(lkey.w>>i)&0x1};

      flags[index.x+CONFLICT_FREE_OFFSET(index.x)] = 1<<(16*flag.x);
      flags[index.y+CONFLICT_FREE_OFFSET(index.y)] = 1<<(16*flag.y);
      flags[index.z+CONFLICT_FREE_OFFSET(index.z)] = 1<<(16*flag.z);
      flags[index.w+CONFLICT_FREE_OFFSET(index.w)] = 1<<(16*flag.w);

      scan (flags);

      index.x = (flags[index.x+CONFLICT_FREE_OFFSET(index.x)]>>(16*flag.x))&0xFFFF;
      index.y = (flags[index.y+CONFLICT_FREE_OFFSET(index.y)]>>(16*flag.y))&0xFFFF;
      index.z = (flags[index.z+CONFLICT_FREE_OFFSET(index.z)]>>(16*flag.z))&0xFFFF;
      index.w = (flags[index.w+CONFLICT_FREE_OFFSET(index.w)]>>(16*flag.w))&0xFFFF;

      unsigned short offset = flags[4*blockDim.x+CONFLICT_FREE_OFFSET(4*blockDim.x)]&0xFFFF;
      index.x += (flag.x) ? offset : 0;
      index.y += (flag.y) ? offset : 0;
      index.z += (flag.z) ? offset : 0;
      index.w += (flag.w) ? offset : 0;

      __syncthreads();
    }

    // write to log
    if (gid < numElems) {
      ST_WT_INT(&NVM_log1[blockIdx.x*4*SORT_BS+index.x], keys[blockIdx.x*4*SORT_BS+index.x]);
      ST_WT_INT(&NVM_log1[blockIdx.x*4*SORT_BS+index.y], keys[blockIdx.x*4*SORT_BS+index.y]);
      ST_WT_INT(&NVM_log1[blockIdx.x*4*SORT_BS+index.z], keys[blockIdx.x*4*SORT_BS+index.y]);
      ST_WT_INT(&NVM_log1[blockIdx.x*4*SORT_BS+index.w], keys[blockIdx.x*4*SORT_BS+index.w]);
      ST_WT_INT(&NVM_log2[blockIdx.x*4*SORT_BS+index.x], values[blockIdx.x*4*SORT_BS+index.x]);
      ST_WT_INT(&NVM_log2[blockIdx.x*4*SORT_BS+index.y], values[blockIdx.x*4*SORT_BS+index.y]);
      ST_WT_INT(&NVM_log2[blockIdx.x*4*SORT_BS+index.z], values[blockIdx.x*4*SORT_BS+index.z]);
      ST_WT_INT(&NVM_log2[blockIdx.x*4*SORT_BS+index.w], values[blockIdx.x*4*SORT_BS+index.w]);
    }
    if (tid < (1<<BITS)) {
      ST_WT_INT(&NVM_log3[gridDim.x*threadIdx.x+blockIdx.x], histo[gridDim.x*threadIdx.x+blockIdx.x]);      
    }
    MEM_FENCE; __syncthreads();
    SET_NVM_FLAG(1);

    // Write result.
    if (gid < numElems){
      ST_WT_INT(&keys[blockIdx.x*4*SORT_BS+index.x], lkey.x);
      ST_WT_INT(&keys[blockIdx.x*4*SORT_BS+index.y], lkey.y);
      ST_WT_INT(&keys[blockIdx.x*4*SORT_BS+index.z], lkey.z);
      ST_WT_INT(&keys[blockIdx.x*4*SORT_BS+index.w], lkey.w);
      ST_WT_INT(&values[blockIdx.x*4*SORT_BS+index.x], lvalue.x);
      ST_WT_INT(&values[blockIdx.x*4*SORT_BS+index.y], lvalue.y);
      ST_WT_INT(&values[blockIdx.x*4*SORT_BS+index.z], lvalue.z);
      ST_WT_INT(&values[blockIdx.x*4*SORT_BS+index.w], lvalue.w);
    }
    if (tid < (1<<BITS)){
      ST_WT_INT(&histo[gridDim.x*threadIdx.x+blockIdx.x], histo_s[tid]);
      MEM_FENCE;
    }
    MEM_FENCE; __syncthreads();
    SET_NVM_FLAG(2);
}

__global__ static void splitSort_nvm3(int numElems, int iter, unsigned int* keys, unsigned int* values, unsigned int* histo)
{
    __shared__ unsigned int flags[BLOCK_P_OFFSET];
    __shared__ unsigned int histo_s[1<<BITS];

    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x*4*SORT_BS+4*threadIdx.x;

    // Copy input to shared mem. Assumes input is always even numbered
    uint4 lkey = { UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
    uint4 lvalue;
    if (gid < numElems){
      lkey = *((uint4*)(keys+gid));
      lvalue = *((uint4*)(values+gid));
    }

    if(tid < (1<<BITS)){
      histo_s[tid] = 0;
    }
    __syncthreads();

    histo_s[((lkey.x&((1<<(BITS*(iter+1)))-1))>>(BITS*iter))] += 1;
    histo_s[((lkey.y&((1<<(BITS*(iter+1)))-1))>>(BITS*iter))] += 1;
    histo_s[((lkey.z&((1<<(BITS*(iter+1)))-1))>>(BITS*iter))] += 1;
    histo_s[((lkey.w&((1<<(BITS*(iter+1)))-1))>>(BITS*iter))] += 1;

    uint4 index = {4*tid, 4*tid+1, 4*tid+2, 4*tid+3};

    for (int i=BITS*iter; i<BITS*(iter+1);i++){
      const uint4 flag = {(lkey.x>>i)&0x1,(lkey.y>>i)&0x1,(lkey.z>>i)&0x1,(lkey.w>>i)&0x1};

      flags[index.x+CONFLICT_FREE_OFFSET(index.x)] = 1<<(16*flag.x);
      flags[index.y+CONFLICT_FREE_OFFSET(index.y)] = 1<<(16*flag.y);
      flags[index.z+CONFLICT_FREE_OFFSET(index.z)] = 1<<(16*flag.z);
      flags[index.w+CONFLICT_FREE_OFFSET(index.w)] = 1<<(16*flag.w);

      scan (flags);

      index.x = (flags[index.x+CONFLICT_FREE_OFFSET(index.x)]>>(16*flag.x))&0xFFFF;
      index.y = (flags[index.y+CONFLICT_FREE_OFFSET(index.y)]>>(16*flag.y))&0xFFFF;
      index.z = (flags[index.z+CONFLICT_FREE_OFFSET(index.z)]>>(16*flag.z))&0xFFFF;
      index.w = (flags[index.w+CONFLICT_FREE_OFFSET(index.w)]>>(16*flag.w))&0xFFFF;

      unsigned short offset = flags[4*blockDim.x+CONFLICT_FREE_OFFSET(4*blockDim.x)]&0xFFFF;
      index.x += (flag.x) ? offset : 0;
      index.y += (flag.y) ? offset : 0;
      index.z += (flag.z) ? offset : 0;
      index.w += (flag.w) ? offset : 0;

      __syncthreads();
    }

    // write to log
    if (gid < numElems) {
      ST_WB_INT(&NVM_log1[blockIdx.x*4*SORT_BS+index.x], keys[blockIdx.x*4*SORT_BS+index.x]);
      ST_WB_INT(&NVM_log1[blockIdx.x*4*SORT_BS+index.y], keys[blockIdx.x*4*SORT_BS+index.y]);
      ST_WB_INT(&NVM_log1[blockIdx.x*4*SORT_BS+index.z], keys[blockIdx.x*4*SORT_BS+index.y]);
      ST_WB_INT(&NVM_log1[blockIdx.x*4*SORT_BS+index.w], keys[blockIdx.x*4*SORT_BS+index.w]);
      ST_WB_INT(&NVM_log2[blockIdx.x*4*SORT_BS+index.x], values[blockIdx.x*4*SORT_BS+index.x]);
      ST_WB_INT(&NVM_log2[blockIdx.x*4*SORT_BS+index.y], values[blockIdx.x*4*SORT_BS+index.y]);
      ST_WB_INT(&NVM_log2[blockIdx.x*4*SORT_BS+index.z], values[blockIdx.x*4*SORT_BS+index.z]);
      ST_WB_INT(&NVM_log2[blockIdx.x*4*SORT_BS+index.w], values[blockIdx.x*4*SORT_BS+index.w]);
      CLWB(&NVM_log1[blockIdx.x*4*SORT_BS+index.x]);
      CLWB(&NVM_log1[blockIdx.x*4*SORT_BS+index.y]);
      CLWB(&NVM_log1[blockIdx.x*4*SORT_BS+index.z]);
      CLWB(&NVM_log1[blockIdx.x*4*SORT_BS+index.w]);
      CLWB(&NVM_log2[blockIdx.x*4*SORT_BS+index.x]);
      CLWB(&NVM_log2[blockIdx.x*4*SORT_BS+index.y]);
      CLWB(&NVM_log2[blockIdx.x*4*SORT_BS+index.z]);
      CLWB(&NVM_log2[blockIdx.x*4*SORT_BS+index.w]);
    }
    if (tid < (1<<BITS)) {
      ST_WB_INT(&NVM_log3[gridDim.x*threadIdx.x+blockIdx.x], histo[gridDim.x*threadIdx.x+blockIdx.x]);
      CLWB(&NVM_log3[gridDim.x*threadIdx.x+blockIdx.x]);
    }
    MEM_FENCE; __syncthreads();
    SET_NVM_FLAG_WB(1);

    // Write result.
    if (gid < numElems){
      keys[blockIdx.x*4*SORT_BS+index.x] = lkey.x;
      keys[blockIdx.x*4*SORT_BS+index.y] = lkey.y;
      keys[blockIdx.x*4*SORT_BS+index.z] = lkey.z;
      keys[blockIdx.x*4*SORT_BS+index.w] = lkey.w;

      values[blockIdx.x*4*SORT_BS+index.x] = lvalue.x;
      values[blockIdx.x*4*SORT_BS+index.y] = lvalue.y;
      values[blockIdx.x*4*SORT_BS+index.z] = lvalue.z;
      values[blockIdx.x*4*SORT_BS+index.w] = lvalue.w;
    }
    if (tid < (1<<BITS)){
      histo[gridDim.x*threadIdx.x+blockIdx.x] = histo_s[tid];
    }
    CLWB(&keys[blockIdx.x*4*SORT_BS+index.x]); 
    CLWB(&keys[blockIdx.x*4*SORT_BS+index.y]); 
    CLWB(&keys[blockIdx.x*4*SORT_BS+index.z]); 
    CLWB(&keys[blockIdx.x*4*SORT_BS+index.w]); 
    CLWB(&values[blockIdx.x*4*SORT_BS+index.x]); 
    CLWB(&values[blockIdx.x*4*SORT_BS+index.w]); 
    CLWB(&values[blockIdx.x*4*SORT_BS+index.z]); 
    CLWB(&values[blockIdx.x*4*SORT_BS+index.y]); 
    CLWB(&histo[gridDim.x*threadIdx.x+blockIdx.x]);
    MEM_FENCE; __syncthreads();
    SET_NVM_FLAG_WB(2);
}

__global__ static void splitSort_nvm4(int numElems, int iter, unsigned int* keys, unsigned int* values, unsigned int* histo)
{
    __shared__ unsigned int flags[BLOCK_P_OFFSET];
    __shared__ unsigned int histo_s[1<<BITS];

    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x*4*SORT_BS+4*threadIdx.x;

    // Copy input to shared mem. Assumes input is always even numbered
    uint4 lkey = { UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
    uint4 lvalue;
    if (gid < numElems){
      lkey = *((uint4*)(keys+gid));
      lvalue = *((uint4*)(values+gid));
    }

    if(tid < (1<<BITS)){
      histo_s[tid] = 0;
    }
    __syncthreads();

    histo_s[((lkey.x&((1<<(BITS*(iter+1)))-1))>>(BITS*iter))] += 1;
    histo_s[((lkey.y&((1<<(BITS*(iter+1)))-1))>>(BITS*iter))] += 1;
    histo_s[((lkey.z&((1<<(BITS*(iter+1)))-1))>>(BITS*iter))] += 1;
    histo_s[((lkey.w&((1<<(BITS*(iter+1)))-1))>>(BITS*iter))] += 1;

    uint4 index = {4*tid, 4*tid+1, 4*tid+2, 4*tid+3};

    for (int i=BITS*iter; i<BITS*(iter+1);i++){
      const uint4 flag = {(lkey.x>>i)&0x1,(lkey.y>>i)&0x1,(lkey.z>>i)&0x1,(lkey.w>>i)&0x1};

      flags[index.x+CONFLICT_FREE_OFFSET(index.x)] = 1<<(16*flag.x);
      flags[index.y+CONFLICT_FREE_OFFSET(index.y)] = 1<<(16*flag.y);
      flags[index.z+CONFLICT_FREE_OFFSET(index.z)] = 1<<(16*flag.z);
      flags[index.w+CONFLICT_FREE_OFFSET(index.w)] = 1<<(16*flag.w);

      scan (flags);

      index.x = (flags[index.x+CONFLICT_FREE_OFFSET(index.x)]>>(16*flag.x))&0xFFFF;
      index.y = (flags[index.y+CONFLICT_FREE_OFFSET(index.y)]>>(16*flag.y))&0xFFFF;
      index.z = (flags[index.z+CONFLICT_FREE_OFFSET(index.z)]>>(16*flag.z))&0xFFFF;
      index.w = (flags[index.w+CONFLICT_FREE_OFFSET(index.w)]>>(16*flag.w))&0xFFFF;

      unsigned short offset = flags[4*blockDim.x+CONFLICT_FREE_OFFSET(4*blockDim.x)]&0xFFFF;
      index.x += (flag.x) ? offset : 0;
      index.y += (flag.y) ? offset : 0;
      index.z += (flag.z) ? offset : 0;
      index.w += (flag.w) ? offset : 0;

      __syncthreads();
    }

    // write to log
    if (gid < numElems) {
      ST_WB_INT(&NVM_log1[blockIdx.x*4*SORT_BS+index.x], keys[blockIdx.x*4*SORT_BS+index.x]);
      ST_WB_INT(&NVM_log1[blockIdx.x*4*SORT_BS+index.y], keys[blockIdx.x*4*SORT_BS+index.y]);
      ST_WB_INT(&NVM_log1[blockIdx.x*4*SORT_BS+index.z], keys[blockIdx.x*4*SORT_BS+index.y]);
      ST_WB_INT(&NVM_log1[blockIdx.x*4*SORT_BS+index.w], keys[blockIdx.x*4*SORT_BS+index.w]);
      ST_WB_INT(&NVM_log2[blockIdx.x*4*SORT_BS+index.x], values[blockIdx.x*4*SORT_BS+index.x]);
      ST_WB_INT(&NVM_log2[blockIdx.x*4*SORT_BS+index.y], values[blockIdx.x*4*SORT_BS+index.y]);
      ST_WB_INT(&NVM_log2[blockIdx.x*4*SORT_BS+index.z], values[blockIdx.x*4*SORT_BS+index.z]);
      ST_WB_INT(&NVM_log2[blockIdx.x*4*SORT_BS+index.w], values[blockIdx.x*4*SORT_BS+index.w]);
      CLWB(&NVM_log1[blockIdx.x*4*SORT_BS+index.x]);
      CLWB(&NVM_log1[blockIdx.x*4*SORT_BS+index.y]);
      CLWB(&NVM_log1[blockIdx.x*4*SORT_BS+index.z]);
      CLWB(&NVM_log1[blockIdx.x*4*SORT_BS+index.w]);
      CLWB(&NVM_log2[blockIdx.x*4*SORT_BS+index.x]);
      CLWB(&NVM_log2[blockIdx.x*4*SORT_BS+index.y]);
      CLWB(&NVM_log2[blockIdx.x*4*SORT_BS+index.z]);
      CLWB(&NVM_log2[blockIdx.x*4*SORT_BS+index.w]);
    }
    if (tid < (1<<BITS)) {
      ST_WB_INT(&NVM_log3[gridDim.x*threadIdx.x+blockIdx.x], histo[gridDim.x*threadIdx.x+blockIdx.x]);
      CLWB(&NVM_log3[gridDim.x*threadIdx.x+blockIdx.x]);
    }
    MEM_FENCE; PCOMMIT; MEM_FENCE; __syncthreads();
    SET_NVM_FLAG_WB_PC(1);

    // Write result.
    if (gid < numElems){
      keys[blockIdx.x*4*SORT_BS+index.x] = lkey.x;
      keys[blockIdx.x*4*SORT_BS+index.y] = lkey.y;
      keys[blockIdx.x*4*SORT_BS+index.z] = lkey.z;
      keys[blockIdx.x*4*SORT_BS+index.w] = lkey.w;

      values[blockIdx.x*4*SORT_BS+index.x] = lvalue.x;
      values[blockIdx.x*4*SORT_BS+index.y] = lvalue.y;
      values[blockIdx.x*4*SORT_BS+index.z] = lvalue.z;
      values[blockIdx.x*4*SORT_BS+index.w] = lvalue.w;
    }
    if (tid < (1<<BITS)){
      histo[gridDim.x*threadIdx.x+blockIdx.x] = histo_s[tid];
    }
    CLWB(&keys[blockIdx.x*4*SORT_BS+index.x]); 
    CLWB(&keys[blockIdx.x*4*SORT_BS+index.y]); 
    CLWB(&keys[blockIdx.x*4*SORT_BS+index.z]); 
    CLWB(&keys[blockIdx.x*4*SORT_BS+index.w]); 
    CLWB(&values[blockIdx.x*4*SORT_BS+index.x]); 
    CLWB(&values[blockIdx.x*4*SORT_BS+index.w]); 
    CLWB(&values[blockIdx.x*4*SORT_BS+index.z]); 
    CLWB(&values[blockIdx.x*4*SORT_BS+index.y]); 
    CLWB(&histo[gridDim.x*threadIdx.x+blockIdx.x]);
    MEM_FENCE; PCOMMIT; MEM_FENCE; __syncthreads();
    SET_NVM_FLAG_WB_PC(2);
}


__global__ static void splitSort_nvmi(int numElems, int iter, unsigned int* keys, unsigned int* values, unsigned int* histo)
{
    __shared__ unsigned int flags[BLOCK_P_OFFSET];
    __shared__ unsigned int histo_s[1<<BITS];

    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x*4*SORT_BS+4*threadIdx.x;

    // Copy input to shared mem. Assumes input is always even numbered
    uint4 lkey = { UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
    uint4 lvalue;
    if (gid < numElems){
      lkey = *((uint4*)(keys+gid));
      lvalue = *((uint4*)(values+gid));
    }

    if(tid < (1<<BITS)){
      histo_s[tid] = 0;
    }
    __syncthreads();

    histo_s[((lkey.x&((1<<(BITS*(iter+1)))-1))>>(BITS*iter))] += 1;
    histo_s[((lkey.y&((1<<(BITS*(iter+1)))-1))>>(BITS*iter))] += 1;
    histo_s[((lkey.z&((1<<(BITS*(iter+1)))-1))>>(BITS*iter))] += 1;
    histo_s[((lkey.w&((1<<(BITS*(iter+1)))-1))>>(BITS*iter))] += 1;

    uint4 index = {4*tid, 4*tid+1, 4*tid+2, 4*tid+3};

    for (int i=BITS*iter; i<BITS*(iter+1);i++){
      const uint4 flag = {(lkey.x>>i)&0x1,(lkey.y>>i)&0x1,(lkey.z>>i)&0x1,(lkey.w>>i)&0x1};

      flags[index.x+CONFLICT_FREE_OFFSET(index.x)] = 1<<(16*flag.x);
      flags[index.y+CONFLICT_FREE_OFFSET(index.y)] = 1<<(16*flag.y);
      flags[index.z+CONFLICT_FREE_OFFSET(index.z)] = 1<<(16*flag.z);
      flags[index.w+CONFLICT_FREE_OFFSET(index.w)] = 1<<(16*flag.w);

      scan (flags);

      index.x = (flags[index.x+CONFLICT_FREE_OFFSET(index.x)]>>(16*flag.x))&0xFFFF;
      index.y = (flags[index.y+CONFLICT_FREE_OFFSET(index.y)]>>(16*flag.y))&0xFFFF;
      index.z = (flags[index.z+CONFLICT_FREE_OFFSET(index.z)]>>(16*flag.z))&0xFFFF;
      index.w = (flags[index.w+CONFLICT_FREE_OFFSET(index.w)]>>(16*flag.w))&0xFFFF;

      unsigned short offset = flags[4*blockDim.x+CONFLICT_FREE_OFFSET(4*blockDim.x)]&0xFFFF;
      index.x += (flag.x) ? offset : 0;
      index.y += (flag.y) ? offset : 0;
      index.z += (flag.z) ? offset : 0;
      index.w += (flag.w) ? offset : 0;

      __syncthreads();
    }

    // write to log
    if (gid < numElems) {
      ST_WT_INT(&NVM_log1[blockIdx.x*4*SORT_BS+index.x], keys[blockIdx.x*4*SORT_BS+index.x]);
      ST_WT_INT(&NVM_log1[blockIdx.x*4*SORT_BS+index.y], keys[blockIdx.x*4*SORT_BS+index.y]);
      ST_WT_INT(&NVM_log1[blockIdx.x*4*SORT_BS+index.z], keys[blockIdx.x*4*SORT_BS+index.y]);
      ST_WT_INT(&NVM_log1[blockIdx.x*4*SORT_BS+index.w], keys[blockIdx.x*4*SORT_BS+index.w]);
      ST_WT_INT(&NVM_log2[blockIdx.x*4*SORT_BS+index.x], values[blockIdx.x*4*SORT_BS+index.x]);
      ST_WT_INT(&NVM_log2[blockIdx.x*4*SORT_BS+index.y], values[blockIdx.x*4*SORT_BS+index.y]);
      ST_WT_INT(&NVM_log2[blockIdx.x*4*SORT_BS+index.z], values[blockIdx.x*4*SORT_BS+index.z]);
      ST_WT_INT(&NVM_log2[blockIdx.x*4*SORT_BS+index.w], values[blockIdx.x*4*SORT_BS+index.w]);
    }
    MEM_FENCE; __syncthreads();
    SET_NVM_FLAG(1);

    // Write result.
    if (gid < numElems){
      ST_WT_INT(&keys[blockIdx.x*4*SORT_BS+index.x], lkey.x);
      ST_WT_INT(&keys[blockIdx.x*4*SORT_BS+index.y], lkey.y);
      ST_WT_INT(&keys[blockIdx.x*4*SORT_BS+index.z], lkey.z);
      ST_WT_INT(&keys[blockIdx.x*4*SORT_BS+index.w], lkey.w);
      ST_WT_INT(&values[blockIdx.x*4*SORT_BS+index.x], lvalue.x);
      ST_WT_INT(&values[blockIdx.x*4*SORT_BS+index.y], lvalue.y);
      ST_WT_INT(&values[blockIdx.x*4*SORT_BS+index.z], lvalue.z);
      ST_WT_INT(&values[blockIdx.x*4*SORT_BS+index.w], lvalue.w);
    }
    if (tid < (1<<BITS)){
      ST_WT_INT(&histo[gridDim.x*threadIdx.x+blockIdx.x], histo_s[tid]);
      MEM_FENCE;
    }
    MEM_FENCE; __syncthreads();
    SET_NVM_FLAG(2);
}

__global__ static void splitSort_nvm5(int numElems, int iter, unsigned int* keys, unsigned int* values, unsigned int* histo)
{
    __shared__ unsigned int flags[BLOCK_P_OFFSET];
    __shared__ unsigned int histo_s[1<<BITS];

    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x*4*SORT_BS+4*threadIdx.x;

    // Copy input to shared mem. Assumes input is always even numbered
    uint4 lkey = { UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
    uint4 lvalue;
    if (gid < numElems){
      lkey = *((uint4*)(keys+gid));
      lvalue = *((uint4*)(values+gid));
    }

    if(tid < (1<<BITS)){
      histo_s[tid] = 0;
    }
    __syncthreads();

    histo_s[((lkey.x&((1<<(BITS*(iter+1)))-1))>>(BITS*iter))] += 1;
    histo_s[((lkey.y&((1<<(BITS*(iter+1)))-1))>>(BITS*iter))] += 1;
    histo_s[((lkey.z&((1<<(BITS*(iter+1)))-1))>>(BITS*iter))] += 1;
    histo_s[((lkey.w&((1<<(BITS*(iter+1)))-1))>>(BITS*iter))] += 1;

    uint4 index = {4*tid, 4*tid+1, 4*tid+2, 4*tid+3};

    for (int i=BITS*iter; i<BITS*(iter+1);i++){
      const uint4 flag = {(lkey.x>>i)&0x1,(lkey.y>>i)&0x1,(lkey.z>>i)&0x1,(lkey.w>>i)&0x1};

      flags[index.x+CONFLICT_FREE_OFFSET(index.x)] = 1<<(16*flag.x);
      flags[index.y+CONFLICT_FREE_OFFSET(index.y)] = 1<<(16*flag.y);
      flags[index.z+CONFLICT_FREE_OFFSET(index.z)] = 1<<(16*flag.z);
      flags[index.w+CONFLICT_FREE_OFFSET(index.w)] = 1<<(16*flag.w);

      scan (flags);

      index.x = (flags[index.x+CONFLICT_FREE_OFFSET(index.x)]>>(16*flag.x))&0xFFFF;
      index.y = (flags[index.y+CONFLICT_FREE_OFFSET(index.y)]>>(16*flag.y))&0xFFFF;
      index.z = (flags[index.z+CONFLICT_FREE_OFFSET(index.z)]>>(16*flag.z))&0xFFFF;
      index.w = (flags[index.w+CONFLICT_FREE_OFFSET(index.w)]>>(16*flag.w))&0xFFFF;

      unsigned short offset = flags[4*blockDim.x+CONFLICT_FREE_OFFSET(4*blockDim.x)]&0xFFFF;
      index.x += (flag.x) ? offset : 0;
      index.y += (flag.y) ? offset : 0;
      index.z += (flag.z) ? offset : 0;
      index.w += (flag.w) ? offset : 0;

      __syncthreads();
    }

    // write to log
    if (gid < numElems) {
      ST_WB_INT(&NVM_log1[blockIdx.x*4*SORT_BS+index.x], keys[blockIdx.x*4*SORT_BS+index.x]);
      ST_WB_INT(&NVM_log1[blockIdx.x*4*SORT_BS+index.y], keys[blockIdx.x*4*SORT_BS+index.y]);
      ST_WB_INT(&NVM_log1[blockIdx.x*4*SORT_BS+index.z], keys[blockIdx.x*4*SORT_BS+index.y]);
      ST_WB_INT(&NVM_log1[blockIdx.x*4*SORT_BS+index.w], keys[blockIdx.x*4*SORT_BS+index.w]);
      ST_WB_INT(&NVM_log2[blockIdx.x*4*SORT_BS+index.x], values[blockIdx.x*4*SORT_BS+index.x]);
      ST_WB_INT(&NVM_log2[blockIdx.x*4*SORT_BS+index.y], values[blockIdx.x*4*SORT_BS+index.y]);
      ST_WB_INT(&NVM_log2[blockIdx.x*4*SORT_BS+index.z], values[blockIdx.x*4*SORT_BS+index.z]);
      ST_WB_INT(&NVM_log2[blockIdx.x*4*SORT_BS+index.w], values[blockIdx.x*4*SORT_BS+index.w]);
    }
    MEM_FENCE; __syncthreads();
    SET_NVM_FLAG_WB(1);

    // Write result.
    if (gid < numElems){
      keys[blockIdx.x*4*SORT_BS+index.x] = lkey.x;
      keys[blockIdx.x*4*SORT_BS+index.y] = lkey.y;
      keys[blockIdx.x*4*SORT_BS+index.z] = lkey.z;
      keys[blockIdx.x*4*SORT_BS+index.w] = lkey.w;

      values[blockIdx.x*4*SORT_BS+index.x] = lvalue.x;
      values[blockIdx.x*4*SORT_BS+index.y] = lvalue.y;
      values[blockIdx.x*4*SORT_BS+index.z] = lvalue.z;
      values[blockIdx.x*4*SORT_BS+index.w] = lvalue.w;
    }
    if (tid < (1<<BITS)){
      histo[gridDim.x*threadIdx.x+blockIdx.x] = histo_s[tid];
    }
    CLWB(&keys[blockIdx.x*4*SORT_BS+index.x]); 
    CLWB(&keys[blockIdx.x*4*SORT_BS+index.y]); 
    CLWB(&keys[blockIdx.x*4*SORT_BS+index.z]); 
    CLWB(&keys[blockIdx.x*4*SORT_BS+index.w]); 
    CLWB(&values[blockIdx.x*4*SORT_BS+index.x]); 
    CLWB(&values[blockIdx.x*4*SORT_BS+index.w]); 
    CLWB(&values[blockIdx.x*4*SORT_BS+index.z]); 
    CLWB(&values[blockIdx.x*4*SORT_BS+index.y]); 
    CLWB(&histo[gridDim.x*threadIdx.x+blockIdx.x]);
    MEM_FENCE; __syncthreads();
    SET_NVM_FLAG_WB(2);
}
__global__ static void splitSort_nvm6(int numElems, int iter, unsigned int* keys, unsigned int* values, unsigned int* histo)
{
    __shared__ unsigned int flags[BLOCK_P_OFFSET];
    __shared__ unsigned int histo_s[1<<BITS];

    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x*4*SORT_BS+4*threadIdx.x;

    // Copy input to shared mem. Assumes input is always even numbered
    uint4 lkey = { UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
    uint4 lvalue;
    if (gid < numElems){
      lkey = *((uint4*)(keys+gid));
      lvalue = *((uint4*)(values+gid));
    }

    if(tid < (1<<BITS)){
      histo_s[tid] = 0;
    }
    __syncthreads();

    histo_s[((lkey.x&((1<<(BITS*(iter+1)))-1))>>(BITS*iter))] += 1;
    histo_s[((lkey.y&((1<<(BITS*(iter+1)))-1))>>(BITS*iter))] += 1;
    histo_s[((lkey.z&((1<<(BITS*(iter+1)))-1))>>(BITS*iter))] += 1;
    histo_s[((lkey.w&((1<<(BITS*(iter+1)))-1))>>(BITS*iter))] += 1;

    uint4 index = {4*tid, 4*tid+1, 4*tid+2, 4*tid+3};

    for (int i=BITS*iter; i<BITS*(iter+1);i++){
      const uint4 flag = {(lkey.x>>i)&0x1,(lkey.y>>i)&0x1,(lkey.z>>i)&0x1,(lkey.w>>i)&0x1};

      flags[index.x+CONFLICT_FREE_OFFSET(index.x)] = 1<<(16*flag.x);
      flags[index.y+CONFLICT_FREE_OFFSET(index.y)] = 1<<(16*flag.y);
      flags[index.z+CONFLICT_FREE_OFFSET(index.z)] = 1<<(16*flag.z);
      flags[index.w+CONFLICT_FREE_OFFSET(index.w)] = 1<<(16*flag.w);

      scan (flags);

      index.x = (flags[index.x+CONFLICT_FREE_OFFSET(index.x)]>>(16*flag.x))&0xFFFF;
      index.y = (flags[index.y+CONFLICT_FREE_OFFSET(index.y)]>>(16*flag.y))&0xFFFF;
      index.z = (flags[index.z+CONFLICT_FREE_OFFSET(index.z)]>>(16*flag.z))&0xFFFF;
      index.w = (flags[index.w+CONFLICT_FREE_OFFSET(index.w)]>>(16*flag.w))&0xFFFF;

      unsigned short offset = flags[4*blockDim.x+CONFLICT_FREE_OFFSET(4*blockDim.x)]&0xFFFF;
      index.x += (flag.x) ? offset : 0;
      index.y += (flag.y) ? offset : 0;
      index.z += (flag.z) ? offset : 0;
      index.w += (flag.w) ? offset : 0;

      __syncthreads();
    }

    // write to log
    if (gid < numElems) {
      ST_WB_INT(&NVM_log1[blockIdx.x*4*SORT_BS+index.x], keys[blockIdx.x*4*SORT_BS+index.x]);
      ST_WB_INT(&NVM_log1[blockIdx.x*4*SORT_BS+index.y], keys[blockIdx.x*4*SORT_BS+index.y]);
      ST_WB_INT(&NVM_log1[blockIdx.x*4*SORT_BS+index.z], keys[blockIdx.x*4*SORT_BS+index.y]);
      ST_WB_INT(&NVM_log1[blockIdx.x*4*SORT_BS+index.w], keys[blockIdx.x*4*SORT_BS+index.w]);
      ST_WB_INT(&NVM_log2[blockIdx.x*4*SORT_BS+index.x], values[blockIdx.x*4*SORT_BS+index.x]);
      ST_WB_INT(&NVM_log2[blockIdx.x*4*SORT_BS+index.y], values[blockIdx.x*4*SORT_BS+index.y]);
      ST_WB_INT(&NVM_log2[blockIdx.x*4*SORT_BS+index.z], values[blockIdx.x*4*SORT_BS+index.z]);
      ST_WB_INT(&NVM_log2[blockIdx.x*4*SORT_BS+index.w], values[blockIdx.x*4*SORT_BS+index.w]);
    }
    MEM_FENCE; PCOMMIT; MEM_FENCE; __syncthreads();
    SET_NVM_FLAG_WB_PC(1);

    // Write result.
    if (gid < numElems){
      keys[blockIdx.x*4*SORT_BS+index.x] = lkey.x;
      keys[blockIdx.x*4*SORT_BS+index.y] = lkey.y;
      keys[blockIdx.x*4*SORT_BS+index.z] = lkey.z;
      keys[blockIdx.x*4*SORT_BS+index.w] = lkey.w;

      values[blockIdx.x*4*SORT_BS+index.x] = lvalue.x;
      values[blockIdx.x*4*SORT_BS+index.y] = lvalue.y;
      values[blockIdx.x*4*SORT_BS+index.z] = lvalue.z;
      values[blockIdx.x*4*SORT_BS+index.w] = lvalue.w;
    }
    if (tid < (1<<BITS)){
      histo[gridDim.x*threadIdx.x+blockIdx.x] = histo_s[tid];
    }
    CLWB(&keys[blockIdx.x*4*SORT_BS+index.x]); 
    CLWB(&keys[blockIdx.x*4*SORT_BS+index.y]); 
    CLWB(&keys[blockIdx.x*4*SORT_BS+index.z]); 
    CLWB(&keys[blockIdx.x*4*SORT_BS+index.w]); 
    CLWB(&values[blockIdx.x*4*SORT_BS+index.x]); 
    CLWB(&values[blockIdx.x*4*SORT_BS+index.w]); 
    CLWB(&values[blockIdx.x*4*SORT_BS+index.z]); 
    CLWB(&values[blockIdx.x*4*SORT_BS+index.y]); 
    CLWB(&histo[gridDim.x*threadIdx.x+blockIdx.x]);
    MEM_FENCE; PCOMMIT; MEM_FENCE; __syncthreads();
    SET_NVM_FLAG_WB_PC(2);
}

__global__ void splitRearrange_nvmg (int numElems, int iter, unsigned int* keys_i, unsigned int* keys_o, unsigned int* values_i, unsigned int* values_o, unsigned int* histo){
  __shared__ unsigned int histo_s[(1<<BITS)];
  __shared__ unsigned int array_s[4*SORT_BS];
  int index = blockIdx.x*4*SORT_BS + 4*threadIdx.x;

  if (threadIdx.x < (1<<BITS)){
    histo_s[threadIdx.x] = histo[gridDim.x*threadIdx.x+blockIdx.x];
  }

  uint4 mine, value;
  if (index < numElems){
    mine = *((uint4*)(keys_i+index));
    value = *((uint4*)(values_i+index));
  } else {
    mine.x = UINT32_MAX;
    mine.y = UINT32_MAX;
    mine.z = UINT32_MAX;
    mine.w = UINT32_MAX;
  }
  uint4 masks = {(mine.x&((1<<(BITS*(iter+1)))-1))>>(BITS*iter),
                 (mine.y&((1<<(BITS*(iter+1)))-1))>>(BITS*iter),
                 (mine.z&((1<<(BITS*(iter+1)))-1))>>(BITS*iter),
                 (mine.w&((1<<(BITS*(iter+1)))-1))>>(BITS*iter)};

  ((uint4*)array_s)[threadIdx.x] = masks;
  __syncthreads();

  uint4 new_index = {histo_s[masks.x],histo_s[masks.y],histo_s[masks.z],histo_s[masks.w]};

  int i = 4*threadIdx.x-1;
  while (i >= 0){
    if (array_s[i] == masks.x){
      new_index.x++;
      i--;
    } else {
      break;
    }
  }

  new_index.y = (masks.y == masks.x) ? new_index.x+1 : new_index.y;
  new_index.z = (masks.z == masks.y) ? new_index.y+1 : new_index.z;
  new_index.w = (masks.w == masks.z) ? new_index.z+1 : new_index.w;

  // NVM log
  if (index < numElems){
    ST_WT_INT(&NVM_log1[new_index.x], keys_o[new_index.x]);
    ST_WT_INT(&NVM_log1[new_index.y], keys_o[new_index.y]);
    ST_WT_INT(&NVM_log1[new_index.z], keys_o[new_index.z]);
    ST_WT_INT(&NVM_log1[new_index.w], keys_o[new_index.w]);
    ST_WT_INT(&NVM_log2[new_index.x], values_o[new_index.x]);
    ST_WT_INT(&NVM_log2[new_index.y], values_o[new_index.y]);
    ST_WT_INT(&NVM_log2[new_index.z], values_o[new_index.z]);
    ST_WT_INT(&NVM_log2[new_index.w], values_o[new_index.w]);
  }
  MEM_FENCE; __syncthreads();
  SET_NVM_FLAG(1);

  if (index < numElems){
    ST_WT_INT(&keys_o[new_index.x], mine.x);
    ST_WT_INT(&keys_o[new_index.y], mine.y);
    ST_WT_INT(&keys_o[new_index.z], mine.z);
    ST_WT_INT(&keys_o[new_index.w], mine.w);
    ST_WT_INT(&values_o[new_index.x], value.x);
    ST_WT_INT(&values_o[new_index.y], value.y);
    ST_WT_INT(&values_o[new_index.z], value.z);
    ST_WT_INT(&values_o[new_index.w], value.w);
  }
  MEM_FENCE; __syncthreads();
  SET_NVM_FLAG(2);
}

__global__ void splitRearrange_nvm3 (int numElems, int iter, unsigned int* keys_i, unsigned int* keys_o, unsigned int* values_i, unsigned int* values_o, unsigned int* histo){
  __shared__ unsigned int histo_s[(1<<BITS)];
  __shared__ unsigned int array_s[4*SORT_BS];
  int index = blockIdx.x*4*SORT_BS + 4*threadIdx.x;

  if (threadIdx.x < (1<<BITS)){
    histo_s[threadIdx.x] = histo[gridDim.x*threadIdx.x+blockIdx.x];
  }

  uint4 mine, value;
  if (index < numElems){
    mine = *((uint4*)(keys_i+index));
    value = *((uint4*)(values_i+index));
  } else {
    mine.x = UINT32_MAX;
    mine.y = UINT32_MAX;
    mine.z = UINT32_MAX;
    mine.w = UINT32_MAX;
  }
  uint4 masks = {(mine.x&((1<<(BITS*(iter+1)))-1))>>(BITS*iter),
                 (mine.y&((1<<(BITS*(iter+1)))-1))>>(BITS*iter),
                 (mine.z&((1<<(BITS*(iter+1)))-1))>>(BITS*iter),
                 (mine.w&((1<<(BITS*(iter+1)))-1))>>(BITS*iter)};

  ((uint4*)array_s)[threadIdx.x] = masks;
  __syncthreads();

  uint4 new_index = {histo_s[masks.x],histo_s[masks.y],histo_s[masks.z],histo_s[masks.w]};

  int i = 4*threadIdx.x-1;
  while (i >= 0){
    if (array_s[i] == masks.x){
      new_index.x++;
      i--;
    } else {
      break;
    }
  }

  new_index.y = (masks.y == masks.x) ? new_index.x+1 : new_index.y;
  new_index.z = (masks.z == masks.y) ? new_index.y+1 : new_index.z;
  new_index.w = (masks.w == masks.z) ? new_index.z+1 : new_index.w;

  // NVM log
  if (index < numElems){
    ST_WB_INT(&NVM_log1[new_index.x], keys_o[new_index.x]);
    ST_WB_INT(&NVM_log1[new_index.y], keys_o[new_index.y]);
    ST_WB_INT(&NVM_log1[new_index.z], keys_o[new_index.z]);
    ST_WB_INT(&NVM_log1[new_index.w], keys_o[new_index.w]);
    ST_WB_INT(&NVM_log2[new_index.x], values_o[new_index.x]);
    ST_WB_INT(&NVM_log2[new_index.y], values_o[new_index.y]);
    ST_WB_INT(&NVM_log2[new_index.z], values_o[new_index.z]);
    ST_WB_INT(&NVM_log2[new_index.w], values_o[new_index.w]);
  }
  MEM_FENCE; __syncthreads();
  SET_NVM_FLAG_WB(1);

  if (index < numElems){
    keys_o[new_index.x] = mine.x;
    keys_o[new_index.y] = mine.y;
    keys_o[new_index.z] = mine.z;
    keys_o[new_index.w] = mine.w;
    values_o[new_index.x] = value.x;
    values_o[new_index.y] = value.y;
    values_o[new_index.z] = value.z;
    values_o[new_index.w] = value.w;
    CLWB(&values_o[new_index.w]); 
    CLWB(&keys_o[new_index.x]); 
    CLWB(&keys_o[new_index.y]); 
    CLWB(&keys_o[new_index.z]); 
    CLWB(&keys_o[new_index.w]); 
    CLWB(&values_o[new_index.x]); 
    CLWB(&values_o[new_index.y]); 
    CLWB(&values_o[new_index.z]); 
  }
  MEM_FENCE; __syncthreads();
  SET_NVM_FLAG_WB(2);
}


__global__ void splitRearrange_nvm4 (int numElems, int iter, unsigned int* keys_i, unsigned int* keys_o, unsigned int* values_i, unsigned int* values_o, unsigned int* histo){
  __shared__ unsigned int histo_s[(1<<BITS)];
  __shared__ unsigned int array_s[4*SORT_BS];
  int index = blockIdx.x*4*SORT_BS + 4*threadIdx.x;

  if (threadIdx.x < (1<<BITS)){
    histo_s[threadIdx.x] = histo[gridDim.x*threadIdx.x+blockIdx.x];
  }

  uint4 mine, value;
  if (index < numElems){
    mine = *((uint4*)(keys_i+index));
    value = *((uint4*)(values_i+index));
  } else {
    mine.x = UINT32_MAX;
    mine.y = UINT32_MAX;
    mine.z = UINT32_MAX;
    mine.w = UINT32_MAX;
  }
  uint4 masks = {(mine.x&((1<<(BITS*(iter+1)))-1))>>(BITS*iter),
                 (mine.y&((1<<(BITS*(iter+1)))-1))>>(BITS*iter),
                 (mine.z&((1<<(BITS*(iter+1)))-1))>>(BITS*iter),
                 (mine.w&((1<<(BITS*(iter+1)))-1))>>(BITS*iter)};

  ((uint4*)array_s)[threadIdx.x] = masks;
  __syncthreads();

  uint4 new_index = {histo_s[masks.x],histo_s[masks.y],histo_s[masks.z],histo_s[masks.w]};

  int i = 4*threadIdx.x-1;
  while (i >= 0){
    if (array_s[i] == masks.x){
      new_index.x++;
      i--;
    } else {
      break;
    }
  }

  new_index.y = (masks.y == masks.x) ? new_index.x+1 : new_index.y;
  new_index.z = (masks.z == masks.y) ? new_index.y+1 : new_index.z;
  new_index.w = (masks.w == masks.z) ? new_index.z+1 : new_index.w;

  // NVM log
  if (index < numElems){
    ST_WB_INT(&NVM_log1[new_index.x], keys_o[new_index.x]);
    ST_WB_INT(&NVM_log1[new_index.y], keys_o[new_index.y]);
    ST_WB_INT(&NVM_log1[new_index.z], keys_o[new_index.z]);
    ST_WB_INT(&NVM_log1[new_index.w], keys_o[new_index.w]);
    ST_WB_INT(&NVM_log2[new_index.x], values_o[new_index.x]);
    ST_WB_INT(&NVM_log2[new_index.y], values_o[new_index.y]);
    ST_WB_INT(&NVM_log2[new_index.z], values_o[new_index.z]);
    ST_WB_INT(&NVM_log2[new_index.w], values_o[new_index.w]);
  }
  MEM_FENCE; PCOMMIT; MEM_FENCE; __syncthreads();
  SET_NVM_FLAG_WB_PC(1);

  if (index < numElems){
    keys_o[new_index.x] = mine.x;
    keys_o[new_index.y] = mine.y;
    keys_o[new_index.z] = mine.z;
    keys_o[new_index.w] = mine.w;
    values_o[new_index.x] = value.x;
    values_o[new_index.y] = value.y;
    values_o[new_index.z] = value.z;
    values_o[new_index.w] = value.w;
    CLWB(&values_o[new_index.w]); 
    CLWB(&keys_o[new_index.x]); 
    CLWB(&keys_o[new_index.y]); 
    CLWB(&keys_o[new_index.z]); 
    CLWB(&keys_o[new_index.w]); 
    CLWB(&values_o[new_index.x]); 
    CLWB(&values_o[new_index.y]); 
    CLWB(&values_o[new_index.z]); 
  }
  MEM_FENCE; PCOMMIT; MEM_FENCE; __syncthreads();
  SET_NVM_FLAG_WB_PC(2);
}


__global__ void splitRearrange_nvmi (int numElems, int iter, unsigned int* keys_i, unsigned int* keys_o, unsigned int* values_i, unsigned int* values_o, unsigned int* histo){
  __shared__ unsigned int histo_s[(1<<BITS)];
  __shared__ unsigned int array_s[4*SORT_BS];
  int index = blockIdx.x*4*SORT_BS + 4*threadIdx.x;

  if (threadIdx.x < (1<<BITS)){
    histo_s[threadIdx.x] = histo[gridDim.x*threadIdx.x+blockIdx.x];
  }

  uint4 mine, value;
  if (index < numElems){
    mine = *((uint4*)(keys_i+index));
    value = *((uint4*)(values_i+index));
  } else {
    mine.x = UINT32_MAX;
    mine.y = UINT32_MAX;
    mine.z = UINT32_MAX;
    mine.w = UINT32_MAX;
  }
  uint4 masks = {(mine.x&((1<<(BITS*(iter+1)))-1))>>(BITS*iter),
                 (mine.y&((1<<(BITS*(iter+1)))-1))>>(BITS*iter),
                 (mine.z&((1<<(BITS*(iter+1)))-1))>>(BITS*iter),
                 (mine.w&((1<<(BITS*(iter+1)))-1))>>(BITS*iter)};

  ((uint4*)array_s)[threadIdx.x] = masks;
  __syncthreads();

  uint4 new_index = {histo_s[masks.x],histo_s[masks.y],histo_s[masks.z],histo_s[masks.w]};

  int i = 4*threadIdx.x-1;
  while (i >= 0){
    if (array_s[i] == masks.x){
      new_index.x++;
      i--;
    } else {
      break;
    }
  }

  new_index.y = (masks.y == masks.x) ? new_index.x+1 : new_index.y;
  new_index.z = (masks.z == masks.y) ? new_index.y+1 : new_index.z;
  new_index.w = (masks.w == masks.z) ? new_index.z+1 : new_index.w;

  if (index < numElems){
    ST_WT_INT(&keys_o[new_index.x], mine.x);
    ST_WT_INT(&keys_o[new_index.y], mine.y);
    ST_WT_INT(&keys_o[new_index.z], mine.z);
    ST_WT_INT(&keys_o[new_index.w], mine.w);
    ST_WT_INT(&values_o[new_index.x], value.x);
    ST_WT_INT(&values_o[new_index.y], value.y);
    ST_WT_INT(&values_o[new_index.z], value.z);
    ST_WT_INT(&values_o[new_index.w], value.w);
  }
  MEM_FENCE; __syncthreads();
  SET_NVM_FLAG(2);
}

__global__ void splitRearrange_nvm6 (int numElems, int iter, unsigned int* keys_i, unsigned int* keys_o, unsigned int* values_i, unsigned int* values_o, unsigned int* histo){
  __shared__ unsigned int histo_s[(1<<BITS)];
  __shared__ unsigned int array_s[4*SORT_BS];
  int index = blockIdx.x*4*SORT_BS + 4*threadIdx.x;

  if (threadIdx.x < (1<<BITS)){
    histo_s[threadIdx.x] = histo[gridDim.x*threadIdx.x+blockIdx.x];
  }

  uint4 mine, value;
  if (index < numElems){
    mine = *((uint4*)(keys_i+index));
    value = *((uint4*)(values_i+index));
  } else {
    mine.x = UINT32_MAX;
    mine.y = UINT32_MAX;
    mine.z = UINT32_MAX;
    mine.w = UINT32_MAX;
  }
  uint4 masks = {(mine.x&((1<<(BITS*(iter+1)))-1))>>(BITS*iter),
                 (mine.y&((1<<(BITS*(iter+1)))-1))>>(BITS*iter),
                 (mine.z&((1<<(BITS*(iter+1)))-1))>>(BITS*iter),
                 (mine.w&((1<<(BITS*(iter+1)))-1))>>(BITS*iter)};

  ((uint4*)array_s)[threadIdx.x] = masks;
  __syncthreads();

  uint4 new_index = {histo_s[masks.x],histo_s[masks.y],histo_s[masks.z],histo_s[masks.w]};

  int i = 4*threadIdx.x-1;
  while (i >= 0){
    if (array_s[i] == masks.x){
      new_index.x++;
      i--;
    } else {
      break;
    }
  }

  new_index.y = (masks.y == masks.x) ? new_index.x+1 : new_index.y;
  new_index.z = (masks.z == masks.y) ? new_index.y+1 : new_index.z;
  new_index.w = (masks.w == masks.z) ? new_index.z+1 : new_index.w;

  SET_NVM_FLAG_WB_PC(1);

  if (index < numElems){
    keys_o[new_index.x] = mine.x;
    keys_o[new_index.y] = mine.y;
    keys_o[new_index.z] = mine.z;
    keys_o[new_index.w] = mine.w;
    values_o[new_index.x] = value.x;
    values_o[new_index.y] = value.y;
    values_o[new_index.z] = value.z;
    values_o[new_index.w] = value.w;
    CLWB(&values_o[new_index.w]); 
    CLWB(&keys_o[new_index.x]); 
    CLWB(&keys_o[new_index.y]); 
    CLWB(&keys_o[new_index.z]); 
    CLWB(&keys_o[new_index.w]); 
    CLWB(&values_o[new_index.x]); 
    CLWB(&values_o[new_index.y]); 
    CLWB(&values_o[new_index.z]); 
  }
  MEM_FENCE; PCOMMIT; MEM_FENCE; __syncthreads();
  SET_NVM_FLAG_WB_PC(2);
}

__global__ void splitRearrange_nvm5 (int numElems, int iter, unsigned int* keys_i, unsigned int* keys_o, unsigned int* values_i, unsigned int* values_o, unsigned int* histo){
  __shared__ unsigned int histo_s[(1<<BITS)];
  __shared__ unsigned int array_s[4*SORT_BS];
  int index = blockIdx.x*4*SORT_BS + 4*threadIdx.x;

  if (threadIdx.x < (1<<BITS)){
    histo_s[threadIdx.x] = histo[gridDim.x*threadIdx.x+blockIdx.x];
  }

  uint4 mine, value;
  if (index < numElems){
    mine = *((uint4*)(keys_i+index));
    value = *((uint4*)(values_i+index));
  } else {
    mine.x = UINT32_MAX;
    mine.y = UINT32_MAX;
    mine.z = UINT32_MAX;
    mine.w = UINT32_MAX;
  }
  uint4 masks = {(mine.x&((1<<(BITS*(iter+1)))-1))>>(BITS*iter),
                 (mine.y&((1<<(BITS*(iter+1)))-1))>>(BITS*iter),
                 (mine.z&((1<<(BITS*(iter+1)))-1))>>(BITS*iter),
                 (mine.w&((1<<(BITS*(iter+1)))-1))>>(BITS*iter)};

  ((uint4*)array_s)[threadIdx.x] = masks;
  __syncthreads();

  uint4 new_index = {histo_s[masks.x],histo_s[masks.y],histo_s[masks.z],histo_s[masks.w]};

  int i = 4*threadIdx.x-1;
  while (i >= 0){
    if (array_s[i] == masks.x){
      new_index.x++;
      i--;
    } else {
      break;
    }
  }

  new_index.y = (masks.y == masks.x) ? new_index.x+1 : new_index.y;
  new_index.z = (masks.z == masks.y) ? new_index.y+1 : new_index.z;
  new_index.w = (masks.w == masks.z) ? new_index.z+1 : new_index.w;

  SET_NVM_FLAG_WB(1);

  if (index < numElems){
    keys_o[new_index.x] = mine.x;
    keys_o[new_index.y] = mine.y;
    keys_o[new_index.z] = mine.z;
    keys_o[new_index.w] = mine.w;
    values_o[new_index.x] = value.x;
    values_o[new_index.y] = value.y;
    values_o[new_index.z] = value.z;
    values_o[new_index.w] = value.w;
    CLWB(&values_o[new_index.w]); 
    CLWB(&keys_o[new_index.x]); 
    CLWB(&keys_o[new_index.y]); 
    CLWB(&keys_o[new_index.z]); 
    CLWB(&keys_o[new_index.w]); 
    CLWB(&values_o[new_index.x]); 
    CLWB(&values_o[new_index.y]); 
    CLWB(&values_o[new_index.z]); 
  }
  MEM_FENCE; __syncthreads();
  SET_NVM_FLAG_WB(2);
}

