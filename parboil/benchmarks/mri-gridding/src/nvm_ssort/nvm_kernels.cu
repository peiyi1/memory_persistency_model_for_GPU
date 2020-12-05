
__global__ static void splitSort_nvmo(int numElems, int iter, unsigned int* keys, unsigned int* values, unsigned int* histo)
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

    // Write result.
    if (gid < numElems){
      keys[blockIdx.x*4*SORT_BS+index.x] = lkey.x;
      CLWB(&keys[blockIdx.x*4*SORT_BS+index.x]); SFENCE;
      keys[blockIdx.x*4*SORT_BS+index.y] = lkey.y;
      CLWB(&keys[blockIdx.x*4*SORT_BS+index.y]); SFENCE;
      keys[blockIdx.x*4*SORT_BS+index.z] = lkey.z;
      CLWB(&keys[blockIdx.x*4*SORT_BS+index.z]); SFENCE;
      keys[blockIdx.x*4*SORT_BS+index.w] = lkey.w;
      CLWB(&keys[blockIdx.x*4*SORT_BS+index.w]); SFENCE;

      values[blockIdx.x*4*SORT_BS+index.x] = lvalue.x;
      CLWB(&values[blockIdx.x*4*SORT_BS+index.x]); SFENCE;
      values[blockIdx.x*4*SORT_BS+index.y] = lvalue.y;
      CLWB(&values[blockIdx.x*4*SORT_BS+index.y]); SFENCE;
      values[blockIdx.x*4*SORT_BS+index.z] = lvalue.z;
      CLWB(&values[blockIdx.x*4*SORT_BS+index.z]); SFENCE;
      values[blockIdx.x*4*SORT_BS+index.w] = lvalue.w;
      CLWB(&values[blockIdx.x*4*SORT_BS+index.w]); SFENCE;
    }
    if (tid < (1<<BITS)){
      histo[gridDim.x*threadIdx.x+blockIdx.x] = histo_s[tid];
      CLWB(&histo[gridDim.x*threadIdx.x+blockIdx.x]); SFENCE;
    }
}

__global__ static void splitSort_nvmq(int numElems, int iter, unsigned int* keys, unsigned int* values, unsigned int* histo)
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
    SFENCE;
}


__global__ static void splitSort_nvmu(int numElems, int iter, unsigned int* keys, unsigned int* values, unsigned int* histo)
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

    // Write result.
    if (gid < numElems){
      keys[blockIdx.x*4*SORT_BS+index.x] = lkey.x;
      CLWB(&keys[blockIdx.x*4*SORT_BS+index.x]); SFENCE; PCOMMIT; SFENCE;
      keys[blockIdx.x*4*SORT_BS+index.y] = lkey.y;
      CLWB(&keys[blockIdx.x*4*SORT_BS+index.y]); SFENCE; PCOMMIT; SFENCE;
      keys[blockIdx.x*4*SORT_BS+index.z] = lkey.z;
      CLWB(&keys[blockIdx.x*4*SORT_BS+index.z]); SFENCE; PCOMMIT; SFENCE;
      keys[blockIdx.x*4*SORT_BS+index.w] = lkey.w;
      CLWB(&keys[blockIdx.x*4*SORT_BS+index.w]); SFENCE; PCOMMIT; SFENCE;

      values[blockIdx.x*4*SORT_BS+index.x] = lvalue.x;
      CLWB(&values[blockIdx.x*4*SORT_BS+index.x]); SFENCE; PCOMMIT; SFENCE;
      values[blockIdx.x*4*SORT_BS+index.y] = lvalue.y;
      CLWB(&values[blockIdx.x*4*SORT_BS+index.y]); SFENCE; PCOMMIT; SFENCE;
      values[blockIdx.x*4*SORT_BS+index.z] = lvalue.z;
      CLWB(&values[blockIdx.x*4*SORT_BS+index.z]); SFENCE; PCOMMIT; SFENCE;
      values[blockIdx.x*4*SORT_BS+index.w] = lvalue.w;
      CLWB(&values[blockIdx.x*4*SORT_BS+index.w]); SFENCE; PCOMMIT; SFENCE;
    }
    if (tid < (1<<BITS)){
      histo[gridDim.x*threadIdx.x+blockIdx.x] = histo_s[tid];
      CLWB(&histo[gridDim.x*threadIdx.x+blockIdx.x]); SFENCE; PCOMMIT; SFENCE;
    }
}


__global__ static void splitSort_nvmw(int numElems, int iter, unsigned int* keys, unsigned int* values, unsigned int* histo)
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
    SFENCE; PCOMMIT; SFENCE;
}

__global__ static void splitSort_nvm1(int numElems, int iter, unsigned int* keys, unsigned int* values, unsigned int* histo)
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
    L2WB;
    SFENCE;
}
__global__ static void splitSort_nvm2(int numElems, int iter, unsigned int* keys, unsigned int* values, unsigned int* histo)
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
    L2WB;
    SFENCE; PCOMMIT; SFENCE;
}


__global__ static void splitSort_nvmb(int numElems, int iter, unsigned int* keys, unsigned int* values, unsigned int* histo)
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

    // Write result.
    if (gid < numElems){
      ST_WT_INT(&keys[blockIdx.x*4*SORT_BS+index.x], lkey.x);
      MEM_FENCE;
      ST_WT_INT(&keys[blockIdx.x*4*SORT_BS+index.y], lkey.y);
      MEM_FENCE;
      ST_WT_INT(&keys[blockIdx.x*4*SORT_BS+index.z], lkey.z);
      MEM_FENCE;
      ST_WT_INT(&keys[blockIdx.x*4*SORT_BS+index.w], lkey.w);
      MEM_FENCE;
      ST_WT_INT(&values[blockIdx.x*4*SORT_BS+index.x], lvalue.x);
      MEM_FENCE;
      ST_WT_INT(&values[blockIdx.x*4*SORT_BS+index.y], lvalue.y);
      MEM_FENCE;
      ST_WT_INT(&values[blockIdx.x*4*SORT_BS+index.z], lvalue.z);
      MEM_FENCE;
      ST_WT_INT(&values[blockIdx.x*4*SORT_BS+index.w], lvalue.w);
      MEM_FENCE;
    }
    if (tid < (1<<BITS)){
      ST_WT_INT(&histo[gridDim.x*threadIdx.x+blockIdx.x], histo_s[tid]);
      MEM_FENCE;
    }
}

__global__ static void splitSort_nvmd(int numElems, int iter, unsigned int* keys, unsigned int* values, unsigned int* histo)
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
    }
      MEM_FENCE;
}




__global__ void splitRearrange_nvmb (int numElems, int iter, unsigned int* keys_i, unsigned int* keys_o, unsigned int* values_i, unsigned int* values_o, unsigned int* histo){
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
    MEM_FENCE;
    ST_WT_INT(&keys_o[new_index.y], mine.y);
    MEM_FENCE;
    ST_WT_INT(&keys_o[new_index.z], mine.z);
    MEM_FENCE;
    ST_WT_INT(&keys_o[new_index.w], mine.w);
    MEM_FENCE;
    ST_WT_INT(&values_o[new_index.x], value.x);
    MEM_FENCE;
    ST_WT_INT(&values_o[new_index.y], value.y);
    MEM_FENCE;
    ST_WT_INT(&values_o[new_index.z], value.z);
    MEM_FENCE;
    ST_WT_INT(&values_o[new_index.w], value.w);
    MEM_FENCE;
  }
}

__global__ void splitRearrange_nvmd (int numElems, int iter, unsigned int* keys_i, unsigned int* keys_o, unsigned int* values_i, unsigned int* values_o, unsigned int* histo){
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
  MEM_FENCE;
}




__global__ void splitRearrange_nvmo (int numElems, int iter, unsigned int* keys_i, unsigned int* keys_o, unsigned int* values_i, unsigned int* values_o, unsigned int* histo){
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
    keys_o[new_index.x] = mine.x;
    CLWB(&keys_o[new_index.x]); SFENCE;
    keys_o[new_index.y] = mine.y;
    CLWB(&keys_o[new_index.y]); SFENCE;
    keys_o[new_index.z] = mine.z;
    CLWB(&keys_o[new_index.z]); SFENCE;
    keys_o[new_index.w] = mine.w;
    CLWB(&keys_o[new_index.w]); SFENCE;

    values_o[new_index.x] = value.x;
    CLWB(&values_o[new_index.x]); SFENCE;
    values_o[new_index.y] = value.y;
    CLWB(&values_o[new_index.y]); SFENCE;
    values_o[new_index.z] = value.z;
    CLWB(&values_o[new_index.z]); SFENCE;
    values_o[new_index.w] = value.w;
    CLWB(&values_o[new_index.w]); SFENCE;
  }
}

__global__ void splitRearrange_nvmq (int numElems, int iter, unsigned int* keys_i, unsigned int* keys_o, unsigned int* values_i, unsigned int* values_o, unsigned int* histo){
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
  SFENCE;
}

__global__ void splitRearrange_nvmu (int numElems, int iter, unsigned int* keys_i, unsigned int* keys_o, unsigned int* values_i, unsigned int* values_o, unsigned int* histo){
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
    keys_o[new_index.x] = mine.x;
    CLWB(&keys_o[new_index.x]); SFENCE; PCOMMIT; SFENCE;
    keys_o[new_index.y] = mine.y;
    CLWB(&keys_o[new_index.y]); SFENCE; PCOMMIT; SFENCE;
    keys_o[new_index.z] = mine.z;
    CLWB(&keys_o[new_index.z]); SFENCE; PCOMMIT; SFENCE;
    keys_o[new_index.w] = mine.w;
    CLWB(&keys_o[new_index.w]); SFENCE; PCOMMIT; SFENCE;

    values_o[new_index.x] = value.x;
    CLWB(&values_o[new_index.x]); SFENCE; PCOMMIT; SFENCE;
    values_o[new_index.y] = value.y;
    CLWB(&values_o[new_index.y]); SFENCE; PCOMMIT; SFENCE;
    values_o[new_index.z] = value.z;
    CLWB(&values_o[new_index.z]); SFENCE; PCOMMIT; SFENCE;
    values_o[new_index.w] = value.w;
    CLWB(&values_o[new_index.w]); SFENCE; PCOMMIT; SFENCE;
  }
}

__global__ void splitRearrange_nvmw (int numElems, int iter, unsigned int* keys_i, unsigned int* keys_o, unsigned int* values_i, unsigned int* values_o, unsigned int* histo){
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
  SFENCE; PCOMMIT; SFENCE;
}

__global__ void splitRearrange_nvm1 (int numElems, int iter, unsigned int* keys_i, unsigned int* keys_o, unsigned int* values_i, unsigned int* values_o, unsigned int* histo){
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
    keys_o[new_index.x] = mine.x;
    keys_o[new_index.y] = mine.y;
    keys_o[new_index.z] = mine.z;
    keys_o[new_index.w] = mine.w;
    values_o[new_index.x] = value.x;
    values_o[new_index.y] = value.y;
    values_o[new_index.z] = value.z;
    values_o[new_index.w] = value.w;
    L2WB;
  }
  SFENCE;
}
__global__ void splitRearrange_nvm2 (int numElems, int iter, unsigned int* keys_i, unsigned int* keys_o, unsigned int* values_i, unsigned int* values_o, unsigned int* histo){
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
    keys_o[new_index.x] = mine.x;
    keys_o[new_index.y] = mine.y;
    keys_o[new_index.z] = mine.z;
    keys_o[new_index.w] = mine.w;
    values_o[new_index.x] = value.x;
    values_o[new_index.y] = value.y;
    values_o[new_index.z] = value.z;
    values_o[new_index.w] = value.w;
    L2WB;
  }
  SFENCE; PCOMMIT; SFENCE;
}

