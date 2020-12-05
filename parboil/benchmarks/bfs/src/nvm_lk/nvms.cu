

struct LocalQueues_nvms : public LocalQueues {
  
  __device__ void reset(int index, dim3 block_dim) {
    LocalQueues::reset(index, block_dim);
    ST_WT_INT(&shadow_sharers[index], sharers[index]);
  }
  __device__ void append(int index, int *overflow, int value) {
    int tail_index = atomicAdd(&tail[index], 1);
    if (tail_index >= W_QUEUE_SIZE)
      //*overflow = 1;
      ST_WT_INT(overflow, 1);
    else {
      elems[index][tail_index] = value;
      ST_WT_INT(&shadow_elems[index][tail_index], value);
    }
  }
  __device__ int size_prefix_sum(int (&prefix_q)[NUM_BIN]) {
    prefix_q[0] = 0;
    //ST_WT_INT(&shadow_prefix_q[0], 0);
    SFENCE;
    for(int i = 1; i < NUM_BIN; i++){
      prefix_q[i] = prefix_q[i-1] + tail[i-1];
      ST_WT_INT(&shadow_prefix_q[i],  prefix_q[i]);
    }
    return prefix_q[NUM_BIN-1] + tail[NUM_BIN-1];
  }
  __device__ void concatenate_smem(int *dst, int (&prefix_q)[NUM_BIN], int *shadow) {
    int q_i = threadIdx.x & MOD_OP; // w-queue index
    int local_shift = threadIdx.x >> EXP; // shift within a w-queue

    while(local_shift < tail[q_i]){
      dst[prefix_q[q_i] + local_shift] = elems[q_i][local_shift];
      ST_WT_INT(shadow + prefix_q[q_i] + local_shift, elems[q_i][local_shift]);
      local_shift += sharers[q_i];
    }
  }
  __device__ void concatenate_gmem(int *dst, int (&prefix_q)[NUM_BIN]) {
    int q_i = threadIdx.x & MOD_OP; // w-queue index
    int local_shift = threadIdx.x >> EXP; // shift within a w-queue

    while(local_shift < tail[q_i]){
      //dst[prefix_q[q_i] + local_shift] = elems[q_i][local_shift];
      ST_WT_INT(&dst[prefix_q[q_i] + local_shift], elems[q_i][local_shift]);
      local_shift += sharers[q_i];
    }
  }

};

__global__ void
BFS_in_GPU_kernel_nvms(int *q1, 
		       int *q2, 
		       Node *g_graph_nodes, 
		       Edge *g_graph_edges, 
		       int *g_color, 
		       int *g_cost, 
		       int no_of_nodes, 
		       int *tail, 
		       int gray_shade, 
		       int k,
		       int *overflow) 
{
  __shared__ LocalQueues_nvms local_q;
  __shared__ int prefix_q[NUM_BIN];
  __shared__ int next_wf[MAX_THREADS_PER_BLOCK];
  __shared__ int  tot_sum;
  if(threadIdx.x == 0)	
    tot_sum = 0;
  int interation=0;
  while(1){
  
	SET_NVM_FLAG(1);
	 __syncthreads();
    if(threadIdx.x < NUM_BIN){
      local_q.reset(threadIdx.x, blockDim);
    }
    __syncthreads();
    int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
    if( tid<no_of_nodes)
      {
	int pid;
	if(tot_sum == 0)//this is the first BFS level of current kernel call
	  pid = q1[tid];  
	else
	  pid = next_wf[tid];//read the current frontier info from last level's propagation
	visit_node(pid, threadIdx.x & MOD_OP, local_q, overflow,
		   g_color, g_cost, gray_shade);
      }
    __syncthreads();
    if(threadIdx.x == 0){
      *tail = tot_sum = local_q.size_prefix_sum(prefix_q);
    }
    __syncthreads();

    if(tot_sum == 0)//the new frontier becomes empty; BFS is over
      return;
    if(tot_sum <= MAX_THREADS_PER_BLOCK){

      local_q.concatenate_smem(next_wf, prefix_q, shadow_next_wf);
      __syncthreads();
      no_of_nodes = tot_sum;
      if(threadIdx.x == 0){
        if(gray_shade == GRAY0)
          gray_shade = GRAY1;
        else
          gray_shade = GRAY0;
      }
    }
    else{
      local_q.concatenate_gmem(q2, prefix_q);
      return;
    }
      SFENCE; __syncthreads();
      SET_NVM_FLAG(0);
      if (tid == 0)
        ST_WT_INT(&NVM_last_iter[blockIdx.x], interation);
    SFENCE;
    interation++;
  }//while
}	
