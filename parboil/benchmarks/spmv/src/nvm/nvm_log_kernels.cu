__device__ float NVM_log[LOG_SIZE_16M];
__device__ float NVM_flag[FLAG_SIZE_1M];

__global__ void spmv_jds_nvmg(float *dst_vector,
			      const float *d_data,const int *d_index, const int *d_perm,
			      const float *x_vec,const int *d_nzcnt,const int dim)
{
  int ix=blockIdx.x*blockDim.x+threadIdx.x;
  int warp_id=ix>>WARP_BITS;

  if(ix<dim)
    {

      ST_WT_FLOAT(&NVM_log[d_perm[ix]], dst_vector[d_perm[ix]]);
      MEM_FENCE; __syncthreads();
      SET_NVM_FLAG(0);


      float sum=0.0f;
      int	bound=sh_zcnt_int[warp_id];
      //prefetch 0
      int j=jds_ptr_int[0]+ix;  
      float d = d_data[j]; 
      int i = d_index[j];  
      float t = x_vec[i];
		
      if (bound>1)  //bound >=2
	{
	  //prefetch 1
	  j=jds_ptr_int[1]+ix;    
	  i =  d_index[j];  
	  int in;
	  float dn;
	  float tn;
	  for(int k=2;k<bound;k++ )
	    {	
	      //prefetch k-1
	      dn = d_data[j]; 
	      //prefetch k
	      j=jds_ptr_int[k]+ix;    
	      in = d_index[j]; 
	      //prefetch k-1
	      tn = x_vec[i];
				
	      //compute k-2
	      sum += d*t; 
	      //sweep to k
	      i = in;  
	      //sweep to k-1
	      d = dn;
	      t =tn; 
	    }	
		
	  //fetch last
	  dn = d_data[j];
	  tn = x_vec[i];
	
	  //compute last-1
	  sum += d*t; 
	  //sweep to last
	  d=dn;
	  t=tn;
	}
      //compute last
      sum += d*t;  // 3 3
		
      //write out data
      //dst_vector[d_perm[ix]]=sum; 
      ST_WT_FLOAT(&dst_vector[d_perm[ix]], sum);
      MEM_FENCE;
    }
  MEM_FENCE; __syncthreads();
  SET_NVM_FLAG(2);
}

__global__ void spmv_jds_nvm3(float *dst_vector,
			      const float *d_data,const int *d_index, const int *d_perm,
			      const float *x_vec,const int *d_nzcnt,const int dim)
{
  int ix=blockIdx.x*blockDim.x+threadIdx.x;
  int warp_id=ix>>WARP_BITS;

  if(ix<dim)
    {

      ST_WB_FLOAT(&NVM_log[d_perm[ix]], dst_vector[d_perm[ix]]);
      MEM_FENCE; __syncthreads();
      SET_NVM_FLAG_WB(0);


      float sum=0.0f;
      int	bound=sh_zcnt_int[warp_id];
      //prefetch 0
      int j=jds_ptr_int[0]+ix;  
      float d = d_data[j]; 
      int i = d_index[j];  
      float t = x_vec[i];
		
      if (bound>1)  //bound >=2
	{
	  //prefetch 1
	  j=jds_ptr_int[1]+ix;    
	  i =  d_index[j];  
	  int in;
	  float dn;
	  float tn;
	  for(int k=2;k<bound;k++ )
	    {	
	      //prefetch k-1
	      dn = d_data[j]; 
	      //prefetch k
	      j=jds_ptr_int[k]+ix;    
	      in = d_index[j]; 
	      //prefetch k-1
	      tn = x_vec[i];
				
	      //compute k-2
	      sum += d*t; 
	      //sweep to k
	      i = in;  
	      //sweep to k-1
	      d = dn;
	      t =tn; 
	    }	
		
	  //fetch last
	  dn = d_data[j];
	  tn = x_vec[i];
	
	  //compute last-1
	  sum += d*t; 
	  //sweep to last
	  d=dn;
	  t=tn;
	}
      //compute last
      sum += d*t;  // 3 3
		
      //write out data
      dst_vector[d_perm[ix]]=sum; 
      CLWB(&dst_vector[d_perm[ix]]);
      MEM_FENCE;
    }
  MEM_FENCE; __syncthreads();
  SET_NVM_FLAG_WB(2);
}

__global__ void spmv_jds_nvm4(float *dst_vector,
			      const float *d_data,const int *d_index, const int *d_perm,
			      const float *x_vec,const int *d_nzcnt,const int dim)
{
  int ix=blockIdx.x*blockDim.x+threadIdx.x;
  int warp_id=ix>>WARP_BITS;

  if(ix<dim)
    {

      ST_WB_FLOAT(&NVM_log[d_perm[ix]], dst_vector[d_perm[ix]]);
      MEM_FENCE; PCOMMIT; MEM_FENCE; __syncthreads();
      SET_NVM_FLAG_WB_PC(0);


      float sum=0.0f;
      int	bound=sh_zcnt_int[warp_id];
      //prefetch 0
      int j=jds_ptr_int[0]+ix;  
      float d = d_data[j]; 
      int i = d_index[j];  
      float t = x_vec[i];
		
      if (bound>1)  //bound >=2
	{
	  //prefetch 1
	  j=jds_ptr_int[1]+ix;    
	  i =  d_index[j];  
	  int in;
	  float dn;
	  float tn;
	  for(int k=2;k<bound;k++ )
	    {	
	      //prefetch k-1
	      dn = d_data[j]; 
	      //prefetch k
	      j=jds_ptr_int[k]+ix;    
	      in = d_index[j]; 
	      //prefetch k-1
	      tn = x_vec[i];
				
	      //compute k-2
	      sum += d*t; 
	      //sweep to k
	      i = in;  
	      //sweep to k-1
	      d = dn;
	      t =tn; 
	    }	
		
	  //fetch last
	  dn = d_data[j];
	  tn = x_vec[i];
	
	  //compute last-1
	  sum += d*t; 
	  //sweep to last
	  d=dn;
	  t=tn;
	}
      //compute last
      sum += d*t;  // 3 3
		
      //write out data
      dst_vector[d_perm[ix]]=sum; 
      CLWB(&dst_vector[d_perm[ix]]);
      MEM_FENCE; PCOMMIT; MEM_FENCE;
    }
  __syncthreads();
  SET_NVM_FLAG_WB_PC(2);
}

__global__ void spmv_jds_nvmi(float *dst_vector,
			      const float *d_data,const int *d_index, const int *d_perm,
			      const float *x_vec,const int *d_nzcnt,const int dim)
{
  int ix=blockIdx.x*blockDim.x+threadIdx.x;
  int warp_id=ix>>WARP_BITS;

  if(ix<dim)
    {

      float sum=0.0f;
      int	bound=sh_zcnt_int[warp_id];
      //prefetch 0
      int j=jds_ptr_int[0]+ix;  
      float d = d_data[j]; 
      int i = d_index[j];  
      float t = x_vec[i];
		
      __syncthreads();
      SET_NVM_FLAG(0);
      if (bound>1)  //bound >=2
	{
	  //prefetch 1
	  j=jds_ptr_int[1]+ix;    
	  i =  d_index[j];  
	  int in;
	  float dn;
	  float tn;
	  for(int k=2;k<bound;k++ )
	    {	
	      //prefetch k-1
	      dn = d_data[j]; 
	      //prefetch k
	      j=jds_ptr_int[k]+ix;    
	      in = d_index[j]; 
	      //prefetch k-1
	      tn = x_vec[i];
				
	      //compute k-2
	      sum += d*t; 
	      //sweep to k
	      i = in;  
	      //sweep to k-1
	      d = dn;
	      t =tn; 
	    }	
		
	  //fetch last
	  dn = d_data[j];
	  tn = x_vec[i];
	
	  //compute last-1
	  sum += d*t; 
	  //sweep to last
	  d=dn;
	  t=tn;
	}
      //compute last
      sum += d*t;  // 3 3
		
      //write out data
      //dst_vector[d_perm[ix]]=sum; 
      ST_WT_FLOAT(&dst_vector[d_perm[ix]], sum);
      MEM_FENCE;
    }
  MEM_FENCE; __syncthreads();
  SET_NVM_FLAG(2);
}

__global__ void spmv_jds_nvm6(float *dst_vector,
			      const float *d_data,const int *d_index, const int *d_perm,
			      const float *x_vec,const int *d_nzcnt,const int dim)
{
  int ix=blockIdx.x*blockDim.x+threadIdx.x;
  int warp_id=ix>>WARP_BITS;

  __syncthreads();
  SET_NVM_FLAG_WB_PC(0);
  if(ix<dim)
    {
      float sum=0.0f;
      int	bound=sh_zcnt_int[warp_id];
      //prefetch 0
      int j=jds_ptr_int[0]+ix;  
      float d = d_data[j]; 
      int i = d_index[j];  
      float t = x_vec[i];
		
      if (bound>1)  //bound >=2
	{
	  //prefetch 1
	  j=jds_ptr_int[1]+ix;    
	  i =  d_index[j];  
	  int in;
	  float dn;
	  float tn;
	  for(int k=2;k<bound;k++ )
	    {	
	      //prefetch k-1
	      dn = d_data[j]; 
	      //prefetch k
	      j=jds_ptr_int[k]+ix;    
	      in = d_index[j]; 
	      //prefetch k-1
	      tn = x_vec[i];
				
	      //compute k-2
	      sum += d*t; 
	      //sweep to k
	      i = in;  
	      //sweep to k-1
	      d = dn;
	      t =tn; 
	    }	
		
	  //fetch last
	  dn = d_data[j];
	  tn = x_vec[i];
	
	  //compute last-1
	  sum += d*t; 
	  //sweep to last
	  d=dn;
	  t=tn;
	}
      //compute last
      sum += d*t;  // 3 3
		
      //write out data
      dst_vector[d_perm[ix]]=sum; 
      CLWB(&dst_vector[d_perm[ix]]);
      MEM_FENCE; PCOMMIT; MEM_FENCE;
    }
  __syncthreads();
  SET_NVM_FLAG_WB_PC(2);
}

__global__ void spmv_jds_nvm5(float *dst_vector,
			      const float *d_data,const int *d_index, const int *d_perm,
			      const float *x_vec,const int *d_nzcnt,const int dim)
{
  int ix=blockIdx.x*blockDim.x+threadIdx.x;
  int warp_id=ix>>WARP_BITS;

  __syncthreads();
  SET_NVM_FLAG_WB(0);
  if(ix<dim)
    {
      float sum=0.0f;
      int	bound=sh_zcnt_int[warp_id];
      //prefetch 0
      int j=jds_ptr_int[0]+ix;  
      float d = d_data[j]; 
      int i = d_index[j];  
      float t = x_vec[i];
		
      if (bound>1)  //bound >=2
	{
	  //prefetch 1
	  j=jds_ptr_int[1]+ix;    
	  i =  d_index[j];  
	  int in;
	  float dn;
	  float tn;
	  for(int k=2;k<bound;k++ )
	    {	
	      //prefetch k-1
	      dn = d_data[j]; 
	      //prefetch k
	      j=jds_ptr_int[k]+ix;    
	      in = d_index[j]; 
	      //prefetch k-1
	      tn = x_vec[i];
				
	      //compute k-2
	      sum += d*t; 
	      //sweep to k
	      i = in;  
	      //sweep to k-1
	      d = dn;
	      t =tn; 
	    }	
		
	  //fetch last
	  dn = d_data[j];
	  tn = x_vec[i];
	
	  //compute last-1
	  sum += d*t; 
	  //sweep to last
	  d=dn;
	  t=tn;
	}
      //compute last
      sum += d*t;  // 3 3
		
      //write out data
      dst_vector[d_perm[ix]]=sum; 
      CLWB(&dst_vector[d_perm[ix]]);
      MEM_FENCE;
    }
  __syncthreads();
  SET_NVM_FLAG_WB(2);
}
