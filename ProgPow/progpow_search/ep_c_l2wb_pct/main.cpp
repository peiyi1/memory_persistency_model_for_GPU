#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <cuda_runtime.h>


#include <iostream>
#include <fstream>

#include "libethcore/EthashAux.h"
#include "libethash/ethash.h"
#include "libethash-cuda/CUDAMiner_cuda.h"
#include "libethash/internal.h"
using namespace std;
using namespace dev;
using namespace eth;

static uint32_t bswap(uint32_t a)
{ 
  a = (a << 16) | (a >> 16);
  return ((a & 0x00ff00ff) << 8) | ((a >> 8) & 0x00ff00ff);
}

static void unhex(hash32_t *dst, const char *src)
{ 
  const char *p = src;
  uint32_t *q = dst->uint32s;
  uint32_t v = 0;
  
  while (*p && q <= &dst->uint32s[7]) {
      if (*p >= '0' && *p <= '9')
          v |= *p - '0'; 
      else if (*p >= 'a' && *p <= 'f')
          v |= *p - ('a' - 10);
      else
          break; 
      if (!((++p - src) & 7))
          *q++ = bswap(v);
      v <<= 4;
  }
}
static int ethash_full_new_callback(unsigned int percent)
{
        fprintf(stderr, "Full DAG init %u%%\r", percent);
        return 0;
}
int main(void ){
	uint32_t s_gridSize=4;
        uint32_t s_blockSize=256;
	int epoch=1;
	uint32_t m_device_num=0;	
	CUDA_SAFE_CALL(cudaSetDevice(m_device_num));
        cout<< "Set Device to current"<<endl;
	
	cout << "Resetting device"<<endl;
        CUDA_SAFE_CALL(cudaDeviceReset());
	static unsigned s_numStreams=2;
	cudaStream_t  * m_streams;
	search_results** m_search_buf;
        m_search_buf = new search_results *[s_numStreams];
	m_streams = new cudaStream_t[s_numStreams];

	cout<<"Generating mining buffers"<<endl;
	for (unsigned i = 0; i != s_numStreams; ++i)
        {
		CUDA_SAFE_CALL(cudaMallocHost(&m_search_buf[i], sizeof(search_results)));
                CUDA_SAFE_CALL(cudaStreamCreate(&m_streams[i]));
        }	
	unsigned int block_number = 30000;
	ethash_light_t light_h = ethash_light_new(block_number);
        if (!light_h) {
                fprintf(stderr, "ethash_light_new() failed\n");
                return 1;
        }
        fprintf(stderr, "Light DAG init done\n");
  
	ethash_full_t full = ethash_full_new(light_h, ethash_full_new_callback);
        if (!full) {
                fprintf(stderr, "ethash_full_new() failed\n");
                return 1;
        }
        fprintf(stderr, "Full DAG init done\n");

	dag_t *g_dag;
	CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&g_dag), ethash_full_dag_size(full)));
	CUDA_SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(g_dag), ethash_full_dag(full), ethash_full_dag_size(full), cudaMemcpyHostToDevice));	
	
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        for (unsigned int i = 0; i < s_numStreams; i++)
		m_search_buf[i]->count = 0;
	
	hash32_t m_current_header;
        uint64_t m_current_target;
        uint64_t m_current_nonce;
        uint64_t m_starting_nonce;
        uint64_t m_current_index;

	memset(&m_current_header, 0, sizeof(hash32_t));
        m_current_target = 0;
        m_current_nonce = 0;
	m_current_index = 0;
	const uint32_t batch_size = s_gridSize * s_blockSize;
		
	m_starting_nonce = 0x123456789abcdef0ULL;
	m_current_nonce = m_starting_nonce - batch_size;
	unhex(&m_current_header, "ffeeddccbbaa9988776655443322110000112233445566778899aabbccddeeff");
	m_current_target = -1;
	bool s_noeval = true;
	search_results* g_buffer;
	search_results* buffer;
	for(int i=0;i<3;i++){
                m_current_index++;
                m_current_nonce += batch_size;
                auto stream_index = m_current_index % s_numStreams;
                cudaStream_t stream = m_streams[stream_index];
		buffer = m_search_buf[stream_index];
                uint32_t found_count = 0;
                uint64_t nonces[SEARCH_RESULTS];
                h256 mixes[SEARCH_RESULTS];
                uint64_t nonce_base = m_current_nonce - s_numStreams * batch_size;	
                if (m_current_index >= s_numStreams)
                {
                        CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
                        found_count = buffer->count;
                        if (found_count) {
                                buffer->count = 0;
                                if (found_count > SEARCH_RESULTS)
                                        found_count = SEARCH_RESULTS;
                                for (unsigned int j = 0; j < found_count; j++) {
                                        nonces[j] = nonce_base + buffer->result[j].gid;
                                        if (s_noeval){
                                                memcpy(mixes[j].data(), (void *)&buffer->result[j].mix, sizeof(buffer->result[j].mix));
        					if(nonces[j] == m_starting_nonce)
                                                        cout << "Digest = " << mixes[j] << "\n";       				
					}
		                 }
                        }
                }
	
		bool hack_false = false;
		CUDA_SAFE_CALL(cudaMalloc((&g_buffer),sizeof(search_results) ));
		CUDA_SAFE_CALL(cudaMemcpy(g_buffer,buffer , sizeof(search_results), cudaMemcpyHostToDevice)); 
		search_kernel(m_current_nonce,m_current_header,m_current_target,g_dag, g_buffer, hack_false,s_gridSize, s_blockSize,stream);		
		CUDA_SAFE_CALL(cudaMemcpy(buffer,g_buffer , sizeof(search_results), cudaMemcpyDeviceToHost));
		cudaFree(g_buffer);
	}
	cudaFree(g_dag);
	return 0;
}
