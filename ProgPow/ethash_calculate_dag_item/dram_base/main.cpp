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

	EthashAux::LightType EthashAux_light;
	EthashAux_light = EthashAux::light(epoch);
	bytesConstRef lightData = EthashAux_light->data();

	ethash_light_t _light = EthashAux_light->light;
	uint8_t const* _lightData =lightData.data();
	uint64_t _lightBytes = lightData.size();

	uint64_t dagBytes = ethash_get_datasize(_light->block_number);
	uint32_t lightWords = (unsigned)(_lightBytes / sizeof(node));

	uint32_t m_device_num=0;	
	CUDA_SAFE_CALL(cudaSetDevice(m_device_num));
        cout<< "Set Device to current"<<endl;
	
	cout << "Resetting device"<<endl;
        CUDA_SAFE_CALL(cudaDeviceReset());

	hash64_t* m_dag = nullptr;
        m_dag = nullptr;
	
	hash64_t * dag = m_dag;
        hash64_t * light = nullptr;

        if(!light){
                cout << "Allocating light with size: " << _lightBytes <<endl;
                CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&light), _lightBytes));
        }
	CUDA_SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(light), _lightData, _lightBytes, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dag), dagBytes));
	
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

	cout<<"Generating DAG for GPU #"<< m_device_num <<" with dagBytes: " << dagBytes <<" gridSize: " << s_gridSize <<endl;
	ethash_generate_dag(dag, dagBytes, light, lightWords, s_gridSize, s_blockSize, m_streams[0], m_device_num);
	cout<<"Finished DAG"<<endl;
	m_dag = dag;
	return 0;
}
