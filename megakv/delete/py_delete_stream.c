/*
 * Copyright (c) 2015 Kai Zhang (kay21s@gmail.com)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <cuda_runtime.h>

#include "gpu_hash.h"
#include "libgpuhash.h"

//#define SELEM_NUM 3000
#define LOOP_TIME 1

#define CPU_FREQUENCY_US 2600 // 2GHz/1e6

char nvm_opt;
static inline uint64_t read_tsc(void)
{
	union {
		uint64_t tsc_64;
		struct {
			uint32_t lo_32;
			uint32_t hi_32;
		};
	} tsc;
	//_mm_mfence();
	asm volatile("rdtsc" :
				"=a" (tsc.lo_32),
				"=d" (tsc.hi_32));
	return tsc.tsc_64;
}

int main(int argc, char *argv[])
{
	nvm_opt = *argv[--argc];
	int SELEM_NUM, THREAD_NUM, THREADS_PER_BLK = 1024;
	int STREAM_NUM = 1; 
	SELEM_NUM = 65536;
	THREAD_NUM = 1024*8;
	SELEM_NUM /= STREAM_NUM;

	uint8_t *device_hash_table;
	uint8_t *host_hash_table;
	uint8_t *device_in[STREAM_NUM];
	uint8_t *host_in[STREAM_NUM];
	double diff;
	int i, j, loop;

	uint64_t start, end;

	cudaStream_t stream[STREAM_NUM];
	for (i = 0; i < STREAM_NUM; i ++) {
		cudaStreamCreate(&stream[i]);
	}

	cudaMalloc((void **)&(device_hash_table), HT_SIZE);
	host_hash_table = malloc(HT_SIZE);
	memset(host_hash_table, 0, HT_SIZE);

	for (i = 0; i < STREAM_NUM; i ++) {
		cudaHostAlloc((void **)&(host_in[i]), SELEM_NUM * sizeof(delem_t), cudaHostAllocDefault);
		cudaMalloc((void **)&(device_in[i]), SELEM_NUM * sizeof(delem_t));
	}

	srand(10);
	int temp, temp1;
	for (i = 0; i < HT_SIZE/sizeof(uint32_t); i ++) {
		/* fill the signature of each bucket to 1~16 */
		temp = i >> ELEM_NUM_P;
		temp1 = temp&0x1;
		if (temp1 == 0) {
			((int *)host_hash_table)[i] = i % ELEM_NUM + 1;
		}
	}

	cudaMemcpy(device_hash_table, host_hash_table, HT_SIZE, cudaMemcpyHostToDevice);

	// warm up
	//for (loop = 0; loop < LOOP_NUM; loop ++) {
	for (i = 0; i < STREAM_NUM; i ++) {
		for (j = 0; j < SELEM_NUM; j ++) {
			(((delem_t *)host_in[i])[j]).hash = rand() % BUC_NUM;
			(((delem_t *)host_in[i])[j]).sig = rand() % ELEM_NUM + 1;
			(((delem_t *)host_in[i])[j]).loc = 0;
		}
	}
	for (i = 0; i < STREAM_NUM; i ++) {
		cudaMemcpyAsync(device_in[i], host_in[i], SELEM_NUM * sizeof(delem_t), cudaMemcpyHostToDevice, stream[i]);
		gpu_hash_delete((delem_t *)device_in[i], 
			(bucket_t *)device_hash_table, SELEM_NUM, THREAD_NUM, THREADS_PER_BLK, stream[i]);
	}
	//}

	cudaDeviceSynchronize();


	return 0;
}
