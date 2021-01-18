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

#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include "gpu_hash.h"

#include "nvm_til.h"

#include "nvmb.cu"
#include "nvmo.cu"
#include "nvmu.cu"

__global__ void kernel_l2wb(void){
        L2WB;
        MEM_FENCE;
}
__global__ void kernel_l2wb_pct(void){
        L2WB;
        MEM_FENCE;
         PCOMMIT; MEM_FENCE;
}


__global__ void hash_delete(
		delem_t			*in,
		bucket_t		*hash_table,
		int				total_elem_num,
		int				thread_num)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	int id = 0;
	// 16 threads to cooperate for one element
	int step = thread_num >> ELEM_NUM_P;
	int ballot;
	
	int simd_lane = idx & ((1 << ELEM_NUM_P) - 1);
	int elem_id = idx >> ELEM_NUM_P;
	bucket_t *b;

	int bit_move;
	bit_move = idx & (((1 << (5 - ELEM_NUM_P)) - 1) << ELEM_NUM_P);

	for (id = elem_id; id < total_elem_num; id += step) {
		delem_t *elem = &(in[id]);

		b = &(hash_table[elem->hash & HASH_MASK]);
		/* first perform ballot */
		ballot = __ballot_sync(FULL_MASK,b->sig[simd_lane] == elem->sig && b->loc[simd_lane] == elem->loc);

		if (b->sig[simd_lane] == elem->sig && b->loc[simd_lane] == elem->loc) {
			b->sig[simd_lane] = 0;
		}

		ballot = (ballot >> bit_move) & ((1 << ELEM_NUM) - 1);
		if (ballot != 0) {
			continue;
		}

		//b = &(hash_table[(elem->hash ^ elem->sig) & HASH_MASK]);
		int hash = (((elem->hash ^ elem->sig) & BLOCK_HASH_MASK) 
				| (elem->hash & ~BLOCK_HASH_MASK)) & HASH_MASK; 
		b = &(hash_table[hash]);
		if (b->sig[simd_lane] == elem->sig && b->loc[simd_lane] == elem->loc) {
			b->sig[simd_lane] = 0;
		}
	}
	return;
}
extern "C" void gpu_hash_delete(
		delem_t 	*in,
		bucket_t	*hash_table,
		int			num_elem,
		int 		num_thread,
		int			threads_per_blk,
		cudaStream_t stream)
{
	int num_blks = (num_thread + threads_per_blk - 1) / threads_per_blk;
	assert(num_thread >= threads_per_blk);
	assert(threads_per_blk <= 1024);
	//assert(num_thread <= num_elem);
	if (num_thread % 32 != 0) {
		num_thread = (num_thread + 31) & 0xffe0;
	}
	assert(num_thread % 32 == 0);

	/* prefer L1 cache rather than shared memory,
	   the other is cudaFuncCachePreferShared
	*/
	//void (*funcPtr)(selem_t *, loc_t *, bucket_t *, int, int);
	//funcPtr = hash_search;
	//cudaFuncSetCacheConfig(*funcPtr, cudaFuncCachePreferL1);
	

	//printf("stream=%d, threads_per_blk =%d, num_blks = %d\n", stream, threads_per_blk, num_blks);
syncLapTimer st;
st.lap_start();
	if (stream == 0) {
		if (nvm_opt == 'a')
		hash_delete<<<num_blks, threads_per_blk>>>(
			in, hash_table, num_elem, num_thread);
	        else if (nvm_opt == 'b')
		hash_delete_nvmb<<<num_blks, threads_per_blk>>>(
			in, hash_table, num_elem, num_thread);
		else if (nvm_opt == 'o')
		hash_delete_nvmo<<<num_blks, threads_per_blk>>>(
			in, hash_table, num_elem, num_thread);
		else if (nvm_opt == 'u')
		hash_delete_nvmu<<<num_blks, threads_per_blk>>>(
			in, hash_table, num_elem, num_thread);
		else if (nvm_opt == 'k'){
		hash_delete<<<num_blks, threads_per_blk>>>(
                        in, hash_table, num_elem, num_thread);
		cudaDeviceSynchronize();
                kernel_l2wb <<< num_blks, threads_per_blk>>>();
		}
		else if (nvm_opt == 'l'){
		hash_delete<<<num_blks, threads_per_blk>>>(
                        in, hash_table, num_elem, num_thread);
                cudaDeviceSynchronize();
		kernel_l2wb_pct <<< num_blks, threads_per_blk>>>();
		}
	} else  {
		if (nvm_opt == 'a')
		hash_delete<<<num_blks, threads_per_blk, 0, stream>>>(
			in, hash_table, num_elem, num_thread);
		else if (nvm_opt == 'b')
		hash_delete_nvmb<<<num_blks, threads_per_blk, 0, stream>>>(
			in, hash_table, num_elem, num_thread);
		else if (nvm_opt == 'o')
		hash_delete_nvmo<<<num_blks, threads_per_blk, 0, stream>>>(
			in, hash_table, num_elem, num_thread);
		else if (nvm_opt == 'u')
		hash_delete_nvmu<<<num_blks, threads_per_blk, 0, stream>>>(
			in, hash_table, num_elem, num_thread);
		else if (nvm_opt == 'k'){
		hash_delete<<<num_blks, threads_per_blk, 0, stream>>>(
                        in, hash_table, num_elem, num_thread);
		cudaDeviceSynchronize();
                kernel_l2wb <<< num_blks, threads_per_blk, 0, stream>>>();
		}
		else if (nvm_opt == 'l'){
		hash_delete<<<num_blks, threads_per_blk, 0, stream>>>(
                        in, hash_table, num_elem, num_thread);
                cudaDeviceSynchronize();
		kernel_l2wb_pct <<< num_blks, threads_per_blk, 0, stream>>>();
		}
	}
st.lap_end();
st.print_total_us("total_us");
st.print_avg_usec("gen_hists", num_blks);
	return;
}
