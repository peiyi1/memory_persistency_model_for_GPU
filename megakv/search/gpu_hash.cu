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

__global__ void hash_search(
		selem_t			*in,
		loc_t			*out,
		bucket_t		*hash_table,
		int				total_elem_num,
		int				thread_num)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	int id = 0;
	// (1 << ELEM_NUM_P) threads to cooperate for one element
	int step = thread_num >> ELEM_NUM_P;
	int ballot;
	
	int simd_lane = idx & ((1 << ELEM_NUM_P) - 1);
	int elem_id = idx >> ELEM_NUM_P;

	int bit_move;
	bit_move = idx & (((1 << (5 - ELEM_NUM_P)) - 1) << ELEM_NUM_P);

	for (id = elem_id; id < total_elem_num; id += step) {
		selem_t *elem = &(in[id]);

		bucket_t *b = &(hash_table[elem->hash & HASH_MASK]);
		if (b->sig[simd_lane] == elem->sig) {
			out[id << 1] = b->loc[simd_lane];
		}
		ballot = __ballot_sync(FULL_MASK,b->sig[simd_lane] == elem->sig);
		ballot = (ballot >> bit_move) & ((1 << ELEM_NUM) - 1);
		
		int hash = (((elem->hash ^ elem->sig) & BLOCK_HASH_MASK) 
				| (elem->hash & ~BLOCK_HASH_MASK)) & HASH_MASK; 
		b = &(hash_table[hash]);
		if (b->sig[simd_lane] == elem->sig) {
			out[(id << 1) + 1] = b->loc[simd_lane];
		}
	}
	return;
}


extern "C" void gpu_hash_search(
		selem_t 	*in,
		loc_t		*out,
		bucket_t	*hash_table,
		int			num_elem,
		int 		num_thread,
		int			threads_per_blk,
		cudaStream_t stream)
{
	int num_blks = (num_thread + threads_per_blk - 1) / threads_per_blk;
	assert(num_thread > threads_per_blk);
	assert(threads_per_blk <= 1024);
	//assert(num_thread <= num_elem);
	if (num_thread % 32 != 0) {
		num_thread = (num_thread + 31) & 0xffe0;
	}
	assert(num_thread % 32 == 0);
syncLapTimer st;
st.lap_start();
	if (stream == 0) {
		if (nvm_opt == 'a')
		hash_search<<<num_blks, threads_per_blk>>>(
			in, out, hash_table, num_elem, num_thread);
		else if (nvm_opt == 'b')
		hash_search_nvmb<<<num_blks, threads_per_blk>>>(
			in, out, hash_table, num_elem, num_thread);
		else if (nvm_opt == 'o')
		hash_search_nvmo<<<num_blks, threads_per_blk>>>(
			in, out, hash_table, num_elem, num_thread);
		else if (nvm_opt == 'u')
		hash_search_nvmu<<<num_blks, threads_per_blk>>>(
			in, out, hash_table, num_elem, num_thread);
		else if (nvm_opt == 'k'){
		hash_search<<<num_blks, threads_per_blk>>>(
                        in, out, hash_table, num_elem, num_thread);
		cudaDeviceSynchronize();
		kernel_l2wb <<< num_blks, threads_per_blk>>>();
		}
		else if (nvm_opt == 'l'){
                hash_search<<<num_blks, threads_per_blk>>>(
                        in, out, hash_table, num_elem, num_thread);
                cudaDeviceSynchronize();
                kernel_l2wb_pct <<< num_blks, threads_per_blk>>>();
                }
	}else{
                if (nvm_opt == 'a')
                hash_search<<<num_blks, threads_per_blk, 0, stream>>>(
                        in, out, hash_table, num_elem, num_thread);
                else if (nvm_opt == 'b')
                hash_search_nvmb<<<num_blks, threads_per_blk, 0, stream>>>(
                        in, out, hash_table, num_elem, num_thread);
                else if (nvm_opt == 'o')
                hash_search_nvmo<<<num_blks, threads_per_blk, 0, stream>>>(
                        in, out, hash_table, num_elem, num_thread);
                else if (nvm_opt == 'u')
                hash_search_nvmu<<<num_blks, threads_per_blk, 0, stream>>>(
                        in, out, hash_table, num_elem, num_thread);
                else if (nvm_opt == 'k'){
                hash_search<<<num_blks, threads_per_blk, 0, stream>>>(
                        in, out, hash_table, num_elem, num_thread);
                cudaDeviceSynchronize();
                kernel_l2wb <<< num_blks, threads_per_blk, 0, stream>>>();
                }
                else if (nvm_opt == 'l'){
                hash_search<<<num_blks, threads_per_blk, 0, stream>>>(
                        in, out, hash_table, num_elem, num_thread);
                cudaDeviceSynchronize();
                kernel_l2wb_pct <<< num_blks, threads_per_blk, 0, stream>>>();
                }
	}
st.lap_end();
st.print_total_us("total_us");
st.print_avg_usec("gen_hists", num_blks);
	return;
}


