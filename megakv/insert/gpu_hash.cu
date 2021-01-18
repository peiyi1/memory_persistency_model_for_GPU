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

__global__ void hash_insert_cuckoo(
		bucket_t		*hash_table,
		ielem_t			**blk_input,
		int				*blk_elem_num)
{
	ielem_t *in = blk_input[blockIdx.x];
	int total_elem_num = blk_elem_num[blockIdx.x];
	// 16 threads to cooperate for one element
	int step = blockDim.x >> ELEM_NUM_P;
	int idx = threadIdx.x;

	hash_t hash, second_hash;
	loc_t loc, new_loc;
	sign_t sig, new_sig;

	int id;
	int cuckoo_num;
	bucket_t *b;
	int chosen_simd;
	int ballot, ml_mask;

	int simd_lane = idx & ((1 << ELEM_NUM_P) - 1);
	int elem_id = idx >> ELEM_NUM_P;
	int bit_move = idx & (((1 << (5 - ELEM_NUM_P)) - 1) << ELEM_NUM_P);

	for (id = elem_id; id < total_elem_num; id += step) {
		ielem_t *elem = &(in[id]);

		if (elem->sig == 0 && elem->loc == 0) {
			printf("error, all is zero\n");
			continue;
		}

		sig = elem->sig;
		hash = elem->hash;
		loc = elem->loc;

		b = &(hash_table[hash & HASH_MASK]);

		/*=====================================================================
		 * The double __syncthreads() seems useless in else, this is to match the two in
		 * if (chosen_simd == simd_lane). As is stated in the paper <Demystifying GPU 
		 * Microarchitecture through Microbenchmarking>, the __syncthreads() will not go
		 * wrong if not all threads in one wrap reach it. However, the wraps in the same
		 * block need to reach a __syncthreads(), even if they are not on the same line */
		/* Check for same signatures in two bucket */
		ballot = __ballot_sync(FULL_MASK,b->sig[simd_lane] == sig);
		/* first half warp(0~15 threads), bit_move = 0
		 * for second half warp(16~31 threads), bit_move = 16 */
		ballot = (ballot >> bit_move) & ((1 << ELEM_NUM) - 1);
		if (ballot != 0) {
			chosen_simd = (__ffs(ballot) - 1) & ((1 << ELEM_NUM_P) - 1);
			if (simd_lane == chosen_simd) {
				b->loc[simd_lane] = loc;
			}
			continue;
		}

		/*=====================================================================*/
		/* Next we try to insert, the while loop breaks if ballot == 0, and the 
		 * __syncthreads() in the two loops match if the code path divergent between
		 * the warps in a block. Or some will terminate, or process the next element. 
		 * FIXME: if some wrap go to process next element, some stays here, will this
		 * lead to mismatch in __syncthreads()? If it does, we should launch one thread
		 * for each element. God knows what nVidia GPU will behave. FIXME;
		 * Here we write b->loc, and the above code also write b->loc. This will not
		 * lead to conflicts, because here all the signatures are 0, while the aboves
		 * are all non-zero */

		/* Major Location : use last 4 bits of signature */
		ml_mask = (1 << (sig & ((1 << ELEM_NUM_P) - 1))) - 1;
		/* find the empty slot for insertion */
		while (1) {
			ballot = __ballot_sync(FULL_MASK,b->sig[simd_lane] == 0);
			ballot = (ballot >> bit_move) & ((1 << ELEM_NUM) - 1);
			/* 1010|0011 => 0000 0011 1010 0000, 16 bits to 32 bits*/
			ballot = ((ballot & ml_mask) << 16) | ((ballot & ~(ml_mask)));
			if (ballot != 0) {
				chosen_simd = (__ffs(ballot) - 1) & ((1 << ELEM_NUM_P) - 1);
				if (simd_lane == chosen_simd) {
					b->sig[simd_lane] = sig;
				}
			}

			__syncthreads();

			if (ballot != 0) {
				if (b->sig[chosen_simd] == sig) {
					if (simd_lane == chosen_simd) {
						b->loc[simd_lane] = loc;
					}
					goto finish;
				}
			} else {
				break;
			}
		}


		/* ==== try next bucket ==== */
		cuckoo_num = 0;

cuckoo_evict:
		second_hash = (((hash ^ sig) & BLOCK_HASH_MASK) 
				| (hash & ~BLOCK_HASH_MASK)) & HASH_MASK; 
		b = &(hash_table[second_hash]);
		/*=====================================================================*/
		/* Check for same signatures in two bucket */
		ballot = __ballot_sync(FULL_MASK,b->sig[simd_lane] == sig);
		/* first half warp(0~15 threads), bit_move = 0
		 * for second half warp(16~31 threads), bit_move = 16 */
		ballot = (ballot >> bit_move) & ((1 << ELEM_NUM) - 1);
		if (0 != ballot) {
			chosen_simd = (__ffs(ballot) - 1) & ((1 << ELEM_NUM_P) - 1);
			if (simd_lane == chosen_simd) {
				b->loc[simd_lane] = loc;
			}
			continue;
		}

		while (1) {
			ballot = __ballot_sync(FULL_MASK,b->sig[simd_lane] == 0);
			ballot = (ballot >> bit_move) & ((1 << ELEM_NUM) - 1);
			ballot = ((ballot & ml_mask) << 16) | ((ballot & ~(ml_mask)));
			if (ballot != 0) {
				chosen_simd = (__ffs(ballot) - 1) & ((1 << ELEM_NUM_P) - 1);
			} else {
				/* No available slot.
				 * Get a Major location between 0 and 15 for insertion */
				chosen_simd = elem->sig & ((1 << ELEM_NUM_P) - 1);
				if (cuckoo_num < MAX_CUCKOO_NUM) {
					/* record the signature to be evicted */
					new_sig = b->sig[chosen_simd];
					new_loc = b->loc[chosen_simd];
				}
			}
			
			/* synchronize before the signature is written by others */
			__syncthreads();

			if (ballot != 0) {
				if (simd_lane == chosen_simd) {
					b->sig[simd_lane] = sig;
				}
			} else {
				/* two situations to handle: 1) cuckoo_num < MAX_CUCKOO_NUM,
				 * replace one element, and reinsert it into its alternative bucket.
				 * 2) cuckoo_num >= MAX_CUCKOO_NUM.
				 * The cuckoo evict exceed the maximum insert time, replace the element.
				 * In each case, we write the signature first.*/
				if (simd_lane == chosen_simd) {
					b->sig[simd_lane] = sig;
				}
			}

			__syncthreads();

			if (ballot != 0) {
				/* write the empty slot or try again when conflict */
				if (b->sig[chosen_simd] == sig) {
					if (simd_lane == chosen_simd) {
						b->loc[simd_lane] = loc;
					}
					goto finish;
				}
			} else {
				if (cuckoo_num < MAX_CUCKOO_NUM) {
					cuckoo_num ++;
					if (b->sig[chosen_simd] == sig) {
						if (simd_lane == chosen_simd) {
							b->loc[simd_lane] = loc;
						}
						sig = new_sig;
						loc = new_loc;
						goto cuckoo_evict;
					} else {
						/* if there is conflict when writing the signature,
						 * it has been replaced by another one. Reinserting
						 * the element is meaningless, because it will evict
						 * the one that is just inserted. Only one will survive,
						 * we just give up the failed one */
						goto finish;
					}
				} else {
					/* exceed the maximum insert time, evict one */
					if (b->sig[chosen_simd] == sig) {
						if (simd_lane == chosen_simd) {
							b->loc[simd_lane] = loc;
						}
					}
					/* whether or not succesfully inserted, finish */
					goto finish;
				}
			}
		}

finish:
		;
		//now we get to the next element
	}
	return;
}

/* num_blks is the array size of blk_input and blk_output */
extern "C" void gpu_hash_insert(
		bucket_t	*hash_table,
		ielem_t		**blk_input,
		int			*blk_elem_num,
		int			num_blks,
		cudaStream_t stream)
{
	int threads_per_blk = 1024;
	//printf("hash_insert: num_blks %d, threads_per_blk %d\n", num_blks, threads_per_blk);

	// prefer L1 cache rather than shared cache
	//void (*funcPtr)(bucket_t *, ielem_t **, loc_t **, int *);
	//funcPtr = hash_insert;
	//cudaFuncSetCacheConfig(*funcPtr, cudaFuncCachePreferL1);
	assert(ELEM_NUM_P < 5 && ELEM_NUM_P > 0);

syncLapTimer st;
st.lap_start();
	if (stream == 0) {
		if (nvm_opt == 'a')
		hash_insert_cuckoo<<<num_blks, threads_per_blk>>>(
			hash_table, blk_input, blk_elem_num);
		else if (nvm_opt == 'b')
		hash_insert_cuckoo_nvmb<<<num_blks, threads_per_blk>>>(
			hash_table, blk_input, blk_elem_num);
		else if (nvm_opt == 'o')
		hash_insert_cuckoo_nvmo<<<num_blks, threads_per_blk>>>(
			hash_table, blk_input, blk_elem_num);
		else if (nvm_opt == 'u')
		hash_insert_cuckoo_nvmu<<<num_blks, threads_per_blk>>>(
			hash_table, blk_input, blk_elem_num);
		
		else if (nvm_opt == 'k'){
		hash_insert_cuckoo<<<num_blks, threads_per_blk>>>(
                        hash_table, blk_input, blk_elem_num);
		cudaDeviceSynchronize();
                kernel_l2wb <<< num_blks, threads_per_blk>>>();
		}
		else if (nvm_opt == 'l'){
		hash_insert_cuckoo<<<num_blks, threads_per_blk>>>(
                        hash_table, blk_input, blk_elem_num);
                cudaDeviceSynchronize();
		kernel_l2wb_pct <<< num_blks, threads_per_blk>>>();
		}
	}else{
                if (nvm_opt == 'a')
                hash_insert_cuckoo<<<num_blks, threads_per_blk, 0, stream>>>(
                        hash_table, blk_input, blk_elem_num);
                else if (nvm_opt == 'b')
                hash_insert_cuckoo_nvmb<<<num_blks, threads_per_blk, 0, stream>>>(
                        hash_table, blk_input, blk_elem_num);
                else if (nvm_opt == 'o')
                hash_insert_cuckoo_nvmo<<<num_blks, threads_per_blk, 0, stream>>>(
                        hash_table, blk_input, blk_elem_num);
                else if (nvm_opt == 'u')
                hash_insert_cuckoo_nvmu<<<num_blks, threads_per_blk, 0, stream>>>(
                        hash_table, blk_input, blk_elem_num);

		else if (nvm_opt == 'k'){
		hash_insert_cuckoo<<<num_blks, threads_per_blk, 0, stream>>>(
                        hash_table, blk_input, blk_elem_num);
                cudaDeviceSynchronize();
                kernel_l2wb <<< num_blks, threads_per_blk, 0, stream>>>();
                }
                else if (nvm_opt == 'l'){
                hash_insert_cuckoo<<<num_blks, threads_per_blk, 0, stream>>>(
                        hash_table, blk_input, blk_elem_num);
                cudaDeviceSynchronize();
                kernel_l2wb_pct <<< num_blks, threads_per_blk, 0, stream>>>();
                }

	}
st.lap_end();
st.print_total_us("total_us");
st.print_avg_usec("gen_hists", num_blks);
	return;
}

