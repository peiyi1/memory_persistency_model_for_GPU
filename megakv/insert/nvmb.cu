__global__ void hash_insert_cuckoo_nvmb(
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
				//b->loc[simd_lane] = loc;
				asm("st.global.wt.u32 [%0], %1;" :: "l"(&(b->loc[simd_lane])), "r"(loc) : "memory");
				asm("membar.gl;");
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
					//b->sig[simd_lane] = sig;
					asm("st.global.wt.u32 [%0], %1;" :: "l"(&(b->sig[simd_lane])), "r"(sig) : "memory");
					asm("membar.gl;");
				}
			}

			__syncthreads();

			if (ballot != 0) {
				if (b->sig[chosen_simd] == sig) {
					if (simd_lane == chosen_simd) {
						//b->loc[simd_lane] = loc;
						asm("st.global.wt.u32 [%0], %1;" :: "l"(&(b->loc[simd_lane])), "r"(loc) : "memory");
						asm("membar.gl;");
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
				//b->loc[simd_lane] = loc;
				asm("st.global.wt.u32 [%0], %1;" :: "l"(&(b->loc[simd_lane])), "r"(loc) : "memory");
				asm("membar.gl;");
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
					//b->sig[simd_lane] = sig;
					asm("st.global.wt.u32 [%0], %1;" :: "l"(&(b->sig[simd_lane])), "r"(sig) : "memory");
					asm("membar.gl;");
				}
			} else {
				/* two situations to handle: 1) cuckoo_num < MAX_CUCKOO_NUM,
				 * replace one element, and reinsert it into its alternative bucket.
				 * 2) cuckoo_num >= MAX_CUCKOO_NUM.
				 * The cuckoo evict exceed the maximum insert time, replace the element.
				 * In each case, we write the signature first.*/
				if (simd_lane == chosen_simd) {
					//b->sig[simd_lane] = sig;
					asm("st.global.wt.u32 [%0], %1;" :: "l"(&(b->sig[simd_lane])), "r"(sig) : "memory");
					asm("membar.gl;");
				}
			}

			__syncthreads();

			if (ballot != 0) {
				/* write the empty slot or try again when conflict */
				if (b->sig[chosen_simd] == sig) {
					if (simd_lane == chosen_simd) {
						//b->loc[simd_lane] = loc;
						asm("st.global.wt.u32 [%0], %1;" :: "l"(&(b->loc[simd_lane])), "r"(loc) : "memory");
						asm("membar.gl;");
					}
					goto finish;
				}
			} else {
				if (cuckoo_num < MAX_CUCKOO_NUM) {
					cuckoo_num ++;
					if (b->sig[chosen_simd] == sig) {
						if (simd_lane == chosen_simd) {
							//b->loc[simd_lane] = loc;
							asm("st.global.wt.u32 [%0], %1;" :: "l"(&(b->loc[simd_lane])), "r"(loc) : "memory");
							asm("membar.gl;");
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
							//b->loc[simd_lane] = loc;
							asm("st.global.wt.u32 [%0], %1;" :: "l"(&(b->loc[simd_lane])), "r"(loc) : "memory");
							asm("membar.gl;");
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
