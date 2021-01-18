//strict_using_clwb
__global__ void hash_delete_nvmu(
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
			CLWB(&(b->sig[simd_lane]));
			MEM_FENCE;
			PCOMMIT;MEM_FENCE;
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
			CLWB(&(b->sig[simd_lane]));
			MEM_FENCE;
			PCOMMIT;MEM_FENCE;
		}
	}
	return;
}
