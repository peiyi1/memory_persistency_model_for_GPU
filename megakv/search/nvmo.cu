//strict_using_clwb
__global__ void hash_search_nvmo(
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
			CLWB(&out[id << 1]);
			MEM_FENCE;
		}
		ballot = __ballot_sync(FULL_MASK,b->sig[simd_lane] == elem->sig);
		ballot = (ballot >> bit_move) & ((1 << ELEM_NUM) - 1);
		
		int hash = (((elem->hash ^ elem->sig) & BLOCK_HASH_MASK) 
				| (elem->hash & ~BLOCK_HASH_MASK)) & HASH_MASK; 
		b = &(hash_table[hash]);
		if (b->sig[simd_lane] == elem->sig) {
			out[(id << 1) + 1] = b->loc[simd_lane];
			CLWB(&out[(id << 1) + 1]);
			MEM_FENCE;
		}
	}
	return;
}
