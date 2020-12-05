__global__ void mb_sad_calc_nvmo(unsigned short *blk_sad,
                            unsigned short *frame,
                            int mb_width,
                            int mb_height)
{
  int tx = (threadIdx.x / CEIL(MAX_POS, POS_PER_THREAD)) % THREADS_W;
  int ty = (threadIdx.x / CEIL(MAX_POS, POS_PER_THREAD)) / THREADS_W;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int img_width = mb_width*16;

  /* Macroblock and sub-block coordinates */
  int mb_x = (tx + bx * THREADS_W) >> 2;
  int mb_y = (ty + by * THREADS_H) >> 2;
  int block_x = (tx + bx * THREADS_W) & 0x03;
  int block_y = (ty + by * THREADS_H) & 0x03;

  /* If this thread is assigned to an invalid 4x4 block, do nothing */
  if ((mb_x < mb_width) && (mb_y < mb_height))
    {
      /* Pixel offset of the origin of the current 4x4 block */
      int frame_x = ((mb_x << 2) + block_x) << 2;
      int frame_y = ((mb_y << 2) + block_y) << 2;

      /* Origin of the search area for this 4x4 block */
      int ref_x = frame_x - SEARCH_RANGE;
      int ref_y = frame_y - SEARCH_RANGE;

      /* Origin in the current frame for this 4x4 block */
      int cur_o = frame_y * img_width + frame_x;

      int search_pos;
      int search_pos_base =
        (threadIdx.x % CEIL(MAX_POS, POS_PER_THREAD)) * POS_PER_THREAD;
      int search_pos_end = search_pos_base + POS_PER_THREAD;

      /* All SADs from this thread are stored in a contiguous chunk
       * of memory starting at this offset */
      blk_sad += mb_width * mb_height * MAX_POS_PADDED * (9 + 16) +
        (mb_y * mb_width + mb_x) * MAX_POS_PADDED * 16 +
        (4 * block_y + block_x) * MAX_POS_PADDED;

      /* Don't go past bounds */
      if (search_pos_end > MAX_POS)
        search_pos_end = MAX_POS;

      /* For each search position, within the range allocated to this thread */
      for (search_pos = search_pos_base;
           search_pos < search_pos_end;
           search_pos++) {
        unsigned short sad4x4 = 0;
        int search_off_x = ref_x + (search_pos % SEARCH_DIMENSION);
        int search_off_y = ref_y + (search_pos / SEARCH_DIMENSION);

        /* 4x4 SAD computation */
        for(int y=0; y<4; y++) {
          for (int x=0; x<4; x++) {
            sad4x4 +=
              abs(tex2D(ref, search_off_x + x, search_off_y + y) -
                  frame[cur_o + y * img_width + x]);
          }
        }

        /* Save this value into the local SAD array */
        blk_sad[search_pos] = sad4x4;
	CLWB16(&blk_sad[search_pos]); SFENCE;
      }
    }
}

__global__ void mb_sad_calc_nvmu(unsigned short *blk_sad,
                            unsigned short *frame,
                            int mb_width,
                            int mb_height)
{
  int tx = (threadIdx.x / CEIL(MAX_POS, POS_PER_THREAD)) % THREADS_W;
  int ty = (threadIdx.x / CEIL(MAX_POS, POS_PER_THREAD)) / THREADS_W;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int img_width = mb_width*16;

  /* Macroblock and sub-block coordinates */
  int mb_x = (tx + bx * THREADS_W) >> 2;
  int mb_y = (ty + by * THREADS_H) >> 2;
  int block_x = (tx + bx * THREADS_W) & 0x03;
  int block_y = (ty + by * THREADS_H) & 0x03;

  /* If this thread is assigned to an invalid 4x4 block, do nothing */
  if ((mb_x < mb_width) && (mb_y < mb_height))
    {
      /* Pixel offset of the origin of the current 4x4 block */
      int frame_x = ((mb_x << 2) + block_x) << 2;
      int frame_y = ((mb_y << 2) + block_y) << 2;

      /* Origin of the search area for this 4x4 block */
      int ref_x = frame_x - SEARCH_RANGE;
      int ref_y = frame_y - SEARCH_RANGE;

      /* Origin in the current frame for this 4x4 block */
      int cur_o = frame_y * img_width + frame_x;

      int search_pos;
      int search_pos_base =
        (threadIdx.x % CEIL(MAX_POS, POS_PER_THREAD)) * POS_PER_THREAD;
      int search_pos_end = search_pos_base + POS_PER_THREAD;

      /* All SADs from this thread are stored in a contiguous chunk
       * of memory starting at this offset */
      blk_sad += mb_width * mb_height * MAX_POS_PADDED * (9 + 16) +
        (mb_y * mb_width + mb_x) * MAX_POS_PADDED * 16 +
        (4 * block_y + block_x) * MAX_POS_PADDED;

      /* Don't go past bounds */
      if (search_pos_end > MAX_POS)
        search_pos_end = MAX_POS;

      /* For each search position, within the range allocated to this thread */
      for (search_pos = search_pos_base;
           search_pos < search_pos_end;
           search_pos++) {
        unsigned short sad4x4 = 0;
        int search_off_x = ref_x + (search_pos % SEARCH_DIMENSION);
        int search_off_y = ref_y + (search_pos / SEARCH_DIMENSION);

        /* 4x4 SAD computation */
        for(int y=0; y<4; y++) {
          for (int x=0; x<4; x++) {
            sad4x4 +=
              abs(tex2D(ref, search_off_x + x, search_off_y + y) -
                  frame[cur_o + y * img_width + x]);
          }
        }

        /* Save this value into the local SAD array */
        blk_sad[search_pos] = sad4x4;
	CLWB16(&blk_sad[search_pos]); SFENCE; PCOMMIT; SFENCE;
      }
    }
}


__global__ void mb_sad_calc_nvmq(unsigned short *blk_sad,
                            unsigned short *frame,
                            int mb_width,
                            int mb_height)
{
  int tx = (threadIdx.x / CEIL(MAX_POS, POS_PER_THREAD)) % THREADS_W;
  int ty = (threadIdx.x / CEIL(MAX_POS, POS_PER_THREAD)) / THREADS_W;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int img_width = mb_width*16;

  /* Macroblock and sub-block coordinates */
  int mb_x = (tx + bx * THREADS_W) >> 2;
  int mb_y = (ty + by * THREADS_H) >> 2;
  int block_x = (tx + bx * THREADS_W) & 0x03;
  int block_y = (ty + by * THREADS_H) & 0x03;

  /* If this thread is assigned to an invalid 4x4 block, do nothing */
  if ((mb_x < mb_width) && (mb_y < mb_height))
    {
      /* Pixel offset of the origin of the current 4x4 block */
      int frame_x = ((mb_x << 2) + block_x) << 2;
      int frame_y = ((mb_y << 2) + block_y) << 2;

      /* Origin of the search area for this 4x4 block */
      int ref_x = frame_x - SEARCH_RANGE;
      int ref_y = frame_y - SEARCH_RANGE;

      /* Origin in the current frame for this 4x4 block */
      int cur_o = frame_y * img_width + frame_x;

      int search_pos;
      int search_pos_base =
        (threadIdx.x % CEIL(MAX_POS, POS_PER_THREAD)) * POS_PER_THREAD;
      int search_pos_end = search_pos_base + POS_PER_THREAD;

      /* All SADs from this thread are stored in a contiguous chunk
       * of memory starting at this offset */
      blk_sad += mb_width * mb_height * MAX_POS_PADDED * (9 + 16) +
        (mb_y * mb_width + mb_x) * MAX_POS_PADDED * 16 +
        (4 * block_y + block_x) * MAX_POS_PADDED;

      /* Don't go past bounds */
      if (search_pos_end > MAX_POS)
        search_pos_end = MAX_POS;

      /* For each search position, within the range allocated to this thread */
      for (search_pos = search_pos_base;
           search_pos < search_pos_end;
           search_pos++) {
        unsigned short sad4x4 = 0;
        int search_off_x = ref_x + (search_pos % SEARCH_DIMENSION);
        int search_off_y = ref_y + (search_pos / SEARCH_DIMENSION);

        /* 4x4 SAD computation */
        for(int y=0; y<4; y++) {
          for (int x=0; x<4; x++) {
            sad4x4 +=
              abs(tex2D(ref, search_off_x + x, search_off_y + y) -
                  frame[cur_o + y * img_width + x]);
          }
        }

        /* Save this value into the local SAD array */
        blk_sad[search_pos] = sad4x4;
      }
      for (search_pos = search_pos_base;
           search_pos < search_pos_end;
           search_pos++) {
	CLWB16(&blk_sad[search_pos]);
      }
      SFENCE;
    }
}



__global__ void mb_sad_calc_nvmw(unsigned short *blk_sad,
                            unsigned short *frame,
                            int mb_width,
                            int mb_height)
{
  int tx = (threadIdx.x / CEIL(MAX_POS, POS_PER_THREAD)) % THREADS_W;
  int ty = (threadIdx.x / CEIL(MAX_POS, POS_PER_THREAD)) / THREADS_W;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int img_width = mb_width*16;

  /* Macroblock and sub-block coordinates */
  int mb_x = (tx + bx * THREADS_W) >> 2;
  int mb_y = (ty + by * THREADS_H) >> 2;
  int block_x = (tx + bx * THREADS_W) & 0x03;
  int block_y = (ty + by * THREADS_H) & 0x03;

  /* If this thread is assigned to an invalid 4x4 block, do nothing */
  if ((mb_x < mb_width) && (mb_y < mb_height))
    {
      /* Pixel offset of the origin of the current 4x4 block */
      int frame_x = ((mb_x << 2) + block_x) << 2;
      int frame_y = ((mb_y << 2) + block_y) << 2;

      /* Origin of the search area for this 4x4 block */
      int ref_x = frame_x - SEARCH_RANGE;
      int ref_y = frame_y - SEARCH_RANGE;

      /* Origin in the current frame for this 4x4 block */
      int cur_o = frame_y * img_width + frame_x;

      int search_pos;
      int search_pos_base =
        (threadIdx.x % CEIL(MAX_POS, POS_PER_THREAD)) * POS_PER_THREAD;
      int search_pos_end = search_pos_base + POS_PER_THREAD;

      /* All SADs from this thread are stored in a contiguous chunk
       * of memory starting at this offset */
      blk_sad += mb_width * mb_height * MAX_POS_PADDED * (9 + 16) +
        (mb_y * mb_width + mb_x) * MAX_POS_PADDED * 16 +
        (4 * block_y + block_x) * MAX_POS_PADDED;

      /* Don't go past bounds */
      if (search_pos_end > MAX_POS)
        search_pos_end = MAX_POS;

      /* For each search position, within the range allocated to this thread */
      for (search_pos = search_pos_base;
           search_pos < search_pos_end;
           search_pos++) {
        unsigned short sad4x4 = 0;
        int search_off_x = ref_x + (search_pos % SEARCH_DIMENSION);
        int search_off_y = ref_y + (search_pos / SEARCH_DIMENSION);

        /* 4x4 SAD computation */
        for(int y=0; y<4; y++) {
          for (int x=0; x<4; x++) {
            sad4x4 +=
              abs(tex2D(ref, search_off_x + x, search_off_y + y) -
                  frame[cur_o + y * img_width + x]);
          }
        }

        /* Save this value into the local SAD array */
        blk_sad[search_pos] = sad4x4;
      }
      for (search_pos = search_pos_base;
           search_pos < search_pos_end;
           search_pos++) {
	CLWB16(&blk_sad[search_pos]);
      }
      SFENCE; PCOMMIT; SFENCE;
    }
}

__global__ void mb_sad_calc_nvm1(unsigned short *blk_sad,
                            unsigned short *frame,
                            int mb_width,
                            int mb_height)
{
  int tx = (threadIdx.x / CEIL(MAX_POS, POS_PER_THREAD)) % THREADS_W;
  int ty = (threadIdx.x / CEIL(MAX_POS, POS_PER_THREAD)) / THREADS_W;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int img_width = mb_width*16;

  /* Macroblock and sub-block coordinates */
  int mb_x = (tx + bx * THREADS_W) >> 2;
  int mb_y = (ty + by * THREADS_H) >> 2;
  int block_x = (tx + bx * THREADS_W) & 0x03;
  int block_y = (ty + by * THREADS_H) & 0x03;

  /* If this thread is assigned to an invalid 4x4 block, do nothing */
  if ((mb_x < mb_width) && (mb_y < mb_height))
    {
      /* Pixel offset of the origin of the current 4x4 block */
      int frame_x = ((mb_x << 2) + block_x) << 2;
      int frame_y = ((mb_y << 2) + block_y) << 2;

      /* Origin of the search area for this 4x4 block */
      int ref_x = frame_x - SEARCH_RANGE;
      int ref_y = frame_y - SEARCH_RANGE;

      /* Origin in the current frame for this 4x4 block */
      int cur_o = frame_y * img_width + frame_x;

      int search_pos;
      int search_pos_base =
        (threadIdx.x % CEIL(MAX_POS, POS_PER_THREAD)) * POS_PER_THREAD;
      int search_pos_end = search_pos_base + POS_PER_THREAD;

      /* All SADs from this thread are stored in a contiguous chunk
       * of memory starting at this offset */
      blk_sad += mb_width * mb_height * MAX_POS_PADDED * (9 + 16) +
        (mb_y * mb_width + mb_x) * MAX_POS_PADDED * 16 +
        (4 * block_y + block_x) * MAX_POS_PADDED;

      /* Don't go past bounds */
      if (search_pos_end > MAX_POS)
        search_pos_end = MAX_POS;

      /* For each search position, within the range allocated to this thread */
      for (search_pos = search_pos_base;
           search_pos < search_pos_end;
           search_pos++) {
        unsigned short sad4x4 = 0;
        int search_off_x = ref_x + (search_pos % SEARCH_DIMENSION);
        int search_off_y = ref_y + (search_pos / SEARCH_DIMENSION);

        /* 4x4 SAD computation */
        for(int y=0; y<4; y++) {
          for (int x=0; x<4; x++) {
            sad4x4 +=
              abs(tex2D(ref, search_off_x + x, search_off_y + y) -
                  frame[cur_o + y * img_width + x]);
          }
        }

        /* Save this value into the local SAD array */
        blk_sad[search_pos] = sad4x4;
      }
      L2WB;
      SFENCE;
    }
}
__global__ void mb_sad_calc_nvm2(unsigned short *blk_sad,
                            unsigned short *frame,
                            int mb_width,
                            int mb_height)
{
  int tx = (threadIdx.x / CEIL(MAX_POS, POS_PER_THREAD)) % THREADS_W;
  int ty = (threadIdx.x / CEIL(MAX_POS, POS_PER_THREAD)) / THREADS_W;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int img_width = mb_width*16;

  /* Macroblock and sub-block coordinates */
  int mb_x = (tx + bx * THREADS_W) >> 2;
  int mb_y = (ty + by * THREADS_H) >> 2;
  int block_x = (tx + bx * THREADS_W) & 0x03;
  int block_y = (ty + by * THREADS_H) & 0x03;

  /* If this thread is assigned to an invalid 4x4 block, do nothing */
  if ((mb_x < mb_width) && (mb_y < mb_height))
    {
      /* Pixel offset of the origin of the current 4x4 block */
      int frame_x = ((mb_x << 2) + block_x) << 2;
      int frame_y = ((mb_y << 2) + block_y) << 2;

      /* Origin of the search area for this 4x4 block */
      int ref_x = frame_x - SEARCH_RANGE;
      int ref_y = frame_y - SEARCH_RANGE;

      /* Origin in the current frame for this 4x4 block */
      int cur_o = frame_y * img_width + frame_x;

      int search_pos;
      int search_pos_base =
        (threadIdx.x % CEIL(MAX_POS, POS_PER_THREAD)) * POS_PER_THREAD;
      int search_pos_end = search_pos_base + POS_PER_THREAD;

      /* All SADs from this thread are stored in a contiguous chunk
       * of memory starting at this offset */
      blk_sad += mb_width * mb_height * MAX_POS_PADDED * (9 + 16) +
        (mb_y * mb_width + mb_x) * MAX_POS_PADDED * 16 +
        (4 * block_y + block_x) * MAX_POS_PADDED;

      /* Don't go past bounds */
      if (search_pos_end > MAX_POS)
        search_pos_end = MAX_POS;

      /* For each search position, within the range allocated to this thread */
      for (search_pos = search_pos_base;
           search_pos < search_pos_end;
           search_pos++) {
        unsigned short sad4x4 = 0;
        int search_off_x = ref_x + (search_pos % SEARCH_DIMENSION);
        int search_off_y = ref_y + (search_pos / SEARCH_DIMENSION);

        /* 4x4 SAD computation */
        for(int y=0; y<4; y++) {
          for (int x=0; x<4; x++) {
            sad4x4 +=
              abs(tex2D(ref, search_off_x + x, search_off_y + y) -
                  frame[cur_o + y * img_width + x]);
          }
        }

        /* Save this value into the local SAD array */
        blk_sad[search_pos] = sad4x4;
      }
      L2WB;
      SFENCE; PCOMMIT; SFENCE;
    }
}


__global__ void mb_sad_calc_nvmj(unsigned short *blk_sad,
                            unsigned short *frame,
                            int mb_width,
                            int mb_height)
{
  int tx = (threadIdx.x / CEIL(MAX_POS, POS_PER_THREAD)) % THREADS_W;
  int ty = (threadIdx.x / CEIL(MAX_POS, POS_PER_THREAD)) / THREADS_W;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int img_width = mb_width*16;

  /* Macroblock and sub-block coordinates */
  int mb_x = (tx + bx * THREADS_W) >> 2;
  int mb_y = (ty + by * THREADS_H) >> 2;
  int block_x = (tx + bx * THREADS_W) & 0x03;
  int block_y = (ty + by * THREADS_H) & 0x03;

  /* If this thread is assigned to an invalid 4x4 block, do nothing */
  if ((mb_x < mb_width) && (mb_y < mb_height))
    {
      /* Pixel offset of the origin of the current 4x4 block */
      int frame_x = ((mb_x << 2) + block_x) << 2;
      int frame_y = ((mb_y << 2) + block_y) << 2;

      /* Origin of the search area for this 4x4 block */
      int ref_x = frame_x - SEARCH_RANGE;
      int ref_y = frame_y - SEARCH_RANGE;

      /* Origin in the current frame for this 4x4 block */
      int cur_o = frame_y * img_width + frame_x;

      int search_pos;
      int search_pos_base =
        (threadIdx.x % CEIL(MAX_POS, POS_PER_THREAD)) * POS_PER_THREAD;
      int search_pos_end = search_pos_base + POS_PER_THREAD;

      /* All SADs from this thread are stored in a contiguous chunk
       * of memory starting at this offset */
      blk_sad += mb_width * mb_height * MAX_POS_PADDED * (9 + 16) +
        (mb_y * mb_width + mb_x) * MAX_POS_PADDED * 16 +
        (4 * block_y + block_x) * MAX_POS_PADDED;

      /* Don't go past bounds */
      if (search_pos_end > MAX_POS)
        search_pos_end = MAX_POS;

      /* For each search position, within the range allocated to this thread */
      for (search_pos = search_pos_base;
           search_pos < search_pos_end;
           search_pos++) {
        unsigned short sad4x4 = 0;
        int search_off_x = ref_x + (search_pos % SEARCH_DIMENSION);
        int search_off_y = ref_y + (search_pos / SEARCH_DIMENSION);

        /* 4x4 SAD computation */
        for(int y=0; y<4; y++) {
          for (int x=0; x<4; x++) {
            sad4x4 +=
              abs(tex2D(ref, search_off_x + x, search_off_y + y) -
                  frame[cur_o + y * img_width + x]);
          }
        }

	ST_WT_U16(&blk_sad[search_pos], sad4x4);
	SFENCE; PCOMMIT; SFENCE;
      }
    }
}

__global__ void mb_sad_calc_nvml(unsigned short *blk_sad,
                            unsigned short *frame,
                            int mb_width,
                            int mb_height)
{
  int tx = (threadIdx.x / CEIL(MAX_POS, POS_PER_THREAD)) % THREADS_W;
  int ty = (threadIdx.x / CEIL(MAX_POS, POS_PER_THREAD)) / THREADS_W;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int img_width = mb_width*16;

  /* Macroblock and sub-block coordinates */
  int mb_x = (tx + bx * THREADS_W) >> 2;
  int mb_y = (ty + by * THREADS_H) >> 2;
  int block_x = (tx + bx * THREADS_W) & 0x03;
  int block_y = (ty + by * THREADS_H) & 0x03;

  /* If this thread is assigned to an invalid 4x4 block, do nothing */
  if ((mb_x < mb_width) && (mb_y < mb_height))
    {
      /* Pixel offset of the origin of the current 4x4 block */
      int frame_x = ((mb_x << 2) + block_x) << 2;
      int frame_y = ((mb_y << 2) + block_y) << 2;

      /* Origin of the search area for this 4x4 block */
      int ref_x = frame_x - SEARCH_RANGE;
      int ref_y = frame_y - SEARCH_RANGE;

      /* Origin in the current frame for this 4x4 block */
      int cur_o = frame_y * img_width + frame_x;

      int search_pos;
      int search_pos_base =
        (threadIdx.x % CEIL(MAX_POS, POS_PER_THREAD)) * POS_PER_THREAD;
      int search_pos_end = search_pos_base + POS_PER_THREAD;

      /* All SADs from this thread are stored in a contiguous chunk
       * of memory starting at this offset */
      blk_sad += mb_width * mb_height * MAX_POS_PADDED * (9 + 16) +
        (mb_y * mb_width + mb_x) * MAX_POS_PADDED * 16 +
        (4 * block_y + block_x) * MAX_POS_PADDED;

      /* Don't go past bounds */
      if (search_pos_end > MAX_POS)
        search_pos_end = MAX_POS;

      /* For each search position, within the range allocated to this thread */
      for (search_pos = search_pos_base;
           search_pos < search_pos_end;
           search_pos++) {
        unsigned short sad4x4 = 0;
        int search_off_x = ref_x + (search_pos % SEARCH_DIMENSION);
        int search_off_y = ref_y + (search_pos / SEARCH_DIMENSION);

        /* 4x4 SAD computation */
        for(int y=0; y<4; y++) {
          for (int x=0; x<4; x++) {
            sad4x4 +=
              abs(tex2D(ref, search_off_x + x, search_off_y + y) -
                  frame[cur_o + y * img_width + x]);
          }
        }

        /* Save this value into the local SAD array */
        //blk_sad[search_pos] = sad4x4;
	ST_WT_U16(&blk_sad[search_pos], sad4x4);
      }
      SFENCE; PCOMMIT; SFENCE;
    }
}
