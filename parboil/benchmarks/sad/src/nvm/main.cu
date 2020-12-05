/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <inttypes.h>
#include <parboil.h>
#include <cuda.h>

#include "sad.h"
#include "sad4.h"
#include "largerBlocks.h"
#include "file.h"
#include "image.h"

#define CUDA_ERRCK \
  {cudaError_t err = cudaGetLastError(); \
    if (err) fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
  }

static unsigned short *
load_sads(char *filename);
static void
write_sads(char *filename,
	   int image_width_macroblocks,
	   int image_height_macroblocks,
	   unsigned short *sads);
static void
write_sads_directly(char *filename,
		    int width,
		    int height,
		    unsigned short *sads);

__global__ void kernel_l2wb(void){
	L2WB;
	MEM_FENCE;
}
__global__ void kernel_l2wb_pct(void){
        L2WB;
        MEM_FENCE;
	 PCOMMIT; MEM_FENCE;
}
/* FILE I/O */

unsigned short *
load_sads(char *filename)
{
  FILE *infile;
  unsigned short *sads;
  int w;
  int h;
  int sads_per_block;

  infile = fopen(filename, "r");

  if (!infile)
    {
      fprintf(stderr, "Cannot find file '%s'\n", filename);
      exit(-1);
    }

  /* Read image dimensions (measured in macroblocks) */
  w = read16u(infile);
  h = read16u(infile);

  /* Read SAD values.  Only interested in the 4x4 SAD values, which are
   * at the end of the file. */
  sads_per_block = MAX_POS_PADDED * (w * h);
  fseek(infile, 25 * sads_per_block * sizeof(unsigned short), SEEK_CUR);

  sads = (unsigned short *)malloc(sads_per_block * 16 * sizeof(unsigned short));
  fread(sads, sizeof(unsigned short), sads_per_block * 16, infile);
  fclose(infile);

  return sads;
}

/* Compare the reference SADs to the expected SADs.
 */
void
check_sads(unsigned short *sads_reference,
	   unsigned short *sads_computed,
	   int image_size_macroblocks)
{
  int block;

  /* Check the 4x4 SAD values.  These are in sads_reference.
   * Ignore the data at the beginning of sads_computed. */
  sads_computed += 25 * MAX_POS_PADDED * image_size_macroblocks;

  for (block = 0; block < image_size_macroblocks; block++)
    {
      int subblock;

      for (subblock = 0; subblock < 16; subblock++)
	{
	  int sad_index;

	  for (sad_index = 0; sad_index < MAX_POS; sad_index++)
	    {
	      int index =
		(block * 16 + subblock) * MAX_POS_PADDED + sad_index;

	      if (sads_reference[index] != sads_computed[index])
		{
#if 0
		  /* Print exactly where the mismatch was seen */
		  printf("M %3d %2d %4d (%d = %d)\n", block, subblock, sad_index, sads_reference[index], sads_computed[index]);
#else
		  goto mismatch;
#endif
		}
	    }
	}
    }

  printf("Success.\n");
  return;

 mismatch:
  printf("Computed SADs do not match expected values.\n");
}

/* Extract the SAD data for a particular block type for a particular
 * macroblock from the array of SADs of that block type. */
static inline void
write_subblocks(FILE *outfile, unsigned short *subblock_array, int macroblock,
		int count)
{
  int block;
  int pos;

  for (block = 0; block < count; block++)
    {
      unsigned short *vec = subblock_array +
	(block + macroblock * count) * MAX_POS_PADDED;

      /* Write all SADs for this sub-block */
      for (pos = 0; pos < MAX_POS; pos++)
	write16u(outfile, *vec++);
    }
}

/* Write some SAD data to a file for output checking.
 *
 * All SAD values for six rows of macroblocks are written.
 * The six rows consist of the top two, middle two, and bottom two image rows.
 */
void
write_sads(char *filename,
	   int mb_width,
	   int mb_height,
	   unsigned short *sads)
{
  FILE *outfile = fopen(filename, "w");
  int mbs = mb_width * mb_height;
  int row_indir;
  int row_indices[6] = {0, 1,
			mb_height / 2 - 1, mb_height / 2,
			mb_height - 2, mb_height - 1};

  if (outfile == NULL)
    {
      fprintf(stderr, "Cannot open output file\n");
      exit(-1);
    }

  /* Write the number of output macroblocks */
  write32u(outfile, mb_width * 6);

  /* Write zeros */
  write32u(outfile, 0);

  /* Each row */
  for (row_indir = 0; row_indir < 6; row_indir++)
    {
      int row = row_indices[row_indir];

      /* Each block in row */
      int block;
      for (block = mb_width * row; block < mb_width * (row + 1); block++)
	{
	  int blocktype;

	  /* Write SADs for all sub-block types */
	  for (blocktype = 1; blocktype <= 7; blocktype++)
	    write_subblocks(outfile,
			    sads + SAD_TYPE_IX(blocktype, mbs),
			    block,
			    SAD_TYPE_CT(blocktype));
	}
    }

  fclose(outfile);
}

/* FILE I/O for debugging */

static void
write_sads_directly(char *filename,
		    int width,
		    int height,
		    unsigned short *sads)
{
  FILE *f = fopen(filename, "w");
  int n;

  write16u(f, width);
  write16u(f, height);
  for (n = 0; n < 41 * MAX_POS_PADDED * (width * height); n++) {
    write16u(f, sads[n]);
  }
  fclose(f);
}

static void
print_test_sad_vector(unsigned short *base, int macroblock, int count)
{
  int n;
  int searchpos = 17*33+17;
  for (n = 0; n < count; n++)
    printf(" %d", base[(count * macroblock + n) * MAX_POS_PADDED + searchpos]);
}

static void
print_test_sads(unsigned short *sads_computed,
		int mbs)
{
  int macroblock = 5;
  int blocktype;

  for (blocktype = 1; blocktype <= 7; blocktype++)
    {
      printf("%d:", blocktype);
      print_test_sad_vector(sads_computed + SAD_TYPE_IX(blocktype, mbs),
			    macroblock, SAD_TYPE_CT(blocktype));
      puts("\n");
    }
}

/* MAIN */
char nvm_opt;

int
main(int argc, char **argv)
{
  nvm_opt = *argv[--argc];
  
  struct image_i16 *ref_image;
  struct image_i16 *cur_image;
  unsigned short *sads_computed; /* SADs generated by the program */

  int image_size_bytes;
  int image_width_macroblocks, image_height_macroblocks;
  int image_size_macroblocks;

  struct pb_TimerSet timers;
  struct pb_Parameters *params;

  pb_InitializeTimerSet(&timers);
  params = pb_ReadParameters(&argc, argv);

  if (pb_Parameters_CountInputs(params) != 2)
    {
      fprintf(stderr, "Expecting two input filenames\n");
      exit(-1);
    }

  /* Read input files */
  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  ref_image = load_image(params->inpFiles[0]);
  cur_image = load_image(params->inpFiles[1]);
  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  if ((ref_image->width != cur_image->width) ||
      (ref_image->height != cur_image->height))
    {
      fprintf(stderr, "Input images must be the same size\n");
      exit(-1);
    }
  if ((ref_image->width % 16) || (ref_image->height % 16))
    {
      fprintf(stderr, "Input image size must be an integral multiple of 16\n");
      exit(-1);
    }

  /* Compute parameters, allocate memory */
  image_size_bytes = ref_image->width * ref_image->height * sizeof(short);
  image_width_macroblocks = ref_image->width >> 4;
  image_height_macroblocks = ref_image->height >> 4;
  image_size_macroblocks = image_width_macroblocks * image_height_macroblocks;
  
  sads_computed = (unsigned short *)
    malloc(41 * MAX_POS_PADDED * image_size_macroblocks * sizeof(short));

  /* Run the kernel code */
  {
    struct cudaArray *ref_ary;  /* Reference image on the device */
    short *d_cur_image;         /* Current image on the device */
    unsigned short *d_sads;     /* SADs on the device */
    dim3 macroblock_grid(image_width_macroblocks, image_height_macroblocks);

    pb_SwitchToTimer(&timers, pb_TimerID_COPY);
    cudaMalloc((void **)&d_cur_image, image_size_bytes);
    CUDA_ERRCK
    cudaMallocArray(&ref_ary, &get_ref().channelDesc,
                    ref_image->width, ref_image->height);
    CUDA_ERRCK

    /* Transfer current image to device */
    cudaMemcpy(d_cur_image, cur_image->data, image_size_bytes,
               cudaMemcpyHostToDevice);
    CUDA_ERRCK

    /* Transfer reference image to device */
    cudaMemcpy2DToArray(ref_ary,
                        0, 0,
                        ref_image->data,
                        ref_image->width * sizeof(unsigned short),
                        ref_image->width * sizeof(unsigned short),
                        ref_image->height,
                        cudaMemcpyHostToDevice);
    CUDA_ERRCK
    cudaBindTextureToArray(get_ref(), ref_ary);
    CUDA_ERRCK

    /* Allocate SAD data on the device */
    cudaMalloc((void **)&d_sads, 41 * MAX_POS_PADDED * image_size_macroblocks *
	       sizeof(unsigned short));
    CUDA_ERRCK;
    unsigned short *NVM_klog;
    cudaMalloc((void **)&NVM_klog, 41 * MAX_POS_PADDED * image_size_macroblocks *
	       sizeof(unsigned short));
    CUDA_ERRCK
    cudaMemset(d_sads, 0, 41 * MAX_POS_PADDED * image_size_macroblocks *
	       sizeof(unsigned short));
    CUDA_ERRCK

    pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);

    printf("width * height = %d\n", image_width_macroblocks * image_height_macroblocks);
    syncLapTimer st_mb, st_8, st_16;
    for (int i = 0; i < 1; i++) {
    // Run the 4x4 kernel
      st_mb.lap_start();
    if (nvm_opt == 'a') {
      mb_sad_calc<<<dim3(CEIL(ref_image->width / 4, THREADS_W),
			 CEIL(ref_image->height / 4, THREADS_H)),
	dim3(CEIL(MAX_POS, POS_PER_THREAD) * THREADS_W * THREADS_H),
	SAD_LOC_SIZE_BYTES>>>
	(d_sads,
	 (unsigned short *)d_cur_image,
	 image_width_macroblocks,
	 image_height_macroblocks);
    } else if (nvm_opt == 'b') {
      mb_sad_calc_nvmb<<<dim3(CEIL(ref_image->width / 4, THREADS_W),
			 CEIL(ref_image->height / 4, THREADS_H)),
	dim3(CEIL(MAX_POS, POS_PER_THREAD) * THREADS_W * THREADS_H),
	SAD_LOC_SIZE_BYTES>>>
	(d_sads,
	 (unsigned short *)d_cur_image,
	 image_width_macroblocks,
	 image_height_macroblocks);
    } else if (nvm_opt == 'd') {
      mb_sad_calc_nvmd<<<dim3(CEIL(ref_image->width / 4, THREADS_W),
			 CEIL(ref_image->height / 4, THREADS_H)),
	dim3(CEIL(MAX_POS, POS_PER_THREAD) * THREADS_W * THREADS_H),
	SAD_LOC_SIZE_BYTES>>>
	(d_sads,
	 (unsigned short *)d_cur_image,
	 image_width_macroblocks,
	 image_height_macroblocks);
    } else if (nvm_opt == 'g') {
      mb_sad_calc_nvmg<<<dim3(CEIL(ref_image->width / 4, THREADS_W),
			 CEIL(ref_image->height / 4, THREADS_H)),
	dim3(CEIL(MAX_POS, POS_PER_THREAD) * THREADS_W * THREADS_H),
	SAD_LOC_SIZE_BYTES>>>
	(d_sads,
	 (unsigned short *)d_cur_image,
	 image_width_macroblocks,
	 image_height_macroblocks);
    } else if (nvm_opt == 'i') {
      mb_sad_calc_nvmi<<<dim3(CEIL(ref_image->width / 4, THREADS_W),
			 CEIL(ref_image->height / 4, THREADS_H)),
	dim3(CEIL(MAX_POS, POS_PER_THREAD) * THREADS_W * THREADS_H),
	SAD_LOC_SIZE_BYTES>>>
	(d_sads,
	 (unsigned short *)d_cur_image,
	 image_width_macroblocks,
	 image_height_macroblocks);
    } else if (nvm_opt == 'o') {
      mb_sad_calc_nvmo<<<dim3(CEIL(ref_image->width / 4, THREADS_W),
			 CEIL(ref_image->height / 4, THREADS_H)),
	dim3(CEIL(MAX_POS, POS_PER_THREAD) * THREADS_W * THREADS_H),
	SAD_LOC_SIZE_BYTES>>>
	(d_sads,
	 (unsigned short *)d_cur_image,
	 image_width_macroblocks,
	 image_height_macroblocks);
    } else if (nvm_opt == 'q') {
      mb_sad_calc_nvmq<<<dim3(CEIL(ref_image->width / 4, THREADS_W),
			 CEIL(ref_image->height / 4, THREADS_H)),
	dim3(CEIL(MAX_POS, POS_PER_THREAD) * THREADS_W * THREADS_H),
	SAD_LOC_SIZE_BYTES>>>
	(d_sads,
	 (unsigned short *)d_cur_image,
	 image_width_macroblocks,
	 image_height_macroblocks);
    } else if (nvm_opt == 'j') {
      mb_sad_calc_nvmj<<<dim3(CEIL(ref_image->width / 4, THREADS_W),
			 CEIL(ref_image->height / 4, THREADS_H)),
	dim3(CEIL(MAX_POS, POS_PER_THREAD) * THREADS_W * THREADS_H),
	SAD_LOC_SIZE_BYTES>>>
	(d_sads,
	 (unsigned short *)d_cur_image,
	 image_width_macroblocks,
	 image_height_macroblocks);
    } else if (nvm_opt == 'l') {
      mb_sad_calc_nvml<<<dim3(CEIL(ref_image->width / 4, THREADS_W),
			 CEIL(ref_image->height / 4, THREADS_H)),
	dim3(CEIL(MAX_POS, POS_PER_THREAD) * THREADS_W * THREADS_H),
	SAD_LOC_SIZE_BYTES>>>
	(d_sads,
	 (unsigned short *)d_cur_image,
	 image_width_macroblocks,
	 image_height_macroblocks);
    } else if (nvm_opt == 'u') {
      mb_sad_calc_nvmu<<<dim3(CEIL(ref_image->width / 4, THREADS_W),
			 CEIL(ref_image->height / 4, THREADS_H)),
	dim3(CEIL(MAX_POS, POS_PER_THREAD) * THREADS_W * THREADS_H),
	SAD_LOC_SIZE_BYTES>>>
	(d_sads,
	 (unsigned short *)d_cur_image,
	 image_width_macroblocks,
	 image_height_macroblocks);
    } else if (nvm_opt == 'w') {
      mb_sad_calc_nvmw<<<dim3(CEIL(ref_image->width / 4, THREADS_W),
			 CEIL(ref_image->height / 4, THREADS_H)),
	dim3(CEIL(MAX_POS, POS_PER_THREAD) * THREADS_W * THREADS_H),
	SAD_LOC_SIZE_BYTES>>>
	(d_sads,
	 (unsigned short *)d_cur_image,
	 image_width_macroblocks,
	 image_height_macroblocks);
    } else if (nvm_opt == '1') {
      mb_sad_calc_nvm1<<<dim3(CEIL(ref_image->width / 4, THREADS_W),
			 CEIL(ref_image->height / 4, THREADS_H)),
	dim3(CEIL(MAX_POS, POS_PER_THREAD) * THREADS_W * THREADS_H),
	SAD_LOC_SIZE_BYTES>>>
	(d_sads,
	 (unsigned short *)d_cur_image,
	 image_width_macroblocks,
	 image_height_macroblocks);
    } else if (nvm_opt == '2') {
      mb_sad_calc_nvm2<<<dim3(CEIL(ref_image->width / 4, THREADS_W),
			 CEIL(ref_image->height / 4, THREADS_H)),
	dim3(CEIL(MAX_POS, POS_PER_THREAD) * THREADS_W * THREADS_H),
	SAD_LOC_SIZE_BYTES>>>
	(d_sads,
	 (unsigned short *)d_cur_image,
	 image_width_macroblocks,
	 image_height_macroblocks);
    } else if (nvm_opt == '3') {
      mb_sad_calc_nvm3<<<dim3(CEIL(ref_image->width / 4, THREADS_W),
			 CEIL(ref_image->height / 4, THREADS_H)),
	dim3(CEIL(MAX_POS, POS_PER_THREAD) * THREADS_W * THREADS_H),
	SAD_LOC_SIZE_BYTES>>>
	(d_sads,
	 (unsigned short *)d_cur_image,
	 image_width_macroblocks,
	 image_height_macroblocks);
    } else if (nvm_opt == '4') {
      mb_sad_calc_nvm4<<<dim3(CEIL(ref_image->width / 4, THREADS_W),
			 CEIL(ref_image->height / 4, THREADS_H)),
	dim3(CEIL(MAX_POS, POS_PER_THREAD) * THREADS_W * THREADS_H),
	SAD_LOC_SIZE_BYTES>>>
	(d_sads,
	 (unsigned short *)d_cur_image,
	 image_width_macroblocks,
	 image_height_macroblocks);
    } else if (nvm_opt == '5') {
      mb_sad_calc_nvm5<<<dim3(CEIL(ref_image->width / 4, THREADS_W),
			 CEIL(ref_image->height / 4, THREADS_H)),
	dim3(CEIL(MAX_POS, POS_PER_THREAD) * THREADS_W * THREADS_H),
	SAD_LOC_SIZE_BYTES>>>
	(d_sads,
	 (unsigned short *)d_cur_image,
	 image_width_macroblocks,
	 image_height_macroblocks);
    } else if (nvm_opt == '6') {
      mb_sad_calc_nvm6<<<dim3(CEIL(ref_image->width / 4, THREADS_W),
			 CEIL(ref_image->height / 4, THREADS_H)),
	dim3(CEIL(MAX_POS, POS_PER_THREAD) * THREADS_W * THREADS_H),
	SAD_LOC_SIZE_BYTES>>>
	(d_sads,
	 (unsigned short *)d_cur_image,
	 image_width_macroblocks,
	 image_height_macroblocks);
    }else{
      mb_sad_calc<<<dim3(CEIL(ref_image->width / 4, THREADS_W),
                         CEIL(ref_image->height / 4, THREADS_H)),
        dim3(CEIL(MAX_POS, POS_PER_THREAD) * THREADS_W * THREADS_H),
        SAD_LOC_SIZE_BYTES>>>
        (d_sads,
         (unsigned short *)d_cur_image,
         image_width_macroblocks,
         image_height_macroblocks);
    }

    st_mb.lap_end();
    CUDA_ERRCK;

    // Run the larger-blocks kernels
    st_8.lap_start();
    if (nvm_opt == 'a') {
      larger_sad_calc_8<<<macroblock_grid, dim3(32, 4)>>>
	(d_sads,
	 image_width_macroblocks,
	 image_height_macroblocks);
    } else if (nvm_opt == 'b') {
      larger_sad_calc_8_nvmb<<<macroblock_grid, dim3(32, 4)>>>
	(d_sads,
	 image_width_macroblocks,
	 image_height_macroblocks);
    } else if (nvm_opt == 'o') {
      larger_sad_calc_8_nvmo<<<macroblock_grid, dim3(32, 4)>>>
	(d_sads,
	 image_width_macroblocks,
	 image_height_macroblocks);
    } else if (nvm_opt == 'u') {
      larger_sad_calc_8_nvmu<<<macroblock_grid, dim3(32, 4)>>>
	(d_sads,
	 image_width_macroblocks,
	 image_height_macroblocks);
    } else if (nvm_opt == 'f') {
      cudaMemcpy(NVM_klog, d_sads, 41 * MAX_POS_PADDED * image_size_macroblocks * sizeof(unsigned short), cudaMemcpyDeviceToDevice);
      cudaDeviceSynchronize();
      larger_sad_calc_8<<<macroblock_grid, dim3(32, 4)>>>
	(d_sads,
	 image_width_macroblocks,
	 image_height_macroblocks);
    } else if (nvm_opt == 'k'){
	larger_sad_calc_8<<<macroblock_grid, dim3(32, 4)>>>
        (d_sads,
         image_width_macroblocks,
         image_height_macroblocks);
	cudaDeviceSynchronize();
	kernel_l2wb<<<macroblock_grid, dim3(32, 4)>>>();
    }else if (nvm_opt == 'l'){
        larger_sad_calc_8<<<macroblock_grid, dim3(32, 4)>>>
        (d_sads,
         image_width_macroblocks,
         image_height_macroblocks);
        cudaDeviceSynchronize();
        kernel_l2wb_pct<<<macroblock_grid, dim3(32, 4)>>>();
    }else if (nvm_opt == 'm') {
      cudaMemcpy(NVM_klog, d_sads, 41 * MAX_POS_PADDED * image_size_macroblocks * sizeof(unsigned short), cudaMemcpyDeviceToDevice);
      cudaDeviceSynchronize();
      larger_sad_calc_8<<<macroblock_grid, dim3(32, 4)>>>
        (d_sads,
         image_width_macroblocks,
         image_height_macroblocks);
        cudaDeviceSynchronize();
        kernel_l2wb<<<macroblock_grid, dim3(32, 4)>>>();

    } 
    st_8.lap_end();
    CUDA_ERRCK;
    exit(0);
    st_16.lap_start();
/*    if (nvm_opt == 'a') {
      larger_sad_calc_16<<<macroblock_grid, dim3(32, 1)>>>
	(d_sads,
	 image_width_macroblocks,
	 image_height_macroblocks);
    }
*/
    st_16.lap_end();
    CUDA_ERRCK;
    } // for 1000

    st_mb.print_avg_usec("mb_sad_calc", CEIL(ref_image->width / 4, THREADS_W) * CEIL(ref_image->height / 4, THREADS_H));
    st_8.print_avg_usec("larger_sad_calc_8", DIM3_NTBS(macroblock_grid));
    st_16.print_avg_usec("larger_sad_calc_16", DIM3_NTBS(macroblock_grid));
    pb_SwitchToTimer(&timers, pb_TimerID_COPY);

    /* Transfer SAD data to the host */
    cudaMemcpy(sads_computed,// + 25 * MAX_POS_PADDED * image_size_macroblocks,
	       d_sads,// + 25 * MAX_POS_PADDED * image_size_macroblocks,
	       41 * MAX_POS_PADDED * image_size_macroblocks * sizeof(unsigned short)
,
           cudaMemcpyDeviceToHost);
    CUDA_ERRCK

    /* Free GPU memory */
    cudaFree(d_sads);
    CUDA_ERRCK
    cudaUnbindTexture(get_ref());
    CUDA_ERRCK
    cudaFreeArray(ref_ary);
    CUDA_ERRCK
    cudaFree(d_cur_image);
    CUDA_ERRCK

    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
  }

  /* Print output */
  if (params->outFile)
    {
      pb_SwitchToTimer(&timers, pb_TimerID_IO);
      write_sads(params->outFile,
		 image_width_macroblocks,
		 image_height_macroblocks,
		 sads_computed);
      pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
    }

#if 0  /* Debugging */
  print_test_sads(sads_computed, image_size_macroblocks);
  write_sads_directly("sad-debug.bin",
		      ref_image->width / 16, ref_image->height / 16,
		      sads_computed);
#endif

  /* Free memory */
  free(sads_computed);
  free_image(ref_image);
  free_image(cur_image);

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(params);

  return 0;
}
