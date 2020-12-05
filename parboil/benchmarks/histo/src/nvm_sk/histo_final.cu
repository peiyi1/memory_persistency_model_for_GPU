#include "util.h"
#include "nvm_util.h"

/* Combine all the sub-histogram results into one final histogram */
__global__ void histo_final_kernel (
    unsigned int sm_range_min, 
    unsigned int sm_range_max,
    unsigned int histo_height, 
    unsigned int histo_width,
    unsigned int *global_subhisto,
    unsigned int *global_histo,
    unsigned int *global_overflow,
    unsigned int *final_histo) //final output
{
    unsigned int start_offset = threadIdx.x + blockIdx.x * blockDim.x;
    const ushort4 zero_short  = {0, 0, 0, 0};
    const uint4 zero_int      = {0, 0, 0, 0};

    unsigned int size_low_histo = sm_range_min * BINS_PER_BLOCK;
    unsigned int size_mid_histo = (sm_range_max - sm_range_min +1) * BINS_PER_BLOCK;

    /* Clear lower region of global histogram */
    for (unsigned int i = start_offset; i < size_low_histo/4; i += gridDim.x * blockDim.x)
    {
        ushort4 global_histo_data = ((ushort4*)global_histo)[i];
	// Zhen: comment out the write to zeros, won't be used
        //((ushort4*)global_histo)[i] = zero_short;

        global_histo_data.x = min (global_histo_data.x, 255);
        global_histo_data.y = min (global_histo_data.y, 255);
        global_histo_data.z = min (global_histo_data.z, 255);
        global_histo_data.w = min (global_histo_data.w, 255);

        uchar4 final_histo_data = {
            global_histo_data.x,
            global_histo_data.y,
            global_histo_data.z,
            global_histo_data.w
        };

        ((uchar4*)final_histo)[i] = final_histo_data;
    }

    /* Clear the middle region of the overflow buffer */
    for (unsigned int i = (size_low_histo/4) + start_offset; i < (size_low_histo+size_mid_histo)/4; i += gridDim.x * blockDim.x)
    {
        uint4 global_histo_data = ((uint4*)global_overflow)[i];
        //((uint4*)global_overflow)[i] = zero_int;

        uint4 internal_histo_data = {
            global_histo_data.x,
            global_histo_data.y,
            global_histo_data.z,
            global_histo_data.w
        };

        unsigned int bin4in0 = ((unsigned int*)global_subhisto)[i*4];
        unsigned int bin4in1 = ((unsigned int*)global_subhisto)[i*4+1];
        unsigned int bin4in2 = ((unsigned int*)global_subhisto)[i*4+2];
        unsigned int bin4in3 = ((unsigned int*)global_subhisto)[i*4+3];

        internal_histo_data.x = min (bin4in0, 255);
        internal_histo_data.y = min (bin4in1, 255);
        internal_histo_data.z = min (bin4in2, 255);
        internal_histo_data.w = min (bin4in3, 255);

        uchar4 final_histo_data = {
            internal_histo_data.x,
            internal_histo_data.y,
            internal_histo_data.z,
            internal_histo_data.w
        };

        ((uchar4*)final_histo)[i] = final_histo_data;
    }

    /* Clear the upper region of global histogram */
    for (unsigned int i = ((size_low_histo+size_mid_histo)/4) + start_offset; i < (histo_height*histo_width)/4; i += gridDim.x * blockDim.x)
    {
        ushort4 global_histo_data = ((ushort4*)global_histo)[i];
        //((ushort4*)global_histo)[i] = zero_short;

        global_histo_data.x = min (global_histo_data.x, 255);
        global_histo_data.y = min (global_histo_data.y, 255);
        global_histo_data.z = min (global_histo_data.z, 255);
        global_histo_data.w = min (global_histo_data.w, 255);

        uchar4 final_histo_data = {
            global_histo_data.x,
            global_histo_data.y,
            global_histo_data.z,
            global_histo_data.w
        };

        ((uchar4*)final_histo)[i] = final_histo_data;
    }
}


__global__ void histo_final_kernel_nvma (
    unsigned int sm_range_min, 
    unsigned int sm_range_max,
    unsigned int histo_height, 
    unsigned int histo_width,
    unsigned int *global_subhisto,
    unsigned int *global_histo,
    unsigned int *global_overflow,
    unsigned int *final_histo) //final output
{
    unsigned int start_offset = threadIdx.x + blockIdx.x * blockDim.x;
    const ushort4 zero_short  = {0, 0, 0, 0};
    const uint4 zero_int      = {0, 0, 0, 0};

    unsigned int size_low_histo = sm_range_min * BINS_PER_BLOCK;
    unsigned int size_mid_histo = (sm_range_max - sm_range_min +1) * BINS_PER_BLOCK;

    /* Clear lower region of global histogram */
    for (unsigned int i = start_offset; i < size_low_histo/4; i += gridDim.x * blockDim.x)
    {
        ushort4 global_histo_data = ((ushort4*)global_histo)[i];
	// Zhen: comment out the write to zeros, won't be used
        //((ushort4*)global_histo)[i] = zero_short;

        global_histo_data.x = min (global_histo_data.x, 255);
        global_histo_data.y = min (global_histo_data.y, 255);
        global_histo_data.z = min (global_histo_data.z, 255);
        global_histo_data.w = min (global_histo_data.w, 255);

        uchar4 final_histo_data = {
            global_histo_data.x,
            global_histo_data.y,
            global_histo_data.z,
            global_histo_data.w
        };

	unsigned tmp = UCHAR4_TO_UINT(final_histo_data);
	final_histo[i] = tmp;
        //((uchar4*)final_histo)[i] = final_histo_data;
    }

    /* Clear the middle region of the overflow buffer */
    for (unsigned int i = (size_low_histo/4) + start_offset; i < (size_low_histo+size_mid_histo)/4; i += gridDim.x * blockDim.x)
    {
        uint4 global_histo_data = ((uint4*)global_overflow)[i];
        //((uint4*)global_overflow)[i] = zero_int;

        uint4 internal_histo_data = {
            global_histo_data.x,
            global_histo_data.y,
            global_histo_data.z,
            global_histo_data.w
        };

        unsigned int bin4in0 = ((unsigned int*)global_subhisto)[i*4];
        unsigned int bin4in1 = ((unsigned int*)global_subhisto)[i*4+1];
        unsigned int bin4in2 = ((unsigned int*)global_subhisto)[i*4+2];
        unsigned int bin4in3 = ((unsigned int*)global_subhisto)[i*4+3];

        internal_histo_data.x = min (bin4in0, 255);
        internal_histo_data.y = min (bin4in1, 255);
        internal_histo_data.z = min (bin4in2, 255);
        internal_histo_data.w = min (bin4in3, 255);

        uchar4 final_histo_data = {
            internal_histo_data.x,
            internal_histo_data.y,
            internal_histo_data.z,
            internal_histo_data.w
        };

	unsigned tmp = UCHAR4_TO_UINT(final_histo_data);
	final_histo[i] = tmp;
        //((uchar4*)final_histo)[i] = final_histo_data;
    }

    /* Clear the upper region of global histogram */
    for (unsigned int i = ((size_low_histo+size_mid_histo)/4) + start_offset; i < (histo_height*histo_width)/4; i += gridDim.x * blockDim.x)
    {
        ushort4 global_histo_data = ((ushort4*)global_histo)[i];
        //((ushort4*)global_histo)[i] = zero_short;

        global_histo_data.x = min (global_histo_data.x, 255);
        global_histo_data.y = min (global_histo_data.y, 255);
        global_histo_data.z = min (global_histo_data.z, 255);
        global_histo_data.w = min (global_histo_data.w, 255);

        uchar4 final_histo_data = {
            global_histo_data.x,
            global_histo_data.y,
            global_histo_data.z,
            global_histo_data.w
        };

	unsigned tmp = UCHAR4_TO_UINT(final_histo_data);
	final_histo[i] = tmp;
	//((uchar4*)final_histo)[i] = final_histo_data;
    }
}

__global__ void histo_final_kernel_nvmb (
    unsigned int sm_range_min, 
    unsigned int sm_range_max,
    unsigned int histo_height, 
    unsigned int histo_width,
    unsigned int *global_subhisto,
    unsigned int *global_histo,
    unsigned int *global_overflow,
    unsigned int *final_histo) //final output
{
    unsigned int start_offset = threadIdx.x + blockIdx.x * blockDim.x;
    const ushort4 zero_short  = {0, 0, 0, 0};
    const uint4 zero_int      = {0, 0, 0, 0};

    unsigned int size_low_histo = sm_range_min * BINS_PER_BLOCK;
    unsigned int size_mid_histo = (sm_range_max - sm_range_min +1) * BINS_PER_BLOCK;

    /* Clear lower region of global histogram */
    for (unsigned int i = start_offset; i < size_low_histo/4; i += gridDim.x * blockDim.x)
    {
        ushort4 global_histo_data = ((ushort4*)global_histo)[i];
        //((ushort4*)global_histo)[i] = zero_short;

        global_histo_data.x = min (global_histo_data.x, 255);
        global_histo_data.y = min (global_histo_data.y, 255);
        global_histo_data.z = min (global_histo_data.z, 255);
        global_histo_data.w = min (global_histo_data.w, 255);

        uchar4 final_histo_data = {
            global_histo_data.x,
            global_histo_data.y,
            global_histo_data.z,
            global_histo_data.w
        };

        //((uchar4*)final_histo)[i] = final_histo_data;
	unsigned tmp = UCHAR4_TO_UINT(final_histo_data);
	ST_WT_INT(&final_histo[i], tmp);
	MEM_FENCE;
    }

    /* Clear the middle region of the overflow buffer */
    for (unsigned int i = (size_low_histo/4) + start_offset; i < (size_low_histo+size_mid_histo)/4; i += gridDim.x * blockDim.x)
    {
        uint4 global_histo_data = ((uint4*)global_overflow)[i];
        //((uint4*)global_overflow)[i] = zero_int;

        uint4 internal_histo_data = {
            global_histo_data.x,
            global_histo_data.y,
            global_histo_data.z,
            global_histo_data.w
        };

        unsigned int bin4in0 = ((unsigned int*)global_subhisto)[i*4];
        unsigned int bin4in1 = ((unsigned int*)global_subhisto)[i*4+1];
        unsigned int bin4in2 = ((unsigned int*)global_subhisto)[i*4+2];
        unsigned int bin4in3 = ((unsigned int*)global_subhisto)[i*4+3];

        internal_histo_data.x = min (bin4in0, 255);
        internal_histo_data.y = min (bin4in1, 255);
        internal_histo_data.z = min (bin4in2, 255);
        internal_histo_data.w = min (bin4in3, 255);

        uchar4 final_histo_data = {
            internal_histo_data.x,
            internal_histo_data.y,
            internal_histo_data.z,
            internal_histo_data.w
        };

        //((uchar4*)final_histo)[i] = final_histo_data;
	unsigned tmp = UCHAR4_TO_UINT(final_histo_data);
	ST_WT_INT(&final_histo[i], tmp);
	MEM_FENCE;
    }

    /* Clear the upper region of global histogram */
    for (unsigned int i = ((size_low_histo+size_mid_histo)/4) + start_offset; i < (histo_height*histo_width)/4; i += gridDim.x * blockDim.x)
    {
        ushort4 global_histo_data = ((ushort4*)global_histo)[i];
        //((ushort4*)global_histo)[i] = zero_short;

        global_histo_data.x = min (global_histo_data.x, 255);
        global_histo_data.y = min (global_histo_data.y, 255);
        global_histo_data.z = min (global_histo_data.z, 255);
        global_histo_data.w = min (global_histo_data.w, 255);

        uchar4 final_histo_data = {
            global_histo_data.x,
            global_histo_data.y,
            global_histo_data.z,
            global_histo_data.w
        };

        //((uchar4*)final_histo)[i] = final_histo_data;
	unsigned tmp = UCHAR4_TO_UINT(final_histo_data);
	ST_WT_INT(&final_histo[i], tmp);
	MEM_FENCE;
    }
}

__global__ void histo_final_kernel_nvmo (
    unsigned int sm_range_min, 
    unsigned int sm_range_max,
    unsigned int histo_height, 
    unsigned int histo_width,
    unsigned int *global_subhisto,
    unsigned int *global_histo,
    unsigned int *global_overflow,
    unsigned int *final_histo) //final output
{
    unsigned int start_offset = threadIdx.x + blockIdx.x * blockDim.x;
    const ushort4 zero_short  = {0, 0, 0, 0};
    const uint4 zero_int      = {0, 0, 0, 0};

    unsigned int size_low_histo = sm_range_min * BINS_PER_BLOCK;
    unsigned int size_mid_histo = (sm_range_max - sm_range_min +1) * BINS_PER_BLOCK;

    /* Clear lower region of global histogram */
    for (unsigned int i = start_offset; i < size_low_histo/4; i += gridDim.x * blockDim.x)
    {
        ushort4 global_histo_data = ((ushort4*)global_histo)[i];
	// Zhen: comment out the write to zeros, won't be used
        //((ushort4*)global_histo)[i] = zero_short;

        global_histo_data.x = min (global_histo_data.x, 255);
        global_histo_data.y = min (global_histo_data.y, 255);
        global_histo_data.z = min (global_histo_data.z, 255);
        global_histo_data.w = min (global_histo_data.w, 255);

        uchar4 final_histo_data = {
            global_histo_data.x,
            global_histo_data.y,
            global_histo_data.z,
            global_histo_data.w
        };

	unsigned tmp = UCHAR4_TO_UINT(final_histo_data);
	final_histo[i] = tmp;
        //((uchar4*)final_histo)[i] = final_histo_data;
	CLWB(&final_histo[i]); SFENCE;
    }

    /* Clear the middle region of the overflow buffer */
    for (unsigned int i = (size_low_histo/4) + start_offset; i < (size_low_histo+size_mid_histo)/4; i += gridDim.x * blockDim.x)
    {
        uint4 global_histo_data = ((uint4*)global_overflow)[i];
        //((uint4*)global_overflow)[i] = zero_int;

        uint4 internal_histo_data = {
            global_histo_data.x,
            global_histo_data.y,
            global_histo_data.z,
            global_histo_data.w
        };

        unsigned int bin4in0 = ((unsigned int*)global_subhisto)[i*4];
        unsigned int bin4in1 = ((unsigned int*)global_subhisto)[i*4+1];
        unsigned int bin4in2 = ((unsigned int*)global_subhisto)[i*4+2];
        unsigned int bin4in3 = ((unsigned int*)global_subhisto)[i*4+3];

        internal_histo_data.x = min (bin4in0, 255);
        internal_histo_data.y = min (bin4in1, 255);
        internal_histo_data.z = min (bin4in2, 255);
        internal_histo_data.w = min (bin4in3, 255);

        uchar4 final_histo_data = {
            internal_histo_data.x,
            internal_histo_data.y,
            internal_histo_data.z,
            internal_histo_data.w
        };

	unsigned tmp = UCHAR4_TO_UINT(final_histo_data);
	final_histo[i] = tmp;
        //((uchar4*)final_histo)[i] = final_histo_data;
	CLWB(&final_histo[i]); SFENCE;
    }

    /* Clear the upper region of global histogram */
    for (unsigned int i = ((size_low_histo+size_mid_histo)/4) + start_offset; i < (histo_height*histo_width)/4; i += gridDim.x * blockDim.x)
    {
        ushort4 global_histo_data = ((ushort4*)global_histo)[i];
        //((ushort4*)global_histo)[i] = zero_short;

        global_histo_data.x = min (global_histo_data.x, 255);
        global_histo_data.y = min (global_histo_data.y, 255);
        global_histo_data.z = min (global_histo_data.z, 255);
        global_histo_data.w = min (global_histo_data.w, 255);

        uchar4 final_histo_data = {
            global_histo_data.x,
            global_histo_data.y,
            global_histo_data.z,
            global_histo_data.w
        };

	unsigned tmp = UCHAR4_TO_UINT(final_histo_data);
	final_histo[i] = tmp;
        //((uchar4*)final_histo)[i] = final_histo_data;
	CLWB(&final_histo[i]); SFENCE;
    }
}
__global__ void histo_final_kernel_nvmu (
    unsigned int sm_range_min, 
    unsigned int sm_range_max,
    unsigned int histo_height, 
    unsigned int histo_width,
    unsigned int *global_subhisto,
    unsigned int *global_histo,
    unsigned int *global_overflow,
    unsigned int *final_histo) //final output
{
    unsigned int start_offset = threadIdx.x + blockIdx.x * blockDim.x;
    const ushort4 zero_short  = {0, 0, 0, 0};
    const uint4 zero_int      = {0, 0, 0, 0};

    unsigned int size_low_histo = sm_range_min * BINS_PER_BLOCK;
    unsigned int size_mid_histo = (sm_range_max - sm_range_min +1) * BINS_PER_BLOCK;

    /* Clear lower region of global histogram */
    for (unsigned int i = start_offset; i < size_low_histo/4; i += gridDim.x * blockDim.x)
    {
        ushort4 global_histo_data = ((ushort4*)global_histo)[i];
	// Zhen: comment out the write to zeros, won't be used
        //((ushort4*)global_histo)[i] = zero_short;

        global_histo_data.x = min (global_histo_data.x, 255);
        global_histo_data.y = min (global_histo_data.y, 255);
        global_histo_data.z = min (global_histo_data.z, 255);
        global_histo_data.w = min (global_histo_data.w, 255);

        uchar4 final_histo_data = {
            global_histo_data.x,
            global_histo_data.y,
            global_histo_data.z,
            global_histo_data.w
        };

	unsigned tmp = UCHAR4_TO_UINT(final_histo_data);
	final_histo[i] = tmp;
        //((uchar4*)final_histo)[i] = final_histo_data;
	CLWB(&final_histo[i]); SFENCE; PCOMMIT; SFENCE;
    }

    /* Clear the middle region of the overflow buffer */
    for (unsigned int i = (size_low_histo/4) + start_offset; i < (size_low_histo+size_mid_histo)/4; i += gridDim.x * blockDim.x)
    {
        uint4 global_histo_data = ((uint4*)global_overflow)[i];
        //((uint4*)global_overflow)[i] = zero_int;

        uint4 internal_histo_data = {
            global_histo_data.x,
            global_histo_data.y,
            global_histo_data.z,
            global_histo_data.w
        };

        unsigned int bin4in0 = ((unsigned int*)global_subhisto)[i*4];
        unsigned int bin4in1 = ((unsigned int*)global_subhisto)[i*4+1];
        unsigned int bin4in2 = ((unsigned int*)global_subhisto)[i*4+2];
        unsigned int bin4in3 = ((unsigned int*)global_subhisto)[i*4+3];

        internal_histo_data.x = min (bin4in0, 255);
        internal_histo_data.y = min (bin4in1, 255);
        internal_histo_data.z = min (bin4in2, 255);
        internal_histo_data.w = min (bin4in3, 255);

        uchar4 final_histo_data = {
            internal_histo_data.x,
            internal_histo_data.y,
            internal_histo_data.z,
            internal_histo_data.w
        };

	unsigned tmp = UCHAR4_TO_UINT(final_histo_data);
	final_histo[i] = tmp;
        //((uchar4*)final_histo)[i] = final_histo_data;
	CLWB(&final_histo[i]); SFENCE; PCOMMIT; SFENCE;
    }

    /* Clear the upper region of global histogram */
    for (unsigned int i = ((size_low_histo+size_mid_histo)/4) + start_offset; i < (histo_height*histo_width)/4; i += gridDim.x * blockDim.x)
    {
        ushort4 global_histo_data = ((ushort4*)global_histo)[i];
        //((ushort4*)global_histo)[i] = zero_short;

        global_histo_data.x = min (global_histo_data.x, 255);
        global_histo_data.y = min (global_histo_data.y, 255);
        global_histo_data.z = min (global_histo_data.z, 255);
        global_histo_data.w = min (global_histo_data.w, 255);

        uchar4 final_histo_data = {
            global_histo_data.x,
            global_histo_data.y,
            global_histo_data.z,
            global_histo_data.w
        };

	unsigned tmp = UCHAR4_TO_UINT(final_histo_data);
	final_histo[i] = tmp;
        //((uchar4*)final_histo)[i] = final_histo_data;
	CLWB(&final_histo[i]); SFENCE; PCOMMIT; SFENCE;
    }
}
