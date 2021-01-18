#pragma once

#include <stdexcept>
#include <string>
#include <sstream>
#include <stdint.h>
#include <cuda_runtime.h>

// It is virtually impossible to get more than
// one solution per stream hash calculation
// Leave room for up to 4 results. A power
// of 2 here will yield better CUDA optimization
//pli11
//#define SEARCH_RESULTS 4
#define SEARCH_RESULTS 0x10000
//pli11
#define min(a,b) ((a<b) ? a : b)
#define mul_hi(a, b) __umulhi(a, b)
#define clz(a) __clz(a)
#define popcount(a) __popc(a)

#define PROGPOW_LANES           16
#define PROGPOW_REGS            32
#define PROGPOW_DAG_LOADS       4
#define PROGPOW_CACHE_WORDS     4096
#define PROGPOW_CNT_DAG         64
#define PROGPOW_CNT_MATH        18

typedef struct __align__(PROGPOW_DAG_LOADS * 4) {uint32_t s[PROGPOW_DAG_LOADS];} dag_t;

#define PROGPOW_DAG_ELEMENTS    4227071
//

typedef struct {
	uint32_t count;
	struct {
		// One word for gid and 8 for mix hash
		uint32_t gid;
		uint32_t mix[8];
	} result[SEARCH_RESULTS];
} search_results;

//pli11
//typedef struct
typedef union
{
	uint4 uint4s[32 / sizeof(uint4)];
	uint32_t uint32s[32 / sizeof(uint32_t)];
} hash32_t;

typedef struct
{
	uint64_t uint64s[256 / sizeof(uint64_t)];
} hash256_t;

typedef union {
	uint32_t words[64 / sizeof(uint32_t)];
	uint2	 uint2s[64 / sizeof(uint2)];
	uint4	 uint4s[64 / sizeof(uint4)];
} hash64_t;

typedef union {
	uint32_t words[200 / sizeof(uint32_t)];
	uint64_t uint64s[200 / sizeof(uint64_t)];
	uint2	 uint2s[200 / sizeof(uint2)];
	uint4	 uint4s[200 / sizeof(uint4)];
} hash200_t;

void ethash_generate_dag(
	hash64_t* dag,
	uint64_t dag_bytes,
	hash64_t * light,
	uint32_t light_words,
	uint32_t blocks,
	uint32_t threads,
	cudaStream_t stream,
	int device
	);
//pli11
void search_kernel(
    uint64_t start_nonce,
    const hash32_t header,
    const uint64_t target,
    const dag_t *g_dag,
	search_results* g_output,
    bool hack_false,
        uint32_t blocks,
        uint32_t threads,
        cudaStream_t stream
    );

//
struct cuda_runtime_error : public virtual std::runtime_error
{
	cuda_runtime_error( std::string msg ) : std::runtime_error(msg) {}
};

#define CUDA_SAFE_CALL(call)				\
do {							\
	cudaError_t result = call;				\
	if (cudaSuccess != result) {			\
		std::stringstream ss;			\
		ss << "CUDA error in func " 		\
            << __FUNCTION__ 		\
			<< " at line "			\
			<< __LINE__			\
			<< " calling " #call " failed with error "     \
			<< cudaGetErrorString(result);	\
		throw cuda_runtime_error(ss.str());	\
	}						\
} while (0)

#define CU_SAFE_CALL(call)								\
do {													\
	CUresult result = call;								\
	if (result != CUDA_SUCCESS) {						\
		std::stringstream ss;							\
		const char *msg;								\
		cuGetErrorName(result, &msg);                   \
		ss << "CUDA error in func " 					\
			<< __FUNCTION__ 							\
			<< " at line "								\
			<< __LINE__									\
			<< " calling " #call " failed with error "  \
			<< msg;										\
		throw cuda_runtime_error(ss.str());				\
	}													\
} while (0)

#define NVRTC_SAFE_CALL(call)							\
do {                                                    \
    nvrtcResult result = call;                          \
    if (result != NVRTC_SUCCESS) {                      \
		std::stringstream ss;							\
		ss << "CUDA NVRTC error in func " 				\
			<< __FUNCTION__ 							\
			<< " at line "								\
			<< __LINE__									\
			<< " calling " #call " failed with error "  \
            << nvrtcGetErrorString(result) << '\n';     \
		throw cuda_runtime_error(ss.str());				\
    }                                                   \
} while(0)
