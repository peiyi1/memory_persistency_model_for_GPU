#ifndef NVM_UTIL_H
#define NVM_UTIL_H

#include <sys/time.h>
#include <stdio.h>

#define ST_WT_U16(add, val) asm("st.global.wt.u16 [%0], %1;" :: "l"(add), "h"(val) : "memory")
#define ST_WT_INT(add, val) asm("st.global.wt.s32 [%0], %1;" :: "l"(add), "r"(val) : "memory")
#define ST_WB_INT(add, val) asm("st.global.s32 [%0], %1;" :: "l"(add), "r"(val) : "memory")
#define ST_WT_U64(add, val) asm("st.global.wt.u64 [%0], %1;" :: "l"(add), "r"(val) : "memory")
#define ST_WB_FLOAT(add, val) asm("st.global.f32 [%0], %1;" :: "l"(add), "f"(val) : "memory")
#define ST_WT_FLOAT(add, val) asm("st.global.wt.f32 [%0], %1;" :: "l"(add), "f"(val) : "memory")
#define UCHAR4_TO_UINT(uc4) (uc4).w << 24 | (uc4).z << 16 | (uc4.y) << 8 | (uc4).x
#define	SFENCE asm("membar.gl;")
#define	MEM_FENCE SFENCE
#define CLWB(add) asm("st.global.u32.cs [%0], %1;" :: "l"(add), "r"(0u) : "memory")
#define CLWB16(add) asm("st.global.u16.cs [%0], %1;" :: "l"(add), "h"((short)0) : "memory")
#define CLFLUSH(add) CLWB(add)
//#define PCOMMIT asm("st.global.f32.cg [%0], %1;" :: "l"(0l), "f"(.0f) : "memory")
#define PCOMMIT asm("st.global.f32.wb [%0], %1;" :: "l"(0l), "f"(.0f) : "memory")
#define L2WB_ASM asm("st.global.f32.cg [%0], %1;" :: "l"(0l), "f"(.0f) : "memory")

#define FLAT_THREAD_ID (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x)
#define FLAT_CTA_ID (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x)
#define SET_NVM_FLAG(val) \
  if (FLAT_THREAD_ID == 0) { ST_WT_INT(&NVM_flag[FLAT_CTA_ID], (val)); MEM_FENCE; }
#define SET_NVM_FLAG_WT(val) SET_NVM_FLAG(val)
#define SET_NVM_FLAG_WB(val) \
  if (FLAT_THREAD_ID == 0) { ST_WB_INT(&NVM_flag[FLAT_CTA_ID], (val)); CLWB(&NVM_flag[FLAT_CTA_ID]);  MEM_FENCE; }
#define SET_NVM_FLAG_WB_PC(val) \
  if (FLAT_THREAD_ID == 0) { ST_WB_INT(&NVM_flag[FLAT_CTA_ID], (val)); CLWB(&NVM_flag[FLAT_CTA_ID]);  MEM_FENCE; PCOMMIT; MEM_FENCE; }

#define L2WB __syncthreads(); if (FLAT_THREAD_ID == 0) L2WB_ASM;
#define CACHEWB L2WB

extern char nvm_opt;

#define CUDA_CKECK_ERR                                                      \
  {cudaError_t err;                                                     \
    if ((err = cudaGetLastError()) != cudaSuccess) {                    \
      fprintf(stderr, "CUDA error in file %s line %d: %s\n", __FILE__,  __LINE__, cudaGetErrorString(err)); \
      exit(-1);                                                         \
    }                                                                   \
  }

#define LOG_SIZE_16M (1024*1024*16) // 16M
#define LOG_SIZE_1G (1024*1024*1024) // 16G
#define FLAG_SIZE_1M (1024*1024)

#define DIM3_NTBS(d3) ((d3).x * (d3).y * (d3).z)

#define NVM_KLOG_ALLOC(ptr) cudaMalloc((void**)(ptr), LOG_SIZE_1G); CUDA_CKECK_ERR;
#define NVM_KLOG_FILL(dst, src, size) cudaMemcpy((dst), (src), (size), cudaMemcpyDeviceToDevice);
#define CHECK_BARRIER CUDA_CKECK_ERR; cudaDeviceSynchronize();

class memCopySim {
 private:
  unsigned long long m_size;
  unsigned int m_invokes;
  static const unsigned INVOKE_OVERHEAD = 1000;
  //static const float BANDWIDTH = 320; // GB/s
  //static const float FREQUENCY = 1.8; // GHz
  static const float BPC = 320/1.8; // bytes per cycle
  bool m_printed;
  
 public:
  memCopySim() {
    m_size = 0;
    m_invokes = 0;
    m_printed = false;
  }
  void log_copy(unsigned long long size) { m_size += size*2; m_invokes++; }
  void print_cycles(const char *msg) {
    if (m_printed) return;
    unsigned cycles = INVOKE_OVERHEAD + m_size / BPC;
    printf("\n%s memcopy overhead: %u cycles, bytes: %llu\n", msg, cycles, m_size/2);
    m_printed = true;
  }
};

class syncLapTimer {
 private:
  struct timeval st, et;
  unsigned long long total_us;
  unsigned nlaps;
 public:
  syncLapTimer() {
    nlaps = 0; total_us = 0; 
  }
  syncLapTimer(const syncLapTimer &st) { nlaps = st.nlaps; total_us = st.total_us; }
  void lap_start() { 
    cudaThreadSynchronize(); 
    gettimeofday(&st, NULL); 
  }
  void lap_end() { 
    cudaThreadSynchronize();
    gettimeofday(&et, NULL);
    nlaps++;
    total_us += (et.tv_sec * 1000000 + et.tv_usec) - (st.tv_sec * 1000000 + st.tv_usec);
    //printf("lap_end: nlaps %d\n", nlaps);
  }
  void print_total_us(const char *msg) {
    printf("\n%s: %u laps, %lld us\n\n", msg, nlaps, total_us);
  }
  void print_total_sec(const char *msg) {
    printf("\n%s: %u laps, %lf s\n\n", msg, nlaps, (double)total_us / (double)1000000);
  }
  void print_avg_usec(const char *msg) {
    printf("\n%s: %u laps, %lf us/lap\n\n", msg, nlaps, (double)total_us / (double)nlaps);
  }
  void print_avg_usec(const char *msg, unsigned ntbs) {
    printf("\n%s: %u laps, %lf us/lap, %lf us/cta\n\n", msg, nlaps, (double)total_us / (double)nlaps, (double)total_us / (double)nlaps / (double)ntbs);
  }
  void print_avg_usec(const char *msg, dim3 grid) {
    printf("\n%s: %u laps, %lf us/lap, %lf us/cta\n\n", msg, nlaps, (double)total_us / (double)nlaps, (double)total_us / (double)nlaps / (double)DIM3_NTBS(grid));
  }
};

#endif

