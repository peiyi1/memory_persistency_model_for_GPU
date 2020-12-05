# Memory Persistency Models for GPUs

# Purpose

The goal of this project is to implements the paper Exploring Memory Persistency Models for GPUs by Zhen Lin, Mohammad Alshboul, Yan Solihin, and Huiyang Zhou [1], and evaluates results using GPGPU-Sim 4.0 [2]. 

# Introduction

The paper [1] proposes strict persistency and relaxed (epoch) persistency models for GPUs. And for the epoch persistency model, three epoch granularities: kernel level, CTA level and loop level are proposed based on the characteristics of a kernel. Finally, the paper [1] evaluates results using GPGPU-Sim simulator version 3.2.2 [3]. 

Compared to the paper [1], this work evaluates results using the new version of GPGPU-Sim 4.0 instead of using the old version 3.2.2. And in order to evaluate results on GPGPU-Sim 4.0, GPGPU-Sim 4.0 is modified to support CLWB, L2WB, PCOMMIT instructions which would be used in memory persistency models.

# Benchmark

Parboil benchmark [4] was used in the experiments. And the benchmark has been modified to implement the strict persistency and relaxed (epoch) persistency models. The implementation is in the location of parboil/benchmark/src. And the results of running the benchmark are in the location of parboil/result. And the detailed statistical data can be seen in the file named Memory_Persistency_Models_for_GPUs.pdf.

# References
[1]. Z. Lin, M. Alshboul, Y. Solihin and H. Zhou, "Exploring memory persistency models for gpus", 2019 28th International Conference on Parallel Architectures and Compilation Techniques (PACT), pp. 311-323, 2019.

[2]. Mahmoud Khairy, Zhesheng Shen, Tor M. Aamodt, Timothy G Rogers. Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling. In proceedings of the 47th IEEE/ACM International Symposium on Computer Architecture (ISCA), May 29 - June 3, 2020.

[3]. A. Bakhoda et al. Analyzing cuda workloads using a detailed gpu simulator. ISPASS-2009.

[4]. J. A. Stratton et al. Parboil: A Revised Benchmark Suite for Scientific and Commercial Throughput Computing. UIUC, Tech. Rep. IMPACT-12-01, March 2012
