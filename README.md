# CUDA Convolution Inference

This repo is a CUDA programming portfolio project for CNN inference acceleration.

This implementation was the second-fastest in ECE408 CUDA programming competition.

## What This Repo Demonstrates

- High-performance CUDA kernel design for convolution-heavy workloads.
- Practical performance engineering techniques for GPU compute.
- Host-side CUDA orchestration: allocation, transfer, launch, sync, and output collection.

## CUDA Optimization Techniques

1. Kernel fusion: reduce launch overhead and global-memory round trips.
2. Shared-memory tiling: increase data reuse and reduce DRAM traffic.
3. Tensor Core (WMMA) compute: accelerate GEMM-like convolution workloads.
4. Constant memory lookup: speed repeated small read-only index accesses.
5. `__restrict__` pointers: improve compiler optimization opportunities.
6. Loop unrolling: reduce loop/control overhead.
7. Index reuse/flattening: reduce arithmetic and address-generation cost.
8. Per-layer kernel tuning: choose launch geometry by layer shape.

## Core Files 

- `cuda_conv_kernel.cu`
- `cuda_conv_interface.h`
- `cuda_conv_host_demo.cu`

## Workflow Diagram

```mermaid
flowchart LR
  A[Host Input Buffer] --> B[H2D Copy and Device Alloc]
  B --> C[Fused Conv CUDA Kernel]
  C --> D[Device Sync]
  D --> E[D2H Copy and Output Checksum]
```

## Quick Compile Demo

```bash
nvcc -O3 -arch=sm_86 cuda_conv_host_demo.cu cuda_conv_kernel.cu -o cuda_conv_demo
./cuda_conv_demo
```


## Archive

Full pipeline code, build wiring, milestones, and legacy assets are preserved under `archive/`.
