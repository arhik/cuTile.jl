# 1. Introduction — Tile IR

[Skip to main content](#main-content)

Back to top Ctrl+K

[![Tile IR - Home](../_static/nvidia-logo-horiz-rgb-blk-for-screen.svg) ![Tile IR - Home](../_static/nvidia-logo-horiz-rgb-wht-for-screen.svg)

Tile IR](../index.html)

Choose version

Search Ctrl+K

Search Ctrl+K

[![Tile IR - Home](../_static/nvidia-logo-horiz-rgb-blk-for-screen.svg) ![Tile IR - Home](../_static/nvidia-logo-horiz-rgb-wht-for-screen.svg)

Tile IR](../index.html)

Choose version

Table of Contents

-   [1. Introduction](#)
-   [2. Programming Model](prog_model.html)
-   [3. Syntax](syntax.html)
-   [4. Binary Format](bytecode.html)
-   [5. Type System](types.html)
-   [6. Semantics](semantics.html)
-   [7. Memory Model](memory_model.html)
-   [8. Operations](operations.html)
-   [9. Debug Info](debug_info.html)
-   [10. Stability](stability.html)
-   [11. Appendix](appendix.html)
-   [12. Release Notes](release_notes.html)

-   [](../index.html)
-   1. Introduction

# 1. Introduction[#](#introduction "Link to this heading")

This document describes **Tile IR**, a portable, low-level tile virtual machine and instruction set.

Unlike PTX, which models the GPU as a data-parallel single instruction multiple thread (SIMT) processor, **Tile IR** models the GPU as a tile-based processor. In **Tile IR**, each logical thread (tile block) computes over partial fragments (tiles) of multi-dimensional arrays (tensors).

## 1.1. Scalable Data Parallel Computing with Tiles on GPU[#](#scalable-data-parallel-computing-with-tiles-on-gpu "Link to this heading")

The CUDA platform provides portability through CUDA C++ and PTX, allowing code written for older GPU generations to run efficiently on newer ones. These mechanisms enable developers to write software for one family of GPUs and continue to deploy it on future generations.

The rapid evolution of hardware features, such as tensor cores, has increased programming-model complexity, requiring greater expertise and hardware understanding to write portable and performant code. Since Volta, each new GPU generation introduces powerful new hardware features, giving rise to new programming paradigms.

To address these challenges, **Tile IR** introduces a virtual instruction set that enables native programming of the hardware in terms of tiles, allowing developers to write higher-level code that can be efficiently executed across multiple GPUs with minimal changes.

While PTX ensures portability for SIMT programs, **Tile IR** extends the CUDA platform with native support for tile-based programs, allowing developers to focus on partitioning their data-parallel programs into tiles and tile blocks, letting **Tile IR** handle the mapping onto hardware resources such as threads, the memory hierarchy, and tensor cores.

By raising the level of abstraction, **Tile IR** enables users to build new higher-level hardware-specific DSLs, compilers, and frameworks for NVIDIA hardware.

## 1.2. Goals and Scope[#](#goals-and-scope "Link to this heading")

**Tile IR** aims to provide a performance-portable programming model and instruction set for tile-based parallel programming.

Core goals:

-   Introduce a data-parallel tile programming abstraction that aligns with programmer intent and acts as a code generation target for DSLs and compilers.
    
-   Abstract tensor-cores and their programming model to enable hardware innovation without disrupting the NVIDIA ecosystem or customers' existing software investments.
    
-   Abstract low-level, architecture-specific details such as CUDA threads and memory hierarchy to ensure programs can be efficiently recompiled for different GPUs.
    
-   Minimize abstraction overhead. While portability has a cost, **Tile IR** aims to keep this cost low, introducing only a modest performance overhead.
    
-   Provide user controls and optimization hints to recover peak performance when needed.
    
-   Provide seamless interoperability with existing CUDA programming models like CUDA C++ and PTX.
    

**Tile IR** achieves these core goals through the following key components, in order of importance:

1.  A versioned specification of the **Tile IR** abstract machine, including a versioned and portable bytecode representation.
    
2.  An optimizing compiler for **Tile IR** programs, available as part of the CUDA driver and as a standalone tool in the CUDA toolkit, similar to PTX.
    
3.  An MLIR dialect that existing compilers can use to target **Tile IR** as a backend compiler target.
    

## 1.3. Document Structure[#](#document-structure "Link to this heading")

1.  [Introduction](#section-introduction) this section.
    
2.  [Programming Model](prog_model.html#section-prog-model) describes the programming model of **Tile IR**.
    
3.  [Syntax](syntax.html#section-syntax) defines the syntax of **Tile IR** programs used throughout the specification.
    
4.  [Binary Format](bytecode.html#section-bytecode) describes the **Tile IR** byte code and its binary format.
    
5.  [Type System](types.html#section-types) defines the type system of **Tile IR**.
    
6.  [Semantics](semantics.html#section-semantics) describes the semi-formal semantics of **Tile IR**.
    
7.  [Memory Model](memory_model.html#section-memory-model) describes the **Tile IR** memory model.
    
8.  [Operations](operations.html#section-operations) a listing of the full set of **Tile IR** operations.
    
9.  [Debug Info](debug_info.html#section-debug-info) describes the debug information format for **Tile IR** programs.
    
10.  [Stability](stability.html#section-stability) describes the stability guarantees of the **Tile IR**.
    
11.  [Appendix](appendix.html#section-appendix) contains relevant reference materials, including full programs and references.
    

[

previous

Tile IR



](../index.html "previous page")[

next

2. Programming Model

](prog_model.html "next page")

On this page

-   [1.1. Scalable Data Parallel Computing with Tiles on GPU](#scalable-data-parallel-computing-with-tiles-on-gpu)
-   [1.2. Goals and Scope](#goals-and-scope)
-   [1.3. Document Structure](#document-structure)

[![NVIDIA](../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg) ![NVIDIA](../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

[Privacy Policy](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/) | [Your Privacy Choices](https://www.nvidia.com/en-us/about-nvidia/privacy-center/) | [Terms of Service](https://www.nvidia.com/en-us/about-nvidia/terms-of-service/) | [Accessibility](https://www.nvidia.com/en-us/about-nvidia/accessibility/) | [Corporate Policies](https://www.nvidia.com/en-us/about-nvidia/company-policies/) | [Product Security](https://www.nvidia.com/en-us/about-nvidia/product-security/) | [Contact](https://www.nvidia.com/en-us/contact/)

Copyright © 2025, NVIDIA.