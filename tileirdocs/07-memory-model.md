# 7. Memory Model — Tile IR

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

-   [1. Introduction](introduction.html)
-   [2. Programming Model](prog_model.html)
-   [3. Syntax](syntax.html)
-   [4. Binary Format](bytecode.html)
-   [5. Type System](types.html)
-   [6. Semantics](semantics.html)
-   [7. Memory Model](#)
-   [8. Operations](operations.html)
-   [9. Debug Info](debug_info.html)
-   [10. Stability](stability.html)
-   [11. Appendix](appendix.html)
-   [12. Release Notes](release_notes.html)

-   [](../index.html)
-   7. Memory Model

# 7. Memory Model[#](#memory-model "Link to this heading")

The memory model defines the legal values that loads can return from memory. This is not as straightforward as one might expect at first glance, to enable compiler and hardware optimizations, we allow the apparent re-ordering of instructions.

This memory model is derived from the PTX memory model, and synchronization primitives are deliberately similar.

## 7.1. Memory model operations[#](#memory-model-operations "Link to this heading")

The memory model is built of relations between individual element accesses of tile operations, and restrictions on cycles of those relations. Therefore, a **Tile IR** memory instruction generates one or more memory model operations. In particular, tile loads, stores, and atomic updates generate one memory operation per element in the tile. When expanding a tile operation into many memory model operations, one might ask what order the memory model operations happen in. That is deliberately left unspecified for implementation flexibility.

To generalize the memory model over **Tile IR** operations, we use the terminology of *memory model operations* throughout the specification of the memory consistency model. We enumerate which **Tile IR** operations expand into which memory model operations in the table below.

## 7.2. Scopes[#](#scopes "Link to this heading")

Memory operations in **Tile IR** may have a scope. Operations without a scope are called `weak`. All memory operations specify a scope or `weak`. Any scope other than weak requires a memory ordering to be set.

Scope

Description

`tile_block`

Tile block scope, for communication within a single tile block.

`device`

Device scope, for communication within the same GPU.

`sys`

System scope, for communication anywhere in the system.

Weak operations cannot be used to communicate through memory between threads, or between fragments of the same tile block which are not ordered by token order. The compiler may assume that tiles accessed with `weak` are not concurrently accessed by any other thread.

Note

Tile Block scope is needed when building communicating algorithms where there is communication within a single tile block. This is necessary when communicating memory through memory between operations which are not ordered by token order, or when storing to a tile with internal aliasing.

## 7.3. Memory ordering[#](#memory-ordering "Link to this heading")

Memory operations have a memory ordering parameter which controls how that operation can be used for synchronization. Synchronization through memory is a two-party process, which requires a releaser and an acquirer observing the same location. When a pair of memory accesses synchronize through memory it establishes a *happens before* relationship. See the happens before definition below.

Any ordering other than `weak` requires a scope to be set.

Memory ordering

Description

`weak`

No concurrent accesses to the source/destination location.

`relaxed`

There may be concurrent access to the location, but this access does not establish a happens-before relationship.

`release`

There may be concurrent access to the location. If this release is observed with an acquire operation, then happens before is established.

`acquire`

There may be concurrent accesses to the location. If this acquire observes a release operation, then happens before is established.

`acq_rel`

There may be concurrent accesses to the location. This has the effect of both a release and acquire operation.

## 7.4. Moral Strength[#](#moral-strength "Link to this heading")

Two accesses to the same location are morally strong if the operations are related in *restricted program order*, or each operation specifies a scope which includes the tile block executing the other operation.

## 7.5. Tokens and token order[#](#tokens-and-token-order "Link to this heading")

In **Tile IR** we have explicit annotation of dependencies between loads and stores for the token-ordered operations. **Tile IR** produces wide loads and stores of whole tiles of data, making efficient use of various resources of the GPU in parallel. We provide token ordered operations to explicitly inform the **Tile IR** toolchain that two operations may happen in parallel, and will not interfere with each other. There is a family of memory operations called token ordered operations which produce and consume tokens. Tokens are abstract values in the **Tile IR** language for building dependencies between memory operations within the same tile block thread. They have no concrete representation at runtime, cannot be compared, computed upon, or stored/loaded to/from memory.

Program dependencies (i.e. dependencies apparent from control flow, data dependency, or address dependency) **do not** provide ordering between two memory operations. Tokens must be used, even where the token ordering appears redundant with program dependencies. Program dependencies may be optimized away by the **Tile IR** toolchain, whereas token dependencies are not.

## 7.6. Base relations[#](#base-relations "Link to this heading")

**Coherence order**

There exists a partial transitive order that relates overlapping write operations, determined at runtime, called *coherence order*. Two overlapping write operations are related in *coherence order* if they are *morally strong* or if they are related in *happens before* order.

**Program order**

A memory operation \(A\) is program order before a memory operation \(B\) if the instruction that gave rise to \(A\) is before the instruction that gave rise to \(B\) in the program source.

**Waits-for order**

An operation \(A\) *waits-for* an operation \(B\) if: - an instruction \(I_A\) gave rise to \(A\), \(I_A\) produced a token \(t\), an instruction \(I_B\) gave rise to \(B\) and \(I_B\) depends upon the token \(t\); or - there is some operation \(C\) such that \(A\) *waits-for* \(C\) and \(C\) *waits-for* \(B\).

**Reads from**

An read operation \(R\) *reads-from* a write operation \(W\) when \(R\) and \(W\) access the same location and \(R\) reads the value written by \(W\).

**Read-modify-write order**

Read-modify-write atomics generate a pair of memory operations for each element location within a tile. Each pair of memory operations is related by *read-modify-write order*.

**Atomicity**

When an atomic operation \(A\) and a write \(W\) overlap and are *morally strong*, then the following two communications cannot both exist in the same execution:

-   \(A\) reads from a write \(W'\) that precedes \(W\) in *coherence order*.
    
-   \(A\) follows \(W\) in *coherence order*.
    

## 7.9. No-thin-air[#](#no-thin-air "Link to this heading")

It is not practical to specify a "no thin air" axiom without preventing useful compiler optimizations. We therefore say informally that the implementation will not provide values out of thin air to satisfy program executions.

## 7.10. Data Races[#](#data-races "Link to this heading")

Two accesses are said to *conflict* when they access the same location and at least one of them is a write.

Two conflicting memory accesses are said to be in a *data race* if they are not related in happens before and are they are not morally strong.

Programs with data races have **undefined behaviour**.

## 7.11. PTX interoperability[#](#ptx-interoperability "Link to this heading")

The axioms and relations of the **Tile IR** memory model are intended to be a strict weakening of the PTX memory model.

The **Tile IR** memory model is designed to allow communication with PTX threads. Release and acquire patterns in **Tile IR** will match up with acquire and release patterns in PTX to build PTX *causality* and **Tile IR** *happens before*.

The same is true for data races, data races between accesses in a **Tile IR** program and a PTX program will result in undefined behavior as if the data race were all in **Tile IR**.

## 7.12. Hazards and intuition in the **Tile IR** Memory Model[#](#hazards-and-intuition-in-the-cuda-tile-memory-model "Link to this heading")

### 7.12.1. Token ordered operations reading from the future[#](#token-ordered-operations-reading-from-the-future "Link to this heading")

Store buffering and load buffering behavior are visible when using token ordered operations without token constraints.

### 7.12.2. Race hazards within a single tile block thread[#](#race-hazards-within-a-single-tile-block-thread "Link to this heading")

The definition of data race relies on two operations being in happens before order. In many programming languages this is always true within a single thread, but this is not the case in **Tile IR**. In **Tile IR** you can construct a data race when there are two accesses to the same location within a single tile block store (because of internal overlap in the destination tile), or with two accesses within a tile block thread to the same location which are not ordered by token order. This motivates the need for a `tile_block` scope, and is a hazard to be aware of in the language.

### 7.12.3. Some intuition on how to use token ordering[#](#some-intuition-on-how-to-use-token-ordering "Link to this heading")

When you have memory accesses to tiles with no overlap between them, it is generally safe to not order these with respect to each other in token order. When you use a release operation, you need to token-order all memory events that must stay before the release to the release itself. Similarly, when an acquire operation is used, all memory events which must remain after the acquire need to be token ordered after the acquire operation itself.

To reiterate: a user of **Tile IR** cannot rely on program dependencies of any form other than token dependencies to enforce ordering within a Tile Block Thread. The **Tile IR** toolchain can remove dependencies through any possible complex reasoning, breaking dependencies which appear to be in the program syntax.

[

previous

6. Semantics



](semantics.html "previous page")[

next

8. Operations

](operations.html "next page")

On this page

-   [7.1. Memory model operations](#memory-model-operations)
-   [7.2. Scopes](#scopes)
-   [7.3. Memory ordering](#memory-ordering)
-   [7.4. Moral Strength](#moral-strength)
-   [7.5. Tokens and token order](#tokens-and-token-order)
-   [7.6. Base relations](#base-relations)
-   [7.9. No-thin-air](#no-thin-air)
-   [7.10. Data Races](#data-races)
-   [7.11. PTX interoperability](#ptx-interoperability)
-   [7.12. Hazards and intuition in the **Tile IR** Memory Model](#hazards-and-intuition-in-the-cuda-tile-memory-model)
    -   [7.12.1. Token ordered operations reading from the future](#token-ordered-operations-reading-from-the-future)
    -   [7.12.2. Race hazards within a single tile block thread](#race-hazards-within-a-single-tile-block-thread)
    -   [7.12.3. Some intuition on how to use token ordering](#some-intuition-on-how-to-use-token-ordering)

[![NVIDIA](../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg) ![NVIDIA](../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

[Privacy Policy](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/) | [Your Privacy Choices](https://www.nvidia.com/en-us/about-nvidia/privacy-center/) | [Terms of Service](https://www.nvidia.com/en-us/about-nvidia/terms-of-service/) | [Accessibility](https://www.nvidia.com/en-us/about-nvidia/accessibility/) | [Corporate Policies](https://www.nvidia.com/en-us/about-nvidia/company-policies/) | [Product Security](https://www.nvidia.com/en-us/about-nvidia/product-security/) | [Contact](https://www.nvidia.com/en-us/contact/)

Copyright © 2025, NVIDIA.