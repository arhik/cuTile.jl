# 2. Programming Model — Tile IR

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
-   [2. Programming Model](#)
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
-   2. Programming Model

# 2. Programming Model[#](#programming-model "Link to this heading")

**Tile IR** extends CUDA's low-level programming model with new abstractions that differ from what has previously existed in CUDA C++ or PTX.

This section introduces the programming model of **Tile IR** and familiarizes the reader with its core concepts and abstractions. We do this by working through a series of real programs, building up to a dynamic, high-performance implementation of GEMM that makes use of the all major features of **Tile IR**.

**Tile IR** has an expressive tile-based programming model. We introduce users to the various ways to represent tensor computation by progressively adapting the example programs to take better advantage of **Tile IR** features which help simplify programs and enable the compiler to provide performance portability. We start by first introducing concepts that readers may be familiar with from prior art.

## 2.1. Tile Kernels[#](#tile-kernels "Link to this heading")

**Tile IR** programs are referred to as *tile kernels*, which like CUDA C++ or PTX, are functions which run as N copies in parallel when invoked. The primary difference is the basic unit of execution: a *tile-block*, which expresses the computation performed by a single logical tile thread operating over a multi-dimensional tile of data.

During execution, each tile kernel is referred to as a tile kernel instance.

Below is a simple **Tile IR** kernel which prints "Hello World!".

cuda_tile.module @hello_world_module {
    entry @hello_world_kernel() {
        print "Hello World!\n"
    }
}

For those familiar with CUDA threads, it is important to note that **Tile IR**'s threads are different. Before we go any further into formalisms, it is essential we highlight those differences.

Tile kernels are the entry point of the program, executing as parallel instances of tile blocks.

### 2.1.1. What's different about tile programming?[#](#what-s-different-about-tile-programming "Link to this heading")

**Tile IR** is an extension to the CUDA programming model that enables first class support for tile programming. Tiled kernels express programs as a grid of logical tile threads that operate over tiles. The mapping of both the grid and individual tile threads to the underlying hardware's threads is abstracted away from the programming model and is handled by the compiler.

The SIMT programming model of NVIDIA's streaming multiprocessor (SM) is one in which threads operate over (relatively) small pieces of data and the user is responsible for dividing and scheduling the threads into the appropriate blocks to compute over the input data in an efficient manner. This model gives flexibility to programmers on how to map threads to data, or vice-versa. SIMT is the programming model exposed by CUDA and PTX and has served NVIDIA GPUs well since its introduction in 2006.

The rise in importance of deep learning has both introduced a greater regularity to user workloads and an ever increasing need to deliver performance for these workloads. As discussed in the [Introduction](introduction.html#section-introduction), this has led to new specialized hardware in the form of tensor cores.

Tensor cores introduce a new dimension to the SIMT programming model. Now, SM threads must cooperate with the tensor cores in order to reach peak performance. With each new generation of hardware the interplay between these two pieces of silicon has unlocked amazing new performance but with increasing programming complexity.

**Tile IR** has been built to aid in the implementation of high-performance algorithms that take full advantage of the underlying hardware's capabilities while mitigating the increase in programming complexity.

By abstracting thread-to-data mapping, **Tile IR** simplifies the use of specialized hardware like Tensor Cores compared to traditional SIMT models.

### 2.1.2. Kernel I/O[#](#kernel-i-o "Link to this heading")

To illustrate the design of **Tile IR** we will move from our simple hello world kernel to one which implements 1-d tensor (i.e., vector) addition with a fixed block size `128`. All examples presented in this section can be found in the [Programming Model Example Programs](appendix.html#section-appendix-sub-prog).

Tile kernels accept inputs and outputs as parameters; this is the only mechanism for consuming and producing data, so we start by defining the kernel parameters.

entry @vector_block_add_128x1_kernel(
    %a_ptr_base_scalar : !cuda_tile.tile<ptr<f32>>,
    %b_ptr_base_scalar : !cuda_tile.tile<ptr<f32>>,
    %c_ptr_base_scalar : !cuda_tile.tile<ptr<f32>>)

The above code fragment defines a kernel named `vector_block_add_128x1_kernel` which takes three arguments, representing the two input buffers `a` and `b`, and the output buffer `c`. The types of all the arguments are scalar pointers, which are represented as zero dimensional tensors containing a single pointer.

All values in **Tile IR** are either tensors, or tensor views (see [Tensor View](types.html#type-views)). A tensor is an n-dimensional rectangular array, described by its rank (number of dimensions), the shape (extent along each dimension), and its primitive element type. Tensors may have a rank of 0 or higher. Rank-0 tensors are scalars. The rank, dimensions, and element type are all part of the tensor type and are statically known. Tensor types are assigned to values which represent a logical view of a multidimensional array contained in global memory. Global memory is always accessed via tensors, and thus pointer arguments always point to CUDA device allocations in global device memory. Tile kernels do not have return values and thus omit a return type annotation (note that tile functions may have a return type, and will be discussed later).

An astute reader might now be wondering why we gave the inputs and outputs scalar pointer types instead of tensors types with statically known rank, shape, and stride. A common pattern in the **Tile IR** programming model is for kernels to take unstructured base pointers as parameters which can then be used to construct the required tensor. This flexibility gives rise to multiple ways to use pointers depending on the desired program behavior, but we will first focus on the most flexible representation by converting our base pointer into a tensor of arbitrary pointers. A tensor of arbitrary pointers is a flexible abstraction which allows you to perform scatter/gather style loads from a set of addresses all at once, and treat a set of addresses as a logical tile.

We must take a few steps to convert a base pointer `%a_ptr_base_scalar` into a tensor of pointers representing the `128x1` tile we want to compute on which contains addresses from `(base + 0, ..., base + 127)`

We start by creating an offset tensor which represents the inclusive `(0, 127)` interval. We use [cuda_tile.iota](operations.html#op-cuda-tile-iota) which constructs a range tensor that counts from `0` to `n - 1` forming an `n` sized vector.

    %offset = iota : tile<128xi32>

We then reshape from a scalar `ptr<f32>` to a 1-d tensor `1xptr<f32>` so we have the correct rank.

%a_ptr_base_tensor = reshape %a_ptr_base_scalar :
    tile<ptr<f32>> -> tile<1xptr<f32>>

We then broadcast the pointer so we have a 1-d tensor of `(base, ..., base)` containing 128 elements.

%a_ptr = broadcast %a_ptr_base_tensor : tile<1xptr<f32>> -> tile<128xptr<f32>>

Add the offset tensor to the tensor of pointers to obtain a `tile<128xptr<f32>>` that contains pointers of `(base + 0, ..., base + 127)` as its values. Now we have a tensor of pointers which represents the tile that we would like to compute on. We perform the same set of steps for `a`, `b`, and `c`.

%a_tensor = offset %a_ptr, %offset :
    tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>

Finally, we load both operands, perform the addition, and store to the output.

%a_val, %token_a = load_ptr_tko weak %a_tensor : tile<128xptr<f32>> -> tile<128xf32>, token
%b_val, %token_b = load_ptr_tko weak %b_tensor : tile<128xptr<f32>> -> tile<128xf32>, token
%c_val = addf %a_val, %b_val rounding<nearest_even> : tile<128xf32>
store_ptr_tko weak %c_tensor, %c_val : tile<128xptr<f32>>, tile<128xf32> -> token

We now have a complete kernel for a single tile-block that performs addition over 128 element vectors. As you can see this code is written from a single thread of control, but its level of parallelism will be determined by the compiler.

Kernels communicate with global memory via pointer arguments, which can be transformed into tensors using shape and stride manipulation for flexible data access.

## 2.2. Tile Grid[#](#tile-grid "Link to this heading")

So far we have only examined kernels which are written for a single tile block. **Tile IR** allows tile blocks to be grouped into a **tile grid**, similar to CUDA C++, enabling users to launch sets of tile blocks that execute in parallel. Tile kernels, as with PTX, are implicitly parameterized over the tile block coordinates (which can be queried via [cuda_tile.get_tile_block_id](operations.html#op-cuda-tile-get-tile-block-id)) and can be 1-d, 2-d, or 3-d.

When a tile kernel is launched the user specifies the grid size, which determines the number of tile blocks launched. The number of tile blocks launched is equal to the size of the grid. For example if we launch our previous example with a `(1, 1, 2)` grid, we will run an identical computation twice.

We can now look at an **improved** hello world program which shows off querying the grid size and coordinates.

cuda_tile.module @hello_world_module {
    // TileIR kernel function
    entry @hello_world_kernel() {
        // Step 1. Get the tile block ID
        %block_x_index, %block_y_index, %block_z_index = cuda_tile.get_tile_block_id : tile<i32>

        // Step 2. Get the tile block dimensions
        %block_dim_x, %block_dim_y, %block_dim_z = cuda_tile.get_num_tile_blocks : tile<i32>

        // Step 3. Print the tile block ID and dimensions. Each tile executes the 
        // following print statement and prints a single line.
        cuda_tile.print "Hello, I am tile <%, %, %> in a kernel with <%, %, %> tiles.\n",
            %block_x_index, %block_y_index, %block_z_index, %block_dim_x, %block_dim_y, %block_dim_z
            : tile<i32>, tile<i32>, tile<i32>,
              tile<i32>, tile<i32>, tile<i32>
        }
}

Each tile kernel can be launched with a 1-d, 2-d, or 3-d grid. Each tile block can query its position in the grid using [cuda_tile.get_tile_block_id](operations.html#op-cuda-tile-get-tile-block-id) and query the first, second, or third dimension index by varying the argument. Tile kernels can also observe the grid dimensions in a similar way using [cuda_tile.get_num_tile_blocks](operations.html#op-cuda-tile-get-num-tile-blocks).

If we use a grid that is `(1, 1, 2)` we will see two prints:

1      "Hello Tile-Grid. I am tile <1, 1, 1> in a kernel with <1, 1, 2> tiles."
2      "Hello Tile-Grid. I am tile <1, 1, 2> in a kernel with <1, 1, 2> tiles."

The tile grid organizes parallel execution of tile blocks, allowing kernels to scale across the problem size by querying their coordinates within the grid.

### 2.2.1. Implementing GEMM[#](#implementing-gemm "Link to this heading")

Now that we understand the basics concepts of the **Tile IR** programming model, we will introduce how to compute a 2-d GEMM for a single block, and then generalize it step by step to a full GEMM by utilizing the tile grid and introducing control flow and manual tiling.

This progression demonstrates how to compose basic tile operations into a complex, high-performance parallel algorithm.

### 2.2.2. GEMM with a Single Block[#](#gemm-with-a-single-block "Link to this heading")

Let's first start by naturally moving from a single static block vector addition to a single static square block matrix multiplication.

This example proceeds much as before:

    entry @gemm_block_64x64_kernel(
        %a_ptr_base_scalar: !cuda_tile.tile<!cuda_tile.ptr<f32>>,
        %b_ptr_base_scalar: !cuda_tile.tile<!cuda_tile.ptr<f32>>,
        %c_ptr_base_scalar: !cuda_tile.tile<!cuda_tile.ptr<f32>>

We start with a set of scalar pointers. To simplify this example, we assume that each of these pointers point to allocations which are at least 64 elements in length with stride equal to 1.

Then much like before we declare a square offset tensor.

    %offset_flat = iota : tile<4096xi32>
    %offset = reshape %offset_flat :
        tile<4096xi32> -> tile<64x64xi32>

We declare pointers to the underlying tiles the same way as before for A, B, C.

    %a_ptr_base_tensor = reshape %a_ptr_base_scalar :
        tile<ptr<f32>> -> tile<1x1xptr<f32>>
    %a_ptr = broadcast %a_ptr_base_tensor : tile<1x1xptr<f32>> -> tile<64x64xptr<f32>>
    %a_tensor = offset %a_ptr, %offset :
        tile<64x64xptr<f32>>, tile<64x64xi32> -> tile<64x64xptr<f32>>

The only difference here is that we are building square tiles in 2D instead of 1D.

We then load the input arguments, compute the MMA operation (see [cuda_tile.mmaf](operations.html#op-cuda-tile-mmaf) for floating-point and [cuda_tile.mmai](operations.html#op-cuda-tile-mmai) for integers) to compute the output tile, and then store it.

    // Load a single 64x64 matrix from the tile.
    %A_block, %token_a = load_ptr_tko weak %a_tensor :
        tile<64x64xptr<f32>> -> tile<64x64xf32>, token
    // Load a single 64x64 matrix from the tile.
    %B_block, %token_b = load_ptr_tko weak %b_tensor :
        tile<64x64xptr<f32>> -> tile<64x64xf32>, token
    %C_block = mmaf %A_block, %B_block, %init_accum: tile<64x64xf32>, tile<64x64xf32>, tile<64x64xf32>
    store_ptr_tko weak %c_tensor, %C_block :
        tile<64x64xptr<f32>>, tile<64x64xf32> -> token

If you are experienced in implementing matrix multiplication, you can see that this works well for a single `64x64` square multiplication, but what happens if we want to run the tile kernel over a large input problem size in parallel?

Basic matrix multiplication is achieved by constructing 2D tiles from pointers and performing matrix-multiply-accumulate (MMA) operations within a single block.

### 2.2.3. GEMM Block by Block[#](#gemm-block-by-block "Link to this heading")

Let us look at how to generalize this to work for a large matrix size – say `4096x4096`, where each tile block will compute a single output tile of the final matrix. Note that we are still working with the assumption that our problem shape is static and we will return to handling dynamically shaped inputs later on this section.

The kernel declaration looks identical to what we have been thus far taking scalar pointers to our inputs and outputs as arguments.

    entry @gemm_square_4096_tile_64x64_kernel(
        %a_ptr_base_scalar: tile<ptr<f32>>,
        %b_ptr_base_scalar: tile<ptr<f32>>,
        %c_ptr_base_scalar: tile<ptr<f32>>

In order to do full matrix multiplication over a large matrix we need to split the problem across multiple tile blocks using the tile grid. We will use a 2-d grid and as such each thread will need to first read its 2-d block coordinates.

    // Read Tile block id's.
    %block_x_index, %block_y_index, %block_z_index = get_tile_block_id : tile<i32>

The block coordinates determine which tile of the output matrix we will be computing on this tile block. The kernel is structured as a reduction over the K dimension of the matrix, producing individual but complete output tiles. It loop over the reduction dimension computing a matrix-multiply-accumulate (MMA) operation over each input tile that contributes to the output tile.

As we have been doing before the kernel begins by setting up the initial state. As this kernel is structured as a loop we will first start by setting up the loop state.

    %m_tile_size = constant <i32: 64> : tile<i32>
    %m_stride_factor = cuda_tile.constant <i32: 64> : tile<64x64xi32> // todo fix the line # and restore this %n_tile_size = cuda_tile.constant <i32: 64> : tile<i32>
    %k_tile_size = cuda_tile.constant <i32: 64> : tile<i32>

First we specify the tile sizes as constants. In this case because we are using square tiles of square matrices everything is equivalent to 64. Constants in **Tile IR** can be tensor valued, and in this case we create 0-d tensors containing a single scalar value.

    %range_start = cuda_tile.constant <i32: 0> : tile<i32>
    %range_step = cuda_tile.constant <i32: 1> : tile<i32>
    %init_accum = cuda_tile.constant <f32: 0.000000e+00> : tile<64x64xf32>

Next we initialize constants for the loop which we will talk about more in a moment, and zero initialize an accumulator value for the MMA. It is worth noting that the tensor here is a value, the **Tile IR** compiler will deal with allocation and execution.

    %tile_size_range = cuda_tile.iota : tile<64xi32>

We also create a shared range from `(0, 63)` using [cuda_tile.iota](operations.html#op-cuda-tile-iota) again for the reduction dimension.

    %a_tile_base = cuda_tile.muli %block_x_index, %k_tile_size : tile<i32>
    %a_tile_base_reshape = cuda_tile.reshape %a_tile_base : tile<i32> -> tile<1xi32>
    %a_tile_base_tensor = cuda_tile.broadcast %a_tile_base_reshape :
        tile<1xi32> -> tile<64xi32>
    %m_offsets_vec = cuda_tile.addi %a_tile_base_tensor, %tile_size_range : tile<64xi32>

We must first compute the tensor of offsets for A and B so that we can obtain a tensor of pointers for each in order to load them from memory.

Conceptually we start by computing the offsets of the M dimension using the below Python pseudo code:

m_offsets = block_x_index * k_tile_size + arange(0, k_tile_size)

This produces a vector starting from the "top-corner" of the tile at `(block_x_index * k_tile_size, block_x_index * k_tile_size + k_tile_size)`.

The striding of the A matrix is `(64, 1)`, meaning we can omit the extra multiplication by 1 for the K dimension as each value in the row of the offset matrix is sequential.

For this the offsets in the K dimension would be `k_offs = arange(0, k_tile_size)`, so we can reuse %tile_size_range below.

Again we can use the following Python pseudo code to compute the offset matrix:

a_tile = reshape(m_offsets, (64, 1)) * m_stride + reshape(k_offsets, (1, 64)) * k_stride

In a libraries like Python's NumPy this computation hides implicit broadcasting that must be expanded in **Tile IR**.

We will do this in two pieces by first computing the offsets along the `M` dimension, the relative offsets along the `K` dimension, and then adding them together to produce the correct tensor of offsets.

We broadcast the `m_offsets` into a matrix where each column is identical and scaled by stride.

%m_offsets_matrix = cuda_tile.reshape %m_offsets_vec :
    tile<64xi32> -> tile<64x1xi32>
%m_offsets_broadcast = cuda_tile.broadcast %m_offsets_matrix :
    tile<64x1xi32> -> tile<64x64xi32>
%m_offsets = cuda_tile.muli %m_offsets_broadcast, %m_stride_factor : tile<64x64xi32>

The matrix, which is now scaled by 64, would then contain these values:

   [[   0,    0,    0,  ...,   0,    0,    0],
[  64,   64,   64,  ...,   64,   64,   64],
[ 128,  128,  128,  ...,  128,  128,  128],
   ...,
[3904, 3904, 3904,  ..., 3904,  3904,  3904],
[3968, 3968, 3968,  ..., 3968,  3968,  3968],
[4032, 4032, 4032,  ..., 4032,  4032,  4032]]

We then broadcast the k_offsets into a matrix where row is identical and scaled by stride.

%ak_offsets_matrix = cuda_tile.reshape %tile_size_range :
     tile<64xi32> -> tile<1x64xi32>
%ak_offsets_broadcast = cuda_tile.broadcast %ak_offsets_matrix :
    tile<1x64xi32> -> tile<64x64xi32>
%ak_offsets = cuda_tile.muli %ak_offsets_broadcast, %m_stride_factor : tile<64x64xi32>

A is a row-major matrix (i.e the K dimension's stride is 1) so the K offsets contains sequential values.

   [[   0,    1,    2,  ...,   61,    62,    63],
[   0,    1,    2,  ...,   61,    62,    63],
[   0,    1,    2,  ...,   61,    62,    63],
   ...,
[   0,    1,    2,  ...,   61,    62,    63],
[   0,    1,    2,  ...,   61,    62,    63],
[   0,    1,    2,  ...,   61,    62,    63]]

We add the two of them together resulting in rows where the i-th`value of :code:`m_offsets` and each column contains the next 63 integers.

%a_tile_offsets = cuda_tile.addi %m_offsets, %ak_offsets : tile<64x64xi32>

   [[   0,    1,    2,  ...,   61,   62,   63],
[  64,   65,   66,  ...,  125,  126,  127],
[ 128,  129,  130,  ...,  189,  190,  191],
   ...,
[3904, 3905, 3906,  ..., 3965, 3966, 3967],
[3968, 3969, 3970,  ..., 4029, 4030, 4031],
[4032, 4033, 4034,  ..., 4093, 4094, 4095]]

Note that because this is the 0-th tile of the output matrix it is centered around (0, 0), the next tile will be centered around (64, 0), then (128, 0) and so on. The above computation will result in the 64 x 64 grid at that point based on the grid coordinates.

Now we must do the same for B.

n_offs = j * n_tile_size + torch.arange(0, n_tile_size)
b_tile = k_offs[:, None] * k_stride + n_offs[None, :] * n_stride

%b_tile_base = cuda_tile.muli %block_y_index, %k_tile_size : tile<i32>
%b_tile_base_reshape = cuda_tile.reshape %b_tile_base :
    tile<i32> -> tile<1xi32>
%b_tile_base_tensor = cuda_tile.broadcast %b_tile_base_reshape :
    tile<1xi32> -> tile<64xi32>
%n_offsets_vec = cuda_tile.addi %b_tile_base_tensor, %tile_size_range : tile<64xi32>

%bk_offsets_matrix = cuda_tile.reshape %tile_size_range : tile<64xi32> -> tile<64x1xi32>
%bk_offsets = cuda_tile.broadcast %bk_offsets_matrix : tile<64x1xi32> -> tile<64x64xi32>

%n_offsets_matrix = cuda_tile.reshape %n_offsets_vec : tile<64xi32> -> tile<1x64xi32>
%n_offsets_broadcast = cuda_tile.broadcast %n_offsets_matrix :  tile<1x64xi32> -> tile<64x64xi32>
%n_offsets = cuda_tile.muli %n_offsets_broadcast, %m_stride_factor : tile<64x64xi32>
%b_tile_offsets = cuda_tile.muli %bk_offsets, %n_offsets : tile<64x64xi32>

Now that we have computed the initial offsets for the pointers we simply convert the base pointers and add the offsets like before.

%a_ptr_base_tensor = cuda_tile.reshape %a_ptr_base_scalar :
    tile<ptr<f32>> -> tile<1x1xptr<f32>>
%a_ptr = cuda_tile.broadcast %a_ptr_base_tensor : tile<1x1xptr<f32>> -> tile<64x64xptr<f32>>
%a_tile_ptr = offset %a_ptr, %a_tile_offsets :
    tile<64x64xptr<f32>>, tile<64x64xi32> -> tile<64x64xptr<f32>>

Large matrix operations are decomposed into smaller tiles processed by a grid of blocks, requiring careful calculation of global memory offsets for each block.

### 2.2.4. Looping Over Tiles[#](#looping-over-tiles "Link to this heading")

Now after all that preparation we can perform the core computation of the kernel.

%C_tile, %a_ptr_final, %b_ptr_final = for %k in (%range_start to %k_tile_size, step %range_start) : tile<i32>
    iter_values(
        %acc_prev = %init_accum,
        %a_tile_ptr_prev = %a_tile_ptr,
        %b_tile_ptr_prev = %b_tile_ptr
    ) -> (tile<64x64xf32>, tile<64x64xptr<f32>>, tile<64x64xptr<f32>>)
{
    // Load a single 64x64 matrix from the tile.
    %A_tile, %token_a = load_ptr_tko weak %a_tile_ptr :
        tile<64x64xptr<f32>> -> tile<64x64xf32>, token

    // Load a single 64x64 matrix from the tile.
    %B_tile, %token_b = load_ptr_tko weak %b_tile_ptr :
        tile<64x64xptr<f32>> -> tile<64x64xf32>, token

    %C_tile_acc = mmaf %A_tile, %B_tile, %acc_prev: tile<64x64xf32>, tile<64x64xf32>, tile<64x64xf32>

    // Advance by K block size.
    %block_size = constant <i32: 64> : tile<64x64xi32>
    %a_tile_ptr_next = offset %a_tile_ptr_prev, %block_size
        : tile<64x64xptr<f32>>, tile<64x64xi32>
            -> tile<64x64xptr<f32>>
    %b_tile_ptr_next = offset %b_tile_ptr_prev, %block_size
        : tile<64x64xptr<f32>>, tile<64x64xi32>
            -> tile<64x64xptr<f32>>

    // Store the partial sum to the 64x64 accumulator.
    continue %C_tile_acc, %a_tile_ptr_next, %b_tile_ptr_next : tile<64x64xf32>, tile<64x64xptr<f32>>, tile<64x64xptr<f32>>
}

After completing the reduction over the K dimension we need to store the output tile to the C matrix.

We do the same as we did with A and B, but this time the computation is a little different as we can compute both the X and Y dimensions using only the block coordinates. To think about this intuitively the tiled "column" "rows" produce a single output tile.

cm_offsets = block_x_index * BLOCK_SIZE_M + arange(0, BLOCK_SIZE_M)
cn_offsets = pid_n * BLOCK_SIZE_N + arange(0, BLOCK_SIZE_N)
c_tile = c_ptr + stride_cm * reshape(cm_offsets, (64, 1)) + stride_cn * reshape(cn_offsets, (64, 1))

We omit the index computation for %c_tile_offset then add it to the base pointer like before.

%c_tile_offsets = muli %c_tile_x_offsets, %c_tile_y_offsets : tile<64x64xi32>
%c_ptr_base_tensor = reshape %c_ptr_base_scalar :
    tile<ptr<f32>> -> tile<1x1xptr<f32>>
%c_ptr = broadcast %c_ptr_base_tensor :
    tile<1x1xptr<f32>> -> tile<64x64xptr<f32>>
%c_tile_ptr = offset %c_ptr, %c_tile_offsets :
    tile<64x64xptr<f32>>, tile<64x64xi32> -> tile<64x64xptr<f32>>

We can then store the value just as we have done in all prior kernels.

store_ptr_tko weak %c_tile_ptr, %C_tile :
    tile<64x64xptr<f32>>, tile<64x64xf32> -> token

Structured control flow allows efficient iteration over reduction dimensions, accumulating results in registers before writing the final output tile to memory.

## 2.3. Structured Pointers[#](#structured-pointers "Link to this heading")

The previous examples have looked at how to define matrix multiplication using tensors of pointers and scatter/gather style loads and stores, which take arbitrary tensors of pointers to operate on. These operations give maximal expressivity to programmers but at the cost of potential performance.

These are valuable to understand; as we saw in the last section, you can build arbitrary tensors of pointers then load and store to them flexibly.

For example, if a user created a complete disjoint tensor of pointers, it is challenging for a human or a compiler to obtain meaningful performance from this program. In the worst case, each element will become a completely disjoint memory operation, preventing vectorized or tensorized operations, or code which makes use of cache or thread locality

This flexibility comes at a cost in both requiring more setup and manual computation to implement even relatively simple algorithms like a tiled GEMM, and the potential of being inefficient. Due to the fact that load/store take arbitrary tensors in the degenerate case it is quite possible that they get decomposed into a sequence of arbitrary loads and stores, which do not make good use of memory locality or the underlying hardware.

**Tile IR** will do its best to obtain good performance with these stores, but optimal performance can more easily be achieved by using a structured pointer, called a `tensor view` in **Tile IR**. tensor views allow us to simplify the programming model of **Tile IR** and to improve the efficiency of user programs.

As we have seen in our previous examples, when a kernel is given a tensor it is a scalar base pointer into global memory.

![A view of global memory](../_images/global_memory.svg)

The base pointer to A points to an allocation that lives in global memory.[#](#id1 "Link to this image")

By formulating our previous problem with the correct sizes, we completely side stepped the complex offset computation, and avoided dealing with imperfect tiling, or store/load masking. See XX for more details about masking.

When constructing a `tensor view` we convert the raw pointer into a tensor using static or dynamic shape and stride information.

In order to both simplify the programming model of **Tile IR** and to improve the efficiency of user programs **Tile IR** also has a structured pointer type known as a tensor view. A tensor view can be constructed from raw pointers via [cuda_tile.make_tensor_view](operations.html#op-cuda-tile-make-tensor-view) and attaches shape and stride information to the pointer and effectively represents a typed pointer to a tensor.

When constructing a tensor view, we convert the raw pointer into a tensor using static or dynamic shape and stride information, which is done via [cuda_tile.make_tensor_view](operations.html#op-cuda-tile-make-tensor-view). This op attaches shape and stride information to the pointer and effectively converts a typed pointer to a tensor.

![A view of global memory](../_images/tensor_view.svg)

The base pointer to A points to an allocation that lives in global memory.[#](#id2 "Link to this image")

Structured pointers, or tensor views, encapsulate shape and stride information, enabling the compiler to optimize memory access and simplifying the user's code.

## 2.4. Tiling & Views[#](#tiling-views "Link to this heading")

Once you have constructed a tensor view, we can make use of [cuda_tile.make_partition_view](operations.html#op-cuda-tile-make-partition-view) to perform tiling of the underlying tensor.

![A view of global memory](../_images/tiled.svg)

The base pointer to A points to an allocation that lives in global memory.[#](#id3 "Link to this image")

In the last section we simplified the problem by choosing tile dimensions that made the problem perfectly square, in the same layout, using simple tiling and static input dimensions, etc. We will relax those constraints to show more complex tile mappings as we explore views and partitions.

### 2.4.1. Vector Addition with Views[#](#vector-addition-with-views "Link to this heading")

We can now implement a more complex vector operation with SAXPY or "Single-Precision A·X Plus Y" (a common BLAS operation). We can use `tensor view` and [cuda_tile.make_partition_view](operations.html#op-cuda-tile-make-partition-view) to implement this operation for arbitrary sized vectors.

The kernel first defines its arguments. We now take the inputs `%X`, `%Y`, `%alpha`, as well as the dimensions of the **full** vectors, as arguments.

entry @saxpy_memref(%X: tile<ptr<f32>>,
                    %Y: tile<ptr<f32>>,
                    %alpha: tile<f32>,
                    %M : tile<i32>,
                    %N : tile<i32>) {

For those familiar with other tile programming models, you might wonder why we need to take the input tensor sizes as arguments instead of using them to control the number of blocks and remain implicit in the program.

The tensor view model differs from some existing tile programming models, wherein each kernel is unaware of the overall dimensions of the problem size beyond the need to mask loads and stores. In contrast, `tensor view`'s require the overall tensor dimensions in order to enable efficient and correct lowering of tiling and its related operations automatically.

%x_memref = make_tensor_view %X, shape = [%M, %N], strides = [%M, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[?,1]>
%y_memref = make_tensor_view %Y, shape = [%M, %N], strides = [%M, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[?,1]>

The tensor view's constructor takes the shapes and strides dynamically resulting in a dynamically shaped tensor view specified by ? in the type like so: `!cuda_tile.tile view<?x?xf32, strides=[?,1]>`.

We use [cuda_tile.make_partition_view](operations.html#op-cuda-tile-make-partition-view) to create views for `%x` and `%y` which represent a `(M/128 x N/256)` tensor of tiles. Each tile will have the size specified by tile parameter of the partition type.

%x_view = make_partition_view %x_memref : partition_view<tile=(128x256), tensor_view<?x?xf32, strides=[?,1]>>
%y_view = make_partition_view %y_memref : partition_view<tile=(128x256), tensor_view<?x?xf32, strides=[?,1]>>

[cuda_tile.load_view_tko](operations.html#op-cuda-tile-load-view-tko) allows us to load the tile specified by `%view[%x, %y]` for both %X and %Y.

%x_tile, %token_x = load_view_tko weak %x_view[%tileIdX, %tileIdY] :
    partition_view<tile=(128x256), tensor_view<?x?xf32, strides=[?,1]>>, tile<i32> -> tile<128x256xf32>, token

We can then simply compute using the tiles directly to obtain our result tile.

// Step 6. Compute sAXPY: y = alpha * A + y

Finally we accumulate the result into `%Y` directly. Here we treat it as both an input and output parameter in this kernel.

Combining tensor views with partitioning simplifies kernel implementation, allowing direct loading and operating on logical tiles without manual offset arithmetic.

### 2.4.2. Dynamic GEMM with Views[#](#dynamic-gemm-with-views "Link to this heading")

We have now introduced the core concepts of **Tile IR**. We can put together many of these ideas to support a dynamic GEMM kernel using structured pointers. To start, we make two changes to this GEMM: the inputs are actually transposed into column-major layout, and the inputs are in `fp16` while the output is in `fp32`.

We take the input and output tensors (as pointers), all dimension, and all strides as arguments.

    entry @gemm_kloop_kernel(
        %A_ptr: !cuda_tile.tile<!cuda_tile.ptr<f16>>,
        %B_ptr: !cuda_tile.tile<!cuda_tile.ptr<f16>>,
        %C_ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>,
        %M: !cuda_tile.tile<i32>, %N: !cuda_tile.tile<i32>, %K: !cuda_tile.tile<i32>,
        %stride_ak: !cuda_tile.tile<i32>, %stride_bn: !cuda_tile.tile<i32>, %stride_cm: !cuda_tile.tile<i32>

The **Tile IR** compiler can optimize the memory loads and stores of [Tensor View](types.html#type-tensor-view) if the alignment of the underlying pointers and strides are known. If these are statically known then we can infer the alignment directly but if they are dynamic, as they are in this case, we can use the [cuda_tile.assume](operations.html#op-cuda-tile-assume) operation to inform the compiler that the pointer is properly aligned.

Here we use the `div_by` predicate to inform the compiler about the divisibility of these values which can be used to infer alignment constraints directly.

%A_ptr_assume = assume #cuda_tile.div_by<16>, %A_ptr : tile<ptr<f16>>
%B_ptr_assume = assume #cuda_tile.div_by<16>, %B_ptr : tile<ptr<f16>>
%C_ptr_assume = assume #cuda_tile.div_by<16>, %C_ptr : tile<ptr<f32>>
%stride_ak_assume = assume #cuda_tile.div_by<8>, %stride_ak : tile<i32>
%stride_bn_assume = assume #cuda_tile.div_by<8>, %stride_bn : tile<i32>
%stride_cm_assume = assume #cuda_tile.div_by<8>, %stride_cm : tile<i32>

We create a tensor view for `%A`, `%B`, and `%C`.

// A reference to the A tensor pointed to by A_ptr, (K x M)
%A = make_tensor_view %A_ptr_assume, shape = [%K, %M], strides = [%stride_ak, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
// A reference to the B tensor pointed to by B_ptr, (N x K)
%B = make_tensor_view %B_ptr_assume, shape = [%N, %K], strides = [%stride_bn, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
// A reference to the C tensor pointed to by C_ptr, (M x N)
%C = make_tensor_view %C_ptr_assume, shape = [%M, %N], strides = [%stride_cm, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[?,1]>

We create a [Partition View](types.html#type-partition-view) for each tensor. First A:

%A_block  = make_partition_view %A : partition_view<tile=(128x64), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>

Then B:

%B_block  = make_partition_view %B : partition_view<tile=(64x128), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>

And last C:

%C_block  = make_partition_view %C : partition_view<tile=(128x128), tensor_view<?x?xf32, strides=[?,1]>, dim_map=[0, 1]>

    // Load a single 64x128 matrix from the tile.
    %B_frag, %t2 = load_view_tko weak %B_block [%k, %bidy] : partition_view<tile=(64x128), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>, tile<i32> -> tile<64x128xf16>, token

We then read the the tile block grid coordinates using [cuda_tile.get_tile_block_id](operations.html#op-cuda-tile-get-tile-block-id).

%bidx, %bidy, %bidz = get_tile_block_id : tile<i32>

Due to the `K` dimension being dynamic we could do calculations ourselves to compute the size of the index space of the reduction but **Tile IR** provides an operation to do this for us. [cuda_tile.get_index_space_shape](operations.html#op-cuda-tile-get-index-space-shape) to get the size of the index space, and thus the size of the reduction loop.

%mk_len_i32:2 = get_index_space_shape %A_block : partition_view<tile=(128x64), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]> -> tile<i32>

No we can use a [cuda_tile.for](operations.html#op-cuda-tile-for) loop to iterate over the tile blocks.

A `for` loop is one of the structured control flow operations in **Tile IR** it steps a loop variable over a range from `(start, end, step)` executing the body for each value in the range. The iter_values are loop carried variables of the body which are initialized in the loop header and are updated each iteration by yielding them with the [cuda_tile.continue](operations.html#op-cuda-tile-continue) operation.

In order to implement the reduction we start from `0` to the size of the tiled `K` dimension, stepping uniformly by `1` on each step and a single loop carried variable representing the accumulator tile.

%result = for %k in (%i0 to %mk_len_i32#1, step %i1) : tile<i32>
    iter_values(%acc_prev = %cst) -> (tile<128x128xf32>)

We load tiles from A and B.

// Load a single 128x64 matrix from the tile.
%A_frag, %t1 = load_view_tko weak %A_block[%bidx, %k] : partition_view<tile=(128x64), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>, tile<i32> -> tile<128x64xf16>, token
// Load a single 64x128 matrix from the tile.
%B_frag, %t2 = load_view_tko weak %B_block [%k, %bidy] : partition_view<tile=(64x128), tensor_view<?x?xf16, strides=[?,1]>, dim_map=[1, 0]>, tile<i32> -> tile<64x128xf16>, token

We compute the MMA, and then continue to the next loop iteration with the value.

    %acc = mmaf %A_frag, %B_frag, %acc_prev: tile<128x64xf16>, tile<64x128xf16>, tile<128x128xf32>
    continue %acc : tile<128x128xf32>

Finally, like our previous implementation, outside the loop we store to the tile back to `%C` this time via the view avoiding the need to compute the offsets again.

%t3 = store_view_tko weak %result, %C_block[%bidx, %bidy] : tile<128x128xf32>, partition_view<tile=(128x128), tensor_view<?x?xf32, strides=[?,1]>, dim_map=[0, 1]>, tile<i32> -> token

Tensor views support dynamic shapes and strides, enabling robust, flexible kernels that handle varying input sizes while maintaining high performance.

## 2.5. Cross TileBlock Communication[#](#cross-tileblock-communication "Link to this heading")

  cuda_tile.module @hello_cross_block {
    global @_global_printf_mutex <i32: 1> : tile<1xi32>

    entry @hello_cross_block_kernel() {
      %idx, %idy, %idz = get_tile_block_id : tile<i32>
      %tilex, %tiley, %tilez = get_num_tile_blocks : tile<i32>
      %2 = get_global @_global_printf_mutex : tile<ptr<i32>>
      %3 = cuda_tile.constant <i32: 0> : tile<i32>
      %4 = cuda_tile.constant <i32: 1> : tile<i32>
      loop {
        %t1 = make_token : token
        %6, %t2 = atomic_cas_tko relaxed device %2, %4, %3 token=%t1: tile<!cuda_tile.ptr<i32>>, tile<i32> -> tile<i32>, !cuda_tile.token
        %7 = trunci %6 : tile<i32> -> tile<i1>
        if %7 {
          break
        }
      }
      print "current tile: %i / %i\0A", %idx, %tilex : tile<i32>, tile<i32>
      %5, %t4 = atomic_rmw_tko relaxed device %2, xchg, %4: tile<ptr<i32>>, tile<i32> -> tile<i32>, !cuda_tile.token
      return
    }
  }

[

previous

1. Introduction



](introduction.html "previous page")[

next

3. Syntax

](syntax.html "next page")

On this page

-   [2.1. Tile Kernels](#tile-kernels)
    -   [2.1.1. What's different about tile programming?](#what-s-different-about-tile-programming)
    -   [2.1.2. Kernel I/O](#kernel-i-o)
-   [2.2. Tile Grid](#tile-grid)
    -   [2.2.1. Implementing GEMM](#implementing-gemm)
    -   [2.2.2. GEMM with a Single Block](#gemm-with-a-single-block)
    -   [2.2.3. GEMM Block by Block](#gemm-block-by-block)
    -   [2.2.4. Looping Over Tiles](#looping-over-tiles)
-   [2.3. Structured Pointers](#structured-pointers)
-   [2.4. Tiling & Views](#tiling-views)
    -   [2.4.1. Vector Addition with Views](#vector-addition-with-views)
    -   [2.4.2. Dynamic GEMM with Views](#dynamic-gemm-with-views)
-   [2.5. Cross TileBlock Communication](#cross-tileblock-communication)

[![NVIDIA](../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg) ![NVIDIA](../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

[Privacy Policy](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/) | [Your Privacy Choices](https://www.nvidia.com/en-us/about-nvidia/privacy-center/) | [Terms of Service](https://www.nvidia.com/en-us/about-nvidia/terms-of-service/) | [Accessibility](https://www.nvidia.com/en-us/about-nvidia/accessibility/) | [Corporate Policies](https://www.nvidia.com/en-us/about-nvidia/company-policies/) | [Product Security](https://www.nvidia.com/en-us/about-nvidia/product-security/) | [Contact](https://www.nvidia.com/en-us/contact/)

Copyright © 2025, NVIDIA.