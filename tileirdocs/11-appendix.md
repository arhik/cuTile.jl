# 11. Appendix — Tile IR

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
-   [7. Memory Model](memory_model.html)
-   [8. Operations](operations.html)
-   [9. Debug Info](debug_info.html)
-   [10. Stability](stability.html)
-   [11. Appendix](#)
-   [12. Release Notes](release_notes.html)

-   [](../index.html)
-   11. Appendix

# 11. Appendix[#](#appendix "Link to this heading")

## 11.1. Programming Model Example Programs[#](#programming-model-example-programs "Link to this heading")

### 11.1.1. Hello Tile Block[#](#hello-tile-block "Link to this heading")

cuda_tile.module @hello_world_module {
    entry @hello_world_kernel() {
        print "Hello World!\n"
    }
}

### 11.1.2. Vector Addition Block 128x1[#](#vector-addition-block-128x1 "Link to this heading")

// A basic implementation of 128 sized vector addition using unstructured load/stores.
//
// This implements addition over a 1-d tensor (vector) with size 128.
//
// 128x1 + 128x1 => 128x1
cuda_tile.module @vector_block_add_128x1 {
    entry @vector_block_add_128x1_kernel(
        %a_ptr_base_scalar : !cuda_tile.tile<ptr<f32>>,
        %b_ptr_base_scalar : !cuda_tile.tile<ptr<f32>>,
        %c_ptr_base_scalar : !cuda_tile.tile<ptr<f32>>)
{
    // Create an offset on the inclusive (0, 127) interval.
    %offset = iota : tile<128xi32>
    // We need a tile<ptr<T>> in order to perform a load or store.
    //
    // We will now convert each raw base pointer into such a pointer.
    //
    // First reshape the scalar pointer ptr<f32> to tile<1xptr<f32>> so it has the correct rank.
    %a_ptr_base_tensor = reshape %a_ptr_base_scalar :
        tile<ptr<f32>> -> tile<1xptr<f32>>
    // Next broadcast the pointer so we have a tensor of (base, ..., base) containing 128 elements.
    %a_ptr = broadcast %a_ptr_base_tensor : tile<1xptr<f32>> -> tile<128xptr<f32>>
    // Finally add the offset tensor to the tensor of pointers to obtain a tile<128xptr<f32>> that contains
    // pointers of (base + 0, ..., base + 127) as its values.
    %a_tensor = offset %a_ptr, %offset :
        tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>

    // Now we do the same for B.
    %b_ptr_base_tensor =reshape %b_ptr_base_scalar :
        tile<ptr<f32>> -> tile<1xptr<f32>>
    %b_ptr = broadcast %b_ptr_base_tensor : tile<1xptr<f32>> -> tile<128xptr<f32>>
    %b_tensor = offset %b_ptr, %offset :
        tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>

    // And the same for C.
    %c_ptr_base_tensor = reshape %c_ptr_base_scalar :
        tile<ptr<f32>> -> tile<1xptr<f32>>
    %c_ptr = broadcast %c_ptr_base_tensor : tile<1xptr<f32>> -> tile<128xptr<f32>>
    %c_tensor = offset %c_ptr, %offset :
        tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>

    // Now that we have prepared all the pointers we can do the real work.
    //
    // First we load A, and B into %a_val and %b_val.
    %a_val, %token_a = load_ptr_tko weak %a_tensor : tile<128xptr<f32>> -> tile<128xf32>, token
    %b_val, %token_b = load_ptr_tko weak %b_tensor : tile<128xptr<f32>> -> tile<128xf32>, token
    // We then compute floating-point vector addition using addf
    %c_val = addf %a_val, %b_val rounding<nearest_even> : tile<128xf32>
    // Finally we store the result to C.
    store_ptr_tko weak %c_tensor, %c_val : tile<128xptr<f32>>, tile<128xf32> -> token
  }
}

### 11.1.3. Hello Tile Grid[#](#hello-tile-grid "Link to this heading")

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

(Note: Additional GEMM examples and operation examples are available in the full appendix section of the official documentation.)

[

previous

10. Stability



](stability.html "previous page")[

next

12. Release Notes

](release_notes.html "next page")

On this page

-   [11.1. Programming Model Example Programs](#programming-model-example-programs)
    -   [11.1.1. Hello Tile Block](#hello-tile-block)
    -   [11.1.2. Vector Addition Block 128x1](#vector-addition-block-128x1)
    -   [11.1.3. Hello Tile Grid](#hello-tile-grid)
-   [11.2. Operation Examples](#operation-examples)

[![NVIDIA](../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg) ![NVIDIA](../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

[Privacy Policy](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/) | [Your Privacy Choices](https://www.nvidia.com/en-us/about-nvidia/privacy-center/) | [Terms of Service](https://www.nvidia.com/en-us/about-nvidia/terms-of-service/) | [Accessibility](https://www.nvidia.com/en-us/about-nvidia/accessibility/) | [Corporate Policies](https://www.nvidia.com/en-us/about-nvidia/company-policies/) | [Product Security](https://www.nvidia.com/en-us/about-nvidia/product-security/) | [Contact](https://www.nvidia.com/en-us/contact/)

Copyright © 2025, NVIDIA.