
# CSDL (Chained Scan with Decoupled Lookback) Scan Example
#
# This example demonstrates parallel prefix sum using cuTile's scan operation.
# The CSDL algorithm is designed for efficient GPU execution with:
# - Single-phase operation (no global synchronization)
# - Chained lookback for progressive accumulation
# - O(1) amortized lookback complexity
#
# Hardware requirements:
# - GPU with sm_90+ architecture (Ada Lovelace, Hopper, or newer)
# - CUDA 12.0+
# - Julia with cuTile installed
#
# SPDX-License-Identifier: Apache-2.0

using CUDA
using cuTile
import cuTile as ct

#=============================================================================
# Basic Scan Kernels
#=============================================================================

"""
    scan_1d_cumsum(input, output, n, tile_h)

1D cumulative sum kernel using CSDL scan.
Processes `n` elements with tile size (32, tile_h).

# Arguments
- `input`: Input CuArray
- `output`: Output CuArray (pre-allocated)
- `n`: Number of elements
- `tile_h`: Tile height (partition length)
"""
function scan_1d_cumsum(input::CuArray{T,1}, output::CuArray{T,1},
                        n::Int, tile_h::Int) where T
    tile_w = 32  # Warp size for efficient intra-warp scan

    num_tiles = cld(n, tile_w * tile_h)
    ct.launch(num_tiles) do bid::Int
        offset = (bid - 1) * tile_w * tile_h
        actual_count = min(tile_w * tile_h, n - offset)
        load_shape = (tile_w, cld(actual_count, tile_w))

        base_idx = ct.Tile(Int32(offset))
        indices = ct.arange(load_shape, Int32) .+ base_idx

        max_idx = ct.Tile(Int32(n - 1))
        clamped_indices = ct.min(indices, max_idx)

        input_tile = ct.gather(input, clamped_indices)
        result = ct.cumsum(input_tile, Val(1))
        ct.scatter(output, clamped_indices, result)
    end

    return output
end

"""
    scan_2d_colwise(input, output, tile_w, tile_h)

2D column-wise cumulative sum kernel.
Scans along dimension 1 (columns) for each row independently.
"""
function scan_2d_colwise(input::CuArray{T,2}, output::CuArray{T,2},
                         tile_w::Int, tile_h::Int) where T
    m, n = size(input)

    num_tiles_x = cld(m, tile_w)
    num_tiles_y = cld(n, tile_h)

    ct.launch((num_tiles_x, num_tiles_y)) do bid_x::Int, bid_y::Int
        offset_x = (bid_x - 1) * tile_w
        offset_y = (bid_y - 1) * tile_h

        actual_w = min(tile_w, m - offset_x)
        actual_h = min(tile_h, n - offset_y)

        base_i = ct.Tile(Int32(offset_x))
        base_j = ct.Tile(Int32(offset_y))
        i_offsets = ct.arange((actual_w, actual_h), Int32) .+ base_i
        j_offsets = ct.broadcast_to(base_j, (actual_w, actual_h))
        indices = (i_offsets, j_offsets)

        input_tile = ct.gather(input, indices...)
        result = ct.cumsum(input_tile, Val(1))
        ct.scatter(output, indices..., result)
    end

    return output
end

"""
    scan_2d_rowwise(input, output, tile_w, tile_h)

2D row-wise cumulative sum kernel.
Scans along dimension 0 (rows) for each column independently.
"""
function scan_2d_rowwise(input::CuArray{T,2}, output::CuArray{T,2},
                         tile_w::Int, tile_h::Int) where T
    m, n = size(input)

    num_tiles_x = cld(m, tile_w)
    num_tiles_y = cld(n, tile_h)

    ct.launch((num_tiles_x, num_tiles_y)) do bid_x::Int, bid_y::Int
        offset_x = (bid_x - 1) * tile_w
        offset_y = (bid_y - 1) * tile_h

        actual_w = min(tile_w, m - offset_x)
        actual_h = min(tile_h, n - offset_y)

        base_i = ct.Tile(Int32(offset_x))
        base_j = ct.Tile(Int32(offset_y))
        i_offsets = ct.broadcast_to(base_i, (actual_w, actual_h))
        j_offsets = ct.arange((actual_w, actual_h), Int32) .+ base_j
        indices = (i_offsets, j_offsets)

        input_tile = ct.gather(input, indices...)
        result = ct.cumsum(input_tile, Val(0))
        ct.scatter(output, indices..., result)
    end

    return output
end

"""
    scan_1d_cumprod(input, output, n, tile_h)

Cumulative product kernel using scan with multiplication.
"""
function scan_1d_cumprod(input::CuArray{T,1}, output::CuArray{T,1},
                         n::Int, tile_h::Int) where T
    tile_w = 32

    num_tiles = cld(n, tile_w * tile_h)
    ct.launch(num_tiles) do bid::Int
        offset = (bid - 1) * tile_w * tile_h
        actual_count = min(tile_w * tile_h, n - offset)
        load_shape = (tile_w, cld(actual_count, tile_w))

        base = ct.Tile(Int32(offset))
        offsets = ct.arange(load_shape, Int32) .+ base
        max_idx = ct.Tile(Int32(n - 1))
        clamped = ct.min(offsets, max_idx)

        input_tile = ct.gather(input, clamped)
        result = ct.cumprod(input_tile, Val(1))
        ct.scatter(output, clamped, result)
    end

    return output
end

"""
    scan_1d_cumsum_reverse(input, output, n, tile_h)

Reverse cumulative sum (right-to-left scan).
Useful for suffix sums or trailing segment processing.
"""
function scan_1d_cumsum_reverse(input::CuArray{T,1}, output::CuArray{T,1},
                                 n::Int, tile_h::Int) where T
    tile_w = 32

    num_tiles = cld(n, tile_w * tile_h)
    ct.launch(num_tiles) do bid::Int
        offset = (bid - 1) * tile_w * tile_h
        actual_count = min(tile_w * tile_h, n - offset)
        load_shape = (tile_w, cld(actual_count, tile_w))

        base = ct.Tile(Int32(offset))
        offsets = ct.arange(load_shape, Int32) .+ base
        max_idx = ct.Tile(Int32(n - 1))
        clamped = ct.min(offsets, max_idx)

        input_tile = ct.gather(input, clamped)
        result = ct.cumsum(input_tile, Val(1), true)
        ct.scatter(output, clamped, result)
    end

    return output
end

#=============================================================================
# Verification Functions
#=============================================================================

function verify_scan(input::CuArray{T,1}, output::CuArray{T,1};
                     op::Symbol=:cumsum) where T
    cpu_input = Array(input)
    cpu_output = Array(output)

    if op == :cumsum
        expected = cumsum(cpu_input, dims=1)
    else
        expected = cumprod(cpu_input, dims=1)
    end

    @assert cpu_output ≈ expected "Verification failed for $op"
    println("  [PASS] $op verification passed")
end

function verify_scan(input::CuArray{T,2}, output::CuArray{T,2}, dim::Int;
                     op::Symbol=:cumsum) where T
    cpu_input = Array(input)
    cpu_output = Array(output)

    if op == :cumsum
        expected = cumsum(cpu_input, dims=dim)
    else
        expected = cumprod(cpu_input, dims=dim)
    end

    @assert cpu_output ≈ expected "Verification failed for $op (dim=$dim)"
    println("  [PASS] $op verification passed (dim=$dim)")
end

function benchmark_scan(kernel_func, args...; warmup::Int=3, iterations::Int=10)
    for _ in 1:warmup
        kernel_func(args...)
    end
    CUDA.synchronize()

    times = Float64[]
    for _ in 1:iterations
        CUDA.@elapsed begin
            kernel_func(args...)
        end |> push!(times)
    end
    CUDA.synchronize()

    mean_time = mean(times) * 1000
    std_time = std(times) * 1000
    min_time = minimum(times) * 1000
    max_time = maximum(times) * 1000

    println("  Mean: $(mean_time:.3f) ms (±$(std_time:.3f))")
    println("  Min:  $(min_time:.3f) ms")
    println("  Max:  $(max_time:.3f) ms")

    return (mean=mean_time, std=std_time, min=min_time, max=max_time)
end

#=============================================================================
# Test Functions
#=============================================================================

function test_1d_cumsum(::Type{T}, n::Int, tile_h::Int) where T
    name = "1D cumsum ($n elements, $T, tile_h=$tile_h)"
    println("--- $name ---")

    input = CUDA.rand(T, n)
    output = CUDA.zeros(T, n)

    benchmark_scan(scan_1d_cumsum, input, output, n, tile_h)
    verify_scan(input, output, op=:cumsum)
end

function test_1d_cumprod(::Type{T}, n::Int, tile_h::Int) where T
    name = "1D cumprod ($n elements, $T, tile_h=$tile_h)"
    println("--- $name ---")

    input = CUDA.rand(T, n) .+ T(0.1)
    output = CUDA.zeros(T, n)

    benchmark_scan(scan_1d_cumprod, input, output, n, tile_h)
    verify_scan(input, output, op=:cumprod)
end

function test_1d_cumsum_reverse(::Type{T}, n::Int, tile_h::Int) where T
    name = "1D cumsum reverse ($n elements, $T, tile_h=$tile_h)"
    println("--- $name ---")

    input = CUDA.rand(T, n)
    output = CUDA.zeros(T, n)

    benchmark_scan(scan_1d_cumsum_reverse, input, output, n, tile_h)

    cpu_input = Array(input)
    cpu_output = Array(output)
    expected = reverse(cumsum(reverse(cpu_input), dims=1))
    @assert cpu_output ≈ expected "Reverse cumsum verification failed"
    println("  [PASS] Reverse cumsum verification passed")
end

function test_2d_cumsum(::Type{T}, m::Int, n::Int, tile_w::Int, tile_h::Int) where T
    name = "2D cumsum ($m×$n, $T, tiles=$tile_w×$tile_h)"
    println("--- $name ---")

    input = CUDA.rand(T, m, n)
    output = CUDA.zeros(T, m, n)

    benchmark_scan(scan_2d_colwise, input, output, tile_w, tile_h)
    verify_scan(input, output, 1, op=:cumsum)
end

function test_large_cumsum(::Type{T}, n::Int) where T
    name = "Large 1D cumsum ($n elements, $T)"
    println("--- $name ---")

    tile_h = 1024
    input = CUDA.rand(T, n)
    output = CUDA.zeros(T, n)

    benchmark_scan(scan_1d_cumsum, input, output, n, tile_h)
    verify_scan(input, output, op=:cumsum)
end

#=============================================================================
# Main Entry Point
#=============================================================================

function main()
    println("="^70)
    println("cuTile CSDL Scan Examples")
    println("="^70)
    println()

    if !CUDA.functional()
        error("CUDA is not available. This example requires a CUDA-capable GPU.")
    end

    println("GPU: $(CUDA.name(CUDA.device()))")
    println("Compute Capability: $(CUDA.capability(Int))")
    println()

    if CUDA.capability(Int) < (9, 0)
        println("WARNING: Tile IR requires sm_90+ (Ampere or newer)")
        println("This GPU may not support Tile IR execution.")
        println()
    end

    println("--- 1D Scan Tests ---")
    println()

    test_1d_cumsum(Float32, 1024, 64)
    println()
    test_1d_cumsum(Float32, 32768, 1024)
    println()
    test_large_cumsum(Float32, 1_000_000)
    println()
    test_1d_cumsum(Float64, 100_000, 1024)
    println()
    test_1d_cumprod(Float32, 10_000, 256)
    println()
    test_1d_cumsum_reverse(Float32, 10_000, 256)
    println()

    println("--- 2D Scan Tests ---")
    println()

    test_2d_cumsum(Float32, 1024, 2048, 32, 64)
    println()
    test_2d_cumsum(Float64, 512, 1024, 32, 64)
    println()

    println("="^70)
    println("All scan examples completed successfully!")
    println("="^70)
end

isinteractive() || main()
