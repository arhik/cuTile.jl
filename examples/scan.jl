
# CSDL (Chained Scan with Decoupled Lookback) Scan Example
#
# This example demonstrates parallel prefix sum using cuTile's scan operation.
# The CSDL algorithm is designed for efficient GPU execution with:
# - Single-phase operation (no global synchronization)
# - Chained lookback for progressive accumulation
# - O(1) amortized lookback complexity
#
# Hardware requirements:
# - GPU with sm_90+ architecture (Ampere or newer)
# - CUDA 12.0+
# - Julia with cuTile installed
#
# SPDX-License-Identifier: Apache-2.0

using CUDA
using cuTile
import cuTile as ct

#=============================================================================
# 1D Cumulative Sum Kernel (CSDL)
#=============================================================================

"""
    scan_1d_kernel(a::ct.TileArray{T,1}, b::ct.TileArray{T,1},
                   tile_size::ct.Constant{Int}) where T

1D cumulative sum kernel using CSDL scan algorithm.

# Arguments
- `a`: Input TileArray (CuArray wrapped with metadata)
- `b`: Output TileArray
- `tile_size`: Tile size for processing (Constant ghost type)

# Algorithm
- Each thread block processes a tile of `tile_size` elements
- Scan is computed along axis 0
- For tiles beyond the first, chained lookback accumulates previous partitions
"""
function scan_1d_kernel(a::ct.TileArray{T,1}, b::ct.TileArray{T,1},
                        tile_size::ct.Constant{Int}) where T
    bid = ct.bid(1)

    # Load tile from TileArray (a is already a TileArray, bid loads from it)
    tile = ct.load(a, bid, (tile_size[],))

    # Compute cumulative sum along axis 0
    result = ct.cumsum(tile, Val(0))

    # Store result back to TileArray
    ct.store(b, bid, result)

    return
end

#=============================================================================
# 2D Cumulative Sum Kernel - Scan Along Rows (Axis 1)
#=============================================================================

"""
    scan_2d_rows_kernel(a::ct.TileArray{T,2}, b::ct.TileArray{T,2},
                        tile_x::ct.Constant{Int}, tile_y::ct.Constant{Int}) where T

2D cumulative sum scanning along rows (axis 1).
Each row is processed independently.
"""
function scan_2d_rows_kernel(a::ct.TileArray{T,2}, b::ct.TileArray{T,2},
                             tile_x::ct.Constant{Int}, tile_y::ct.Constant{Int}) where T
    bid_x = ct.bid(1)
    bid_y = ct.bid(2)

    # Load tile: tile_y rows x tile_x columns
    tile = ct.load(a, (bid_x, bid_y), (tile_y[], tile_x[]))

    # Scan along axis 1 (columns within each row)
    result = ct.cumsum(tile, Val(1))

    ct.store(b, (bid_x, bid_y), result)

    return
end

#=============================================================================
# 2D Cumulative Sum Kernel - Scan Along Columns (Axis 0)
#=============================================================================

"""
    scan_2d_cols_kernel(a::ct.TileArray{T,2}, b::ct.TileArray{T,2},
                        tile_x::ct.Constant{Int}, tile_y::ct.Constant{Int}) where T

2D cumulative sum scanning along columns (axis 0).
Each column is processed independently.
"""
function scan_2d_cols_kernel(a::ct.TileArray{T,2}, b::ct.TileArray{T,2},
                             tile_x::ct.Constant{Int}, tile_y::ct.Constant{Int}) where T
    bid_x = ct.bid(1)
    bid_y = ct.bid(2)

    tile = ct.load(a, (bid_x, bid_y), (tile_y[], tile_x[]))

    # Scan along axis 0 (rows within each column)
    result = ct.cumsum(tile, Val(0))

    ct.store(b, (bid_x, bid_y), result)

    return
end

#=============================================================================
# 1D Cumulative Product Kernel
#=============================================================================

"""
    scan_1d_prod_kernel(a::ct.TileArray{T,1}, b::ct.TileArray{T,1},
                        tile_size::ct.Constant{Int}) where T

1D cumulative product using CSDL scan algorithm.
"""
function scan_1d_prod_kernel(a::ct.TileArray{T,1}, b::ct.TileArray{T,1},
                             tile_size::ct.Constant{Int}) where T
    bid = ct.bid(1)

    tile = ct.load(a, bid, (tile_size[],))
    result = ct.cumprod(tile, Val(0))

    ct.store(b, bid, result)

    return
end

#=============================================================================
# 1D Reverse Cumulative Sum Kernel
#=============================================================================

"""
    scan_1d_reverse_kernel(a::ct.TileArray{T,1}, b::ct.TileArray{T,1},
                           tile_size::ct.Constant{Int}) where T

Reverse cumulative sum (right-to-left scan).
"""
function scan_1d_reverse_kernel(a::ct.TileArray{T,1}, b::ct.TileArray{T,1},
                                tile_size::ct.Constant{Int}) where T
    bid = ct.bid(1)

    tile = ct.load(a, bid, (tile_size[],))
    # reverse=true for right-to-left accumulation
    result = ct.cumsum(tile, Val(0), true)

    ct.store(b, bid, result)

    return
end

#=============================================================================
# Verification Functions
#=============================================================================

"""
Verify scan result against CPU computation.
"""
function verify_scan(input::CuArray{T,1}, output::CuArray{T,1};
                     op::Symbol=:cumsum) where T
    cpu_input = Array(input)
    cpu_output = Array(output)

    if op == :cumsum
        expected = cumsum(cpu_input, dims=1)
    else
        expected = cumprod(cpu_input, dims=1)
    end

    if !(cpu_output ≈ expected)
        error("Verification failed for $op")
    end
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

    if !(cpu_output ≈ expected)
        error("Verification failed for $op (dim=$dim)")
    end
    println("  [PASS] $op verification passed (dim=$dim)")
end

"""
Benchmark kernel execution time.
"""
function benchmark_kernel(kernel_func, args...; warmup=3, iterations=10)
    # Warmup
    for _ in 1:warmup
        kernel_func(args...)
    end
    CUDA.synchronize()

    times = Float64[]
    for _ in 1:iterations
        t = CUDA.@elapsed kernel_func(args...)
        push!(times, t)
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

function test_1d_cumsum(::Type{T}, n, tile_size) where T
    name = "1D cumsum ($n elements, $T, tile=$tile_size)"
    println("--- $name ---")

    a = CUDA.rand(T, n)
    b = CUDA.zeros(T, n)

    # Wrap CuArrays in TileArray for cuTile
    a_tile = ct.TileArray(a)
    b_tile = ct.TileArray(b)

    # Launch kernel with grid dimension
    grid_dim = cld(n, tile_size)
    ct.launch(scan_1d_kernel, grid_dim, a_tile, b_tile, ct.Constant(tile_size))

    CUDA.synchronize()
    benchmark_kernel(scan_1d_kernel, a_tile, b_tile, ct.Constant(tile_size))
    verify_scan(a, b, op=:cumsum)
end

function test_1d_cumprod(::Type{T}, n, tile_size) where T
    name = "1D cumprod ($n elements, $T, tile=$tile_size)"
    println("--- $name ---")

    # Use positive values for cumprod
    a = CUDA.rand(T, n) .+ T(0.1)
    b = CUDA.zeros(T, n)

    a_tile = ct.TileArray(a)
    b_tile = ct.TileArray(b)

    grid_dim = cld(n, tile_size)
    ct.launch(scan_1d_prod_kernel, grid_dim, a_tile, b_tile, ct.Constant(tile_size))

    CUDA.synchronize()
    benchmark_kernel(scan_1d_prod_kernel, a_tile, b_tile, ct.Constant(tile_size))
    verify_scan(a, b, op=:cumprod)
end

function test_1d_cumsum_reverse(::Type{T}, n, tile_size) where T
    name = "1D reverse cumsum ($n elements, $T, tile=$tile_size)"
    println("--- $name ---")

    a = CUDA.rand(T, n)
    b = CUDA.zeros(T, n)

    a_tile = ct.TileArray(a)
    b_tile = ct.TileArray(b)

    grid_dim = cld(n, tile_size)
    ct.launch(scan_1d_reverse_kernel, grid_dim, a_tile, b_tile, ct.Constant(tile_size))

    CUDA.synchronize()
    benchmark_kernel(scan_1d_reverse_kernel, a_tile, b_tile, ct.Constant(tile_size))

    # Manual verification for reverse scan
    cpu_a = Array(a)
    cpu_b = Array(b)
    expected = reverse(cumsum(reverse(cpu_a), dims=1))
    if !(cpu_b ≈ expected)
        error("Reverse cumsum verification failed")
    end
    println("  [PASS] Reverse cumsum verification passed")
end

function test_2d_cumsum(::Type{T}, m, n, tile_x, tile_y) where T
    name = "2D cumsum ($m×$n, $T, tiles=$tile_x×$tile_y)"
    println("--- $name ---")

    # Column-wise scan (dim=1)
    a = CUDA.rand(T, m, n)
    b = CUDA.zeros(T, m, n)

    a_tile = ct.TileArray(a)
    b_tile = ct.TileArray(b)

    grid_dim = (cld(n, tile_x), cld(m, tile_y))
    ct.launch(scan_2d_rows_kernel, grid_dim, a_tile, b_tile,
              ct.Constant(tile_x), ct.Constant(tile_y))

    CUDA.synchronize()
    benchmark_kernel(scan_2d_rows_kernel, a_tile, b_tile,
                     ct.Constant(tile_x), ct.Constant(tile_y))
    verify_scan(a, b, 1, op=:cumsum)

    println()

    # Row-wise scan (dim=0)
    a2 = CUDA.rand(T, m, n)
    b2 = CUDA.zeros(T, m, n)

    a2_tile = ct.TileArray(a2)
    b2_tile = ct.TileArray(b2)

    ct.launch(scan_2d_cols_kernel, grid_dim, a2_tile, b2_tile,
              ct.Constant(tile_x), ct.Constant(tile_y))

    CUDA.synchronize()
    benchmark_kernel(scan_2d_cols_kernel, a2_tile, b2_tile,
                     ct.Constant(tile_x), ct.Constant(tile_y))
    verify_scan(a2, b2, 1, op=:cumsum)
end

function test_large_cumsum(::Type{T}, n) where T
    name = "Large 1D cumsum ($n elements, $T)"
    println("--- $name ---")

    tile_size = 1024
    a = CUDA.rand(T, n)
    b = CUDA.zeros(T, n)

    a_tile = ct.TileArray(a)
    b_tile = ct.TileArray(b)

    grid_dim = cld(n, tile_size)
    ct.launch(scan_1d_kernel, grid_dim, a_tile, b_tile, ct.Constant(tile_size))

    CUDA.synchronize()
    benchmark_kernel(scan_1d_kernel, a_tile, b_tile, ct.Constant(tile_size))
    verify_scan(a, b, op=:cumsum)
end

#=============================================================================
# Main Entry Point
#=============================================================================

function main()
    println("="^70)
    println("cuTile CSDL Scan Examples")
    println("="^70)
    println()

    # Check CUDA availability
    if !CUDA.functional()
        error("CUDA is not available. This example requires a CUDA-capable GPU.")
    end

    device = CUDA.device()
    println("GPU: $(CUDA.name(device))")
    compute_cap = CUDA.capability(device)
    println("Compute Capability: $compute_cap")
    println()

    # Tile IR requires sm_90+ (Ampere or newer)
    # RTX 50xx (Blackwell) is sm_100+ and fully supported
    if compute_cap.major < 9
        println("WARNING: Tile IR requires sm_90+ (Ampere or newer)")
        println("This GPU may not support Tile IR execution.")
        println()
    end

    println("--- 1D Scan Tests ---")
    println()

    test_1d_cumsum(Float32, 1024, 256)
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
