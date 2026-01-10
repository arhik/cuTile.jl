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

# 1D Cumulative Sum Kernel
function cumsum_1d_kernel(a::ct.TileArray{T,1}, b::ct.TileArray{T,1},
                          tile_size::ct.Constant{Int}) where T
    bid = ct.bid(1)
    tile = ct.load(a, bid, (tile_size[],))
    result = ct.cumsum(tile, Val(0))
    ct.store(b, bid, result)
    return
end

# 2D Cumulative Sum - Scan Along Rows (Axis 1)
function cumsum_2d_rows_kernel(a::ct.TileArray{T,2}, b::ct.TileArray{T,2},
                                tile_x::ct.Constant{Int}, tile_y::ct.Constant{Int}) where T
    bid_x = ct.bid(1)
    bid_y = ct.bid(2)
    tile = ct.load(a, (bid_x, bid_y), (tile_y[], tile_x[]))
    result = ct.cumsum(tile, Val(1))
    ct.store(b, (bid_x, bid_y), result)
    return
end

# 2D Cumulative Sum - Scan Along Columns (Axis 0)
function cumsum_2d_cols_kernel(a::ct.TileArray{T,2}, b::ct.TileArray{T,2},
                                tile_x::ct.Constant{Int}, tile_y::ct.Constant{Int}) where T
    bid_x = ct.bid(1)
    bid_y = ct.bid(2)
    tile = ct.load(a, (bid_x, bid_y), (tile_y[], tile_x[]))
    result = ct.cumsum(tile, Val(0))
    ct.store(b, (bid_x, bid_y), result)
    return
end

# 1D Cumulative Product Kernel
function cumprod_1d_kernel(a::ct.TileArray{T,1}, b::ct.TileArray{T,1},
                           tile_size::ct.Constant{Int}) where T
    bid = ct.bid(1)
    tile = ct.load(a, bid, (tile_size[],))
    result = ct.cumprod(tile, Val(0))
    ct.store(b, bid, result)
    return
end

# 1D Reverse Cumulative Sum Kernel
function cumsum_reverse_1d_kernel(a::ct.TileArray{T,1}, b::ct.TileArray{T,1},
                                   tile_size::ct.Constant{Int}) where T
    bid = ct.bid(1)
    tile = ct.load(a, bid, (tile_size[],))
    result = ct.cumsum(tile, Val(0), true)
    ct.store(b, bid, result)
    return
end

# Verify 1D scan result against CPU
function verify_1d(input::CuArray{T,1}, output::CuArray{T,1}; op::Symbol=:cumsum) where T
    cpu_input = Array(input)
    cpu_output = Array(output)
    if op == :cumsum
        expected = cumsum(cpu_input, dims=1)
    else
        expected = cumprod(cpu_input, dims=1)
    end
    if cpu_output ≈ expected
        println("  [PASS] $op verification passed")
    else
        max_err = maximum(abs.(cpu_output .- expected))
        @printf "  [FAIL] $op verification failed, max error: %.6e\n" max_err
    end
end

# Verify 2D scan result against CPU
function verify_2d(input::CuArray{T,2}, output::CuArray{T,2}, dim::Int; op::Symbol=:cumsum) where T
    cpu_input = Array(input)
    cpu_output = Array(output)
    if op == :cumsum
        expected = cumsum(cpu_input, dims=dim)
    else
        expected = cumprod(cpu_input, dims=dim)
    end
    if cpu_output ≈ expected
        println("  [PASS] $op verification passed (dim=$dim)")
    else
        max_err = maximum(abs.(cpu_output .- expected))
        @printf "  [FAIL] $op verification failed (dim=$dim), max error: %.6e\n" max_err
    end
end

# Benchmark kernel
function benchmark_kernel(name::String, kernel_func, args...; warmup=3, iterations=10)
    println("--- $name ---")
    for _ in 1:warmup
        CUDA.@sync kernel_func(args...)
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
    @printf "  Time: %.3f ms (+-%.3f)\n" mean_time std_time
    return mean_time
end

# Test 1D cumsum
function test_1d_cumsum(::Type{T}, n, tile_size) where T
    name = "1D cumsum ($n elements, $T, tile=$tile_size)"
    a = CUDA.rand(T, n)
    b = CUDA.zeros(T, n)
    a_tile = ct.TileArray(a)
    b_tile = ct.TileArray(b)
    grid_dim = cld(n, tile_size)
    ct.launch(cumsum_1d_kernel, grid_dim, a_tile, b_tile, ct.Constant(tile_size))
    CUDA.synchronize()
    benchmark_kernel(name, cumsum_1d_kernel, a_tile, b_tile, ct.Constant(tile_size))
    verify_1d(a, b, op=:cumsum)
end

# Test 1D cumprod
function test_1d_cumprod(::Type{T}, n, tile_size) where T
    name = "1D cumprod ($n elements, $T, tile=$tile_size)"
    a = CUDA.rand(T, n) .+ T(0.1)
    b = CUDA.zeros(T, n)
    a_tile = ct.TileArray(a)
    b_tile = ct.TileArray(b)
    grid_dim = cld(n, tile_size)
    ct.launch(cumprod_1d_kernel, grid_dim, a_tile, b_tile, ct.Constant(tile_size))
    CUDA.synchronize()
    benchmark_kernel(name, cumprod_1d_kernel, a_tile, b_tile, ct.Constant(tile_size))
    verify_1d(a, b, op=:cumprod)
end

# Test 1D reverse cumsum
function test_1d_cumsum_reverse(::Type{T}, n, tile_size) where T
    name = "1D reverse cumsum ($n elements, $T, tile=$tile_size)"
    a = CUDA.rand(T, n)
    b = CUDA.zeros(T, n)
    a_tile = ct.TileArray(a)
    b_tile = ct.TileArray(b)
    grid_dim = cld(n, tile_size)
    ct.launch(cumsum_reverse_1d_kernel, grid_dim, a_tile, b_tile, ct.Constant(tile_size))
    CUDA.synchronize()
    benchmark_kernel(name, cumsum_reverse_1d_kernel, a_tile, b_tile, ct.Constant(tile_size))
    cpu_a = Array(a)
    cpu_b = Array(b)
    expected = reverse(cumsum(reverse(cpu_a), dims=1))
    if cpu_b ≈ expected
        println("  [PASS] Reverse cumsum verification passed")
    else
        println("  [FAIL] Reverse cumsum verification failed")
    end
end

# Test 2D cumsum
function test_2d_cumsum(::Type{T}, m, n, tile_x, tile_y) where T
    name = "2D cumsum ($m x $n, $T, tiles=$tile_x x $tile_y)"
    a = CUDA.rand(T, m, n)
    b = CUDA.zeros(T, m, n)
    a_tile = ct.TileArray(a)
    b_tile = ct.TileArray(b)
    grid_dim = (cld(n, tile_x), cld(m, tile_y))
    ct.launch(cumsum_2d_rows_kernel, grid_dim, a_tile, b_tile, ct.Constant(tile_x), ct.Constant(tile_y))
    CUDA.synchronize()
    benchmark_kernel(name, cumsum_2d_rows_kernel, a_tile, b_tile, ct.Constant(tile_x), ct.Constant(tile_y))
    verify_2d(a, b, 1, op=:cumsum)
end

# Test large 1D cumsum
function test_large_cumsum(::Type{T}, n) where T
    name = "Large 1D cumsum ($n elements, $T)"
    tile_size = 1024
    a = CUDA.rand(T, n)
    b = CUDA.zeros(T, n)
    a_tile = ct.TileArray(a)
    b_tile = ct.TileArray(b)
    grid_dim = cld(n, tile_size)
    ct.launch(cumsum_1d_kernel, grid_dim, a_tile, b_tile, ct.Constant(tile_size))
    CUDA.synchronize()
    benchmark_kernel(name, cumsum_1d_kernel, a_tile, b_tile, ct.Constant(tile_size))
    verify_1d(a, b, op=:cumsum)
end

function main()
    println("======================================================================")
    println("cuTile CSDL Scan Examples")
    println("======================================================================")
    println()

    if !CUDA.functional()
        error("CUDA is not available. This example requires a CUDA-capable GPU.")
    end

    device = CUDA.device()
    println("GPU: $(CUDA.name(device))")
    compute_cap = CUDA.capability(device)
    println("Compute Capability: $compute_cap")
    println()

    if compute_cap.major < 9
        println("WARNING: Tile IR requires sm_90+ (Ampere or newer)")
        println("This GPU may not support Tile IR execution.")
        println()
    end

    println("1D Scan Tests")
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

    println("2D Scan Tests")
    println()

    test_2d_cumsum(Float32, 1024, 2048, 32, 32)
    println()

    test_2d_cumsum(Float64, 512, 1024, 32, 32)
    println()

    println("======================================================================")
    println("All scan examples completed!")
    println("======================================================================")
end

isinteractive() || main()
