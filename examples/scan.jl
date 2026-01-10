# Scan Example for cuTile.jl
# Demonstrates parallel prefix sum using Tile IR scan operation
#
# Run with: julia --project=. examples/scan.jl

using CUDA
using cuTile
import cuTile as ct

# 1D cumulative sum kernel (1-based indexing, returns nothing)
function cumsum_1d_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                          tile_size::ct.Constant{Int})
    bid = ct.bid(1)
    tile = ct.load(a, bid, (tile_size[],))
    result = ct.cumsum(tile, ct.axis(1))
    ct.store(b, bid, result)
    return
end

# 2D cumulative sum kernel - scan along axis 2 (columns)
function cumsum_2d_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2},
                          tile_x::ct.Constant{Int}, tile_y::ct.Constant{Int})
    bid_x = ct.bid(1)
    bid_y = ct.bid(2)
    tile = ct.load(a, (bid_x, bid_y), (tile_y[], tile_x[]))
    result = ct.cumsum(tile, ct.axis(2))
    ct.store(b, (bid_x, bid_y), result)
    return
end

# 1D cumulative product kernel
function cumprod_1d_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                           tile_size::ct.Constant{Int})
    bid = ct.bid(1)
    tile = ct.load(a, bid, (tile_size[],))
    result = ct.cumprod(tile, ct.axis(1))
    ct.store(b, bid, result)
    return
end

# Run scan example
function show_scan_example()
    println("\n=== Running 1D Cumsum Example ===")
    n, sz = 1024, 256
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)
    CUDA.@sync ct.launch(cumsum_1d_kernel, cld(n, sz), a, b, ct.Constant(sz))
    res = Array(b)
    exp = cumsum(Array(a), dims=1)
    println(res ≈ exp ? "  ✓ PASS" : "  ✗ FAIL")
    return
end

# ============================================================================
# CSDL: Chained Scan with Decoupled Lookback
# Full array cumulative sum across multiple tiles
# ============================================================================

# CSDL phase 1: Compute tile-level scan and store tile sums
function cumsum_csdl_phase1(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                            tile_sums::ct.TileArray{Float32,1},
                            tile_size::ct.Constant{Int})
    bid = ct.bid(1)
    # Load tile
    tile = ct.load(a, bid, (tile_size[],))
    # Intra-tile scan
    result = ct.cumsum(tile, ct.axis(1))
    # Store result
    ct.store(b, bid, result)
    # Extract and store tile sum (last element = cumulative sum of tile)
    tile_sum = ct.extract(result, (tile_size[],), (1,))
    ct.store(tile_sums, bid, tile_sum)
    return
end

# CSDL phase 2: Decoupled lookback - add accumulated sums from previous tiles
function cumsum_csdl_phase2(b::ct.TileArray{Float32,1},
                            tile_sums::ct.TileArray{Float32,1},
                            tile_size::ct.Constant{Int})
    bid = ct.bid(1)
    # Skip first block (no previous sums to add)
    if bid == Int32(1)
        return
    end
    # Compute sum of all previous tile sums (prefix sum of tile_sums[1:bid-1])
    prev_sum = ct.zeros((tile_size[],), Float32)
    k = Int32(1)
    while k < bid
        tile_sum_k = ct.load(tile_sums, (k,), (1,))
        prev_sum = prev_sum + tile_sum_k
        k += Int32(1)
    end
    # Load current tile, add accumulated sum, store back
    tile = ct.load(b, bid, (tile_size[],))
    result = tile + prev_sum
    ct.store(b, bid, result)
    return
end

# Full CSDL cumsum (both phases, automatic synchronization via separate launches)
function cumsum_csdl(input::CuArray{Float32,1}, tile_size::Int=256)
    n = length(input)
    num_tiles = cld(n, tile_size)
    # Allocate output and tile sums storage
    output = CUDA.zeros(Float32, n)
    tile_sums = CUDA.zeros(Float32, num_tiles)
    # Phase 1: intra-tile scan + store tile sums
    CUDA.@sync ct.launch(cumsum_csdl_phase1, num_tiles, input, output, tile_sums, ct.Constant(tile_size))
    # Phase 2: decoupled lookback to accumulate previous tile sums
    CUDA.@sync ct.launch(cumsum_csdl_phase2, num_tiles, output, tile_sums, ct.Constant(tile_size))
    return output
end

# Main test
function main()
    println("cuTile Scan Example")
    println("===================")
    println()
    println("GPU: $(CUDA.name(CUDA.device()))")
    println("Compute: $(CUDA.capability(CUDA.device()))")
    println()

    # Run scan examples
    show_scan_example()

    # Run CSDL full array cumsum
    println("\nTest CSDL: Full Array Cumsum (multi-tile)")
    n, tile_sz = 32768, 1024
    a = CUDA.rand(Float32, n)
    exp = cumsum(Array(a), dims=1)
    CUDA.@sync begin
        result = cumsum_csdl(a, tile_sz)
    end
    res = Array(result)
    println(res ≈ exp ? "  ✓ PASS" : "  ✗ FAIL")

    # Test 1: 1D cumsum
    println("\nTest 1: 1D cumsum (32768 elements)")
    n, sz = 32768, 1024
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)
    CUDA.@sync ct.launch(cumsum_1d_kernel, cld(n, sz), a, b, ct.Constant(sz))
    res = Array(b)
    exp = cumsum(Array(a), dims=1)
    println(res ≈ exp ? "  ✓ PASS" : "  ✗ FAIL")

    # Test 2: 2D cumsum along axis 2
    println("\nTest 2: 2D cumsum (256 x 512), axis 2")
    m, n, tx, ty = 256, 512, 32, 32
    a = CUDA.rand(Float32, m, n)
    b = CUDA.zeros(Float32, m, n)
    CUDA.@sync ct.launch(cumsum_2d_kernel, (cld(n, tx), cld(m, ty)),
                         a, b, ct.Constant(tx), ct.Constant(ty))
    res = Array(b)
    exp = cumsum(Array(a), dims=2)
    println(res ≈ exp ? "  ✓ PASS" : "  ✗ FAIL")

    # Test 3: 1D cumprod
    println("\nTest 3: 1D cumprod (10000 elements)")
    n, sz = 10000, 256
    a = CUDA.rand(Float32, n) .+ 0.1f0
    b = CUDA.zeros(Float32, n)
    CUDA.@sync ct.launch(cumprod_1d_kernel, cld(n, sz), a, b, ct.Constant(sz))
    res = Array(b)
    exp = cumprod(Array(a), dims=1)
    println(res ≈ exp ? "  ✓ PASS" : "  ✗ FAIL")

    println("\n=== All tests complete ===")
end

main()
