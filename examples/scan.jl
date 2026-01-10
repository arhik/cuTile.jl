# Scan Example for cuTile.jl
# Demonstrates CSDL (Chained Scan with Decoupled Lookback) for full array cumsum
#
# Run with: julia --project=. examples/scan.jl

using CUDA
using cuTile
import cuTile as ct

# ============================================================================
# Block-level cumsum (single tile)
# ============================================================================

function block_cumsum_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                             tile_size::ct.Constant{Int})
    bid = ct.bid(1)
    tile = ct.load(a, bid, (tile_size[],))
    result = ct.cumsum(tile, ct.axis(1))
    ct.store(b, bid, result)
    return
end

function show_block_cumsum_example()
    println("\n=== Block-level Cumsum (single tile) ===")
    n, sz = 1024, 256
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)
    CUDA.@sync ct.launch(block_cumsum_kernel, cld(n, sz), a, b, ct.Constant(sz))
    res = Array(b)
    exp = cumsum(Array(a), dims=1)
    println(res ≈ exp ? "  ✓ PASS" : "  ✗ FAIL")
    return
end

# ============================================================================
# CSDL: Chained Scan with Decoupled Lookback
# Full array cumulative sum across multiple tiles
# ============================================================================

# CSDL phase 1: Intra-tile scan + store tile sums
function cumsum_csdl_phase1(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                            tile_sums::ct.TileArray{Float32,1},
                            tile_size::ct.Constant{Int})
    bid = ct.bid(1)
    tile = ct.load(a, bid, (tile_size[],))
    result = ct.cumsum(tile, ct.axis(1))
    ct.store(b, bid, result)
    tile_sum = ct.extract(result, (tile_size[],), (1,))
    ct.store(tile_sums, bid, tile_sum)
    return
end

# CSDL phase 2: Decoupled lookback to accumulate previous tile sums
# No if/return - uses while loop predication (first tile: k < 1 never runs, prev_sum = 0)
function cumsum_csdl_phase2(b::ct.TileArray{Float32,1},
                            tile_sums::ct.TileArray{Float32,1},
                            tile_size::ct.Constant{Int})
    bid = ct.bid(1)

    # Accumulate sum of all previous tile sums
    # For bid=1: k < 1 is false, loop never runs, prev_sum = 0
    prev_sum = ct.zeros((tile_size[],), Float32)
    k = Int32(1)
    while k < bid
        tile_sum_k = ct.load(tile_sums, (k,), (1,))
        prev_sum = prev_sum + tile_sum_k
        k += Int32(1)
    end

    # Add accumulated sum to current tile
    tile = ct.load(b, bid, (tile_size[],))
    result = tile + prev_sum
    ct.store(b, bid, result)
    return
end

# Full CSDL cumsum wrapper
function cumsum_csdl(input::CuArray{Float32,1}, tile_size::Int=256)
    n = length(input)
    num_tiles = cld(n, tile_size)
    output = CUDA.zeros(Float32, n)
    tile_sums = CUDA.zeros(Float32, num_tiles)
    CUDA.@sync ct.launch(cumsum_csdl_phase1, num_tiles, input, output, tile_sums, ct.Constant(tile_size))
    CUDA.@sync ct.launch(cumsum_csdl_phase2, num_tiles, output, tile_sums, ct.Constant(tile_size))
    return output
end

# Main test
function main()
    println("cuTile CSDL Scan Example")
    println("========================")
    println()
    println("GPU: $(CUDA.name(CUDA.device()))")
    println("Compute: $(CUDA.capability(CUDA.device()))")
    println()

    show_block_cumsum_example()

    println("\nTest: CSDL Full Array Cumsum (32K elements, 32 tiles of 1K)")
    n, tile_sz = 32768, 1024
    a = CUDA.rand(Float32, n)
    exp = cumsum(Array(a), dims=1)
    CUDA.@sync begin
        result = cumsum_csdl(a, tile_sz)
    end
    res = Array(result)
    println(res ≈ exp ? "  ✓ PASS" : "  ✗ FAIL")

    println("\nTest: CSDL Large Array (1M elements, 1024 tiles of 1K)")
    n, tile_sz = 1048576, 1024
    a = CUDA.rand(Float32, n)
    exp = cumsum(Array(a), dims=1)
    CUDA.@sync begin
        result = cumsum_csdl(a, tile_sz)
    end
    res = Array(result)
    println(res ≈ exp ? "  ✓ PASS" : "  ✗ FAIL")

    println("\n=== All tests complete ===")
end

main()
