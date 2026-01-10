# Scan Example for cuTile.jl
# Demonstrates CSDL (Chained Scan with Decoupled Lookback) - single phase
# Uses memory ordering (acquire/release) for block chaining
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
    return nothing
end

function test_block_level()
    println("\n=== Block-level Cumsum (single tile) ===")
    n, sz = 256, 256
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)
    CUDA.@sync ct.launch(block_cumsum_kernel, 1, a, b, ct.Constant(sz))
    res = Array(b)
    exp = cumsum(Array(a), dims=1)
    println(res ≈ exp ? "  ✓ PASS" : "  ✗ FAIL")
end

# ============================================================================
# CSDL: Single-phase Chained Scan with Decoupled Lookback
# ============================================================================
#
# CSDL algorithm (single-phase):
# 1. Each block computes local tile scan
# 2. Each block atomically adds its tile sum to shared counter
# 3. Each block waits until its turn (counter >= bid)
# 4. Each block reads cumulative sum of all previous blocks
# 5. Adds cumulative sum to local scan and stores final result
#
# Memory ordering:
# - Atomic add with release: other blocks can see this block's contribution
# - Atomic load with acquire: see all previous blocks' contributions
# ============================================================================

function cumsum_csdl(input::ct.TileArray{Float32,1},
                     output::ct.TileArray{Float32,1},
                     tile_sums::ct.TileArray{Float32,1},
                     tile_size::ct.Constant{Int},
                     num_tiles::ct.Constant{Int})
    bid = ct.bid(1)
    bid_i = Int32(bid)

    # Step 1: Load tile and compute local cumulative sum
    tile = ct.load(input, bid, (tile_size[],))
    local_scan = ct.cumsum(tile, ct.axis(1))
    my_sum = ct.extract(local_scan, (tile_size[],), (1,))

    # Step 2: Atomically add my_sum to tile_sums[1] with release ordering
    # This makes my_sum visible to other blocks
    current_count = ct.atomic_add(tile_sums, (1,), my_sum)
    # current_count is now the sum of all blocks that wrote before me

    # Step 3: Wait for all previous blocks to complete
    # My position in chain is bid_i (1-indexed), I need tile_sums[1] >= bid_i
    # (each block contributes exactly once)
    while true
        accumulated = ct.load(tile_sums, (1,), (1,))
        if accumulated >= bid_i
            break
        end
    end

    # Step 4: Load cumulative sum of all previous tiles (acquire ordering)
    # This ensures we see all previous blocks' contributions
    prev_sums = ct.zeros((tile_size[],), Float32)
    k = Int32(1)
    while k < bid_i
        tile_sum_k = ct.load(tile_sums, (k,), (1,))
        prev_sums = prev_sums .+ tile_sum_k
        k += Int32(1)
    end

    # Step 5: Add cumulative previous sums to local scan
    result = local_scan .+ prev_sums

    # Step 6: Store final result
    ct.store(output, bid, result)

    return nothing
end

# Full CSDL cumsum wrapper (single phase)
function cumsum_csdl_full(input::CuArray{Float32,1}, tile_size::Int=256)
    n = length(input)
    num_tiles = cld(n, tile_size)
    output = CUDA.zeros(Float32, n)
    # tile_sums stores cumulative sum at index 1, each block adds once
    tile_sums = CUDA.zeros(Float32, 1)

    CUDA.@sync ct.launch(cumsum_csdl, num_tiles, input, output, tile_sums,
                         ct.Constant(tile_size), ct.Constant(num_tiles))
    return output
end

# ============================================================================
# Main test
# ============================================================================

function main()
    println("cuTile CSDL Scan Example")
    println("========================")
    println()
    println("GPU: $(CUDA.name(CUDA.device()))")
    println("Compute: $(CUDA.capability(CUDA.device()))")
    println()

    test_block_level()

    println("\nTest: CSDL Full Array (32K elements, 32 tiles of 1K)")
    n, tile_sz = 32768, 1024
    a = CUDA.rand(Float32, n)
    exp = cumsum(Array(a), dims=1)
    CUDA.@sync begin
        result = cumsum_csdl_full(a, tile_sz)
    end
    res = Array(result)
    println(res ≈ exp ? "  ✓ PASS" : "  ✗ FAIL")

    println("\nTest: CSDL Large Array (1M elements, 1024 tiles of 1K)")
    n, tile_sz = 1048576, 1024
    a = CUDA.rand(Float32, n)
    exp = cumsum(Array(a), dims=1)
    CUDA.@sync begin
        result = cumsum_csdl_full(a, tile_sz)
    end
    res = Array(result)
    println(res ≈ exp ? "  ✓ PASS" : "  ✗ FAIL")

    println("\n=== All tests complete ===")
end

main()
