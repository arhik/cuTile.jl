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
# CSDL: Chained Scan with Decoupled Lookback
#
# Uses spinlock with atomic_cas for synchronization between blocks.
# Each block:
# 1. Acquires lock to read accumulated tile sums
# 2. Computes local cumsum + accumulated sum from previous tiles
# 3. Stores result
# 4. Releases lock (implicit, next block can proceed)
# ============================================================================

function cumsum_csdl(input::ct.TileArray{Float32,1},
                     output::ct.TileArray{Float32,1},
                     locks::ct.TileArray{Int32,1},
                     tile_sums::ct.TileArray{Float32,2},
                     tile_size::ct.Constant{Int},
                     num_tiles::ct.Constant{Int})
    bid = ct.bid(1)
    bid_i = Int32(bid)

    # Step 1: Compute local cumsum for this tile
    tile = ct.load(input, bid, (tile_size[],))
    local_scan = ct.cumsum(tile, ct.axis(1))
    my_tile_sum = ct.extract(local_scan, (tile_size[],), (1,))

    # Step 2: Acquire lock to read accumulated tile sums and add mine
    # Spin until we acquire the lock
    while ct.atomic_cas(locks, (1,), Int32(0), Int32(1);
                       memory_order=ct.MemoryOrder.Acquire) == Int32(1)
        # Spin - lock is held by another block
    end

    # Now we hold the lock - critical section
    # Read accumulated tile sums up to bid_i-1
    prev_sum = ct.zeros((tile_size[],), Float32)
    k = Int32(1)
    while k < bid_i
        tile_sum_k = ct.load(tile_sums, (k, 1), (1, 1))
        prev_sum = prev_sum .+ tile_sum_k
        k += Int32(1)
    end

    # Add my tile sum to the accumulated column
    ct.store(tile_sums, (bid_i, 1), my_tile_sum)

    # Step 3: Release lock (next block can proceed)
    # Release semantics ensures our writes are visible
    ct.atomic_cas(locks, (1,), Int32(1), Int32(0);
                  memory_order=ct.MemoryOrder.Release)

    # Step 4: Add accumulated sum from previous tiles to local scan
    result = local_scan .+ prev_sum
    ct.store(output, bid, result)

    return nothing
end

# Full CSDL cumsum wrapper
function cumsum_csdl_full(input::CuArray{Float32,1}, tile_size::Int=256)
    n = length(input)
    num_tiles = cld(n, tile_size)
    output = CUDA.zeros(Float32, n)

    # locks: single Int32, 0 = unlocked, 1 = locked
    locks = CUDA.zeros(Int32, 1)

    # tile_sums: (num_tiles, 1) - cumulative sum of each tile
    # We use 2D to allow ct.load with (k, 1) indexing
    tile_sums = CUDA.zeros(Float32, (num_tiles, 1))

    CUDA.@sync ct.launch(cumsum_csdl, num_tiles, input, output, locks, tile_sums,
                         ct.Constant(tile_size), ct.Constant(num_tiles))

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
