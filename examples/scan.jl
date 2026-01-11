using Revise
using Test
using CUDA
using cuTile
import cuTile as ct

function chainedScan(
        input::ct.TileArray{Float32,1},
        output::ct.TileArray{Float32,1},
        locks::ct.TileArray{Int32,1},
        tile_sums::ct.TileArray{Float32,2},
        tile_size::ct.Constant{Int},
        num_tiles::ct.Constant{Int}
    )
    bid = ct.bid(1)
    bid_i = Int32(bid)

    tile = ct.load(input, bid, (tile_size[],))
    local_scan = ct.cumsum(tile, ct.axis(1))
    my_tile_sum = ct.extract(local_scan, (tile_size[],), (1,))

    # # Step 2: Acquire lock to read accumulated tile sums and add mine
    # # Spin until we acquire the lock
    # while ct.atomic_cas(locks, (1,), Int32(0), Int32(1);
    #                    memory_order=ct.MemoryOrder.Acquire) == Int32(1)
    #     # Spin - lock is held by another block
    # end

    # # Now we hold the lock - critical section
    # # Read accumulated tile sums up to bid_i-1
    # prev_sum = ct.zeros((tile_size[],), Float32)
    # k = Int32(1)
    # while k < bid_i
    #     tile_sum_k = ct.load(tile_sums, (k, 1), (1, 1))
    #     prev_sum = prev_sum .+ tile_sum_k
    #     k += Int32(1)
    # end

    # # Add my tile sum to the accumulated column
    # ct.store(tile_sums, (bid_i, 1), my_tile_sum)

    # # Step 3: Release lock (next block can proceed)
    # # Release semantics ensures our writes are visible
    # ct.atomic_cas(locks, (1,), Int32(1), Int32(0);
    #               memory_order=ct.MemoryOrder.Release)

    # # Step 4: Add accumulated sum from previous tiles to local scan
    # result = local_scan .+ prev_sum
    # ct.store(output, bid, result)

    return nothing
end

N = 2^15
elType = Float32
input = CUDA.rand(elType, N)
output = CUDA.zeros(elType, N)
sz = 32
num_tiles = cld(N, sz)
locks = CUDA.zeros(Int32, 1)
tile_sums = CUDA.zeros(elType, (num_tiles, 1))

CUDA.@sync ct.launch(chainedScan, num_tiles, input, output, locks, tile_sums, ct.Constant(sz), ct.Constant(num_tiles))
