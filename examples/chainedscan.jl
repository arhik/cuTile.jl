using Test
using Revise
using CUDA
using cuTile
import cuTile as ct

function chainedScan(
        a::ct.TileArray{Float32,1},
        b::ct.TileArray{Float32,1},
        stage::ct.TileArray{UInt32, 1},
        locks::ct.TileArray{UInt32, 1},
        tile_sums::ct.TileArray{Float32,1},
        tile_size::ct.Constant{Int}
    )
    bid = ct.bid(1)
    tile = ct.load(a, bid, (tile_size[],))
    result = ct.cumsum(tile, ct.axis(1))
    ct.store(b, bid, result)
    tile_sum = ct.extract(result, (tile_size[],), (1,))
    ct.store(tile_sums, bid, tile_sum)
    prev_sum = ct.zeros((tile_size[],), Float32)
    k = Int32(1)
    while k < bid
        tile_sum_k = ct.load(tile_sums, (k,), (1,))
        prev_sum = prev_sum .+ tile_sum_k
        k += Int32(1)
    end
    tile = ct.load(b, bid, (tile_size[],))
    result = tile .+ prev_sum
    ct.store(b, bid, result)
    return nothing
end

N = 2^15
sz = 32
elType = Float32
a = CUDA.rand(elType, N)
b = CUDA.zeros(elType, N)

num_tiles = cld(N, sz)
stage = CUDA.zeros(UInt32, num_tiles)
locks = CUDA.zeros(UInt32, num_tiles)
tile_sums = CUDA.zeros(elType, num_tiles)
CUDA.@sync ct.launch(chainedScan, num_tiles, a, b, stage, locks, tile_sums, ct.Constant(sz))

a_cpu = cumsum(a |> collect, dims=1)
@test isapprox(b |> collect, )

using BenchmarkTools

@benchmark CUDA.@sync begin
    CUDA.@sync ct.launch(chainedScan, num_tiles, a, b, stage, locks, tile_sums, ct.Constant(sz))
end
