using Test
using CUDA
using cuTile
import cuTile as ct

function cumsum_1d_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                          tile_size::ct.Constant{Int})
    bid = ct.bid(1)
    tile = ct.load(a, bid, (tile_size[],))
    result = ct.cumsum(tile, ct.axis(1))
    ct.store(b, bid, result)
    return nothing
end

sz = 32
N = 2^15
a = CUDA.rand(Float32, N)
b = CUDA.zeros(Float32, N)
CUDA.@sync ct.launch(cumsum_1d_kernel, cld(length(a), sz), a, b, ct.Constant(sz))
res = Array(b)
exp = cumsum(Array(a), dims=1)
println(res ≈ exp ? "  ✓ PASS" : "  ✗ FAIL")


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
function cumsum_csdl_phase2(b::ct.TileArray{Float32,1},
                            tile_sums::ct.TileArray{Float32,1},
                            tile_size::ct.Constant{Int})
    bid = ct.bid(1)
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

n = length(a)
num_tiles = cld(n, sz)
tile_sums = CUDA.zeros(Float32, num_tiles)
CUDA.@sync ct.launch(cumsum_csdl_phase1, num_tiles, a, b, tile_sums, ct.Constant(sz))
CUDA.@sync ct.launch(cumsum_csdl_phase2, num_tiles, b, tile_sums, ct.Constant(sz))

@test isapprox(b |> collect, exp)

using BenchmarkTools

@benchmark CUDA.@sync begin
    CUDA.@sync ct.launch(cumsum_csdl_phase1, num_tiles, a, b, tile_sums, ct.Constant(sz))
    CUDA.@sync ct.launch(cumsum_csdl_phase2, num_tiles, b, tile_sums, ct.Constant(sz))
end
