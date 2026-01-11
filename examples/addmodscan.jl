using Test
using CUDA
using cuTile
import cuTile as ct

# Kernel using WrappedAddMod{360} for wrapped addition modulo 360
function addmodKernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                          tile_size::ct.Constant{Int})
    bid = ct.bid(1)
    tile = ct.load(a, bid, (tile_size[],))
    result = ct.scan(tile, ct.axis(1), ct.WrappedAddMod{360}())
    ct.store(b, bid, result)
    return nothing
end

sz = 32
N = 2^15
a = CUDA.rand(Float32, N)
b = CUDA.zeros(Float32, N)
CUDA.@sync ct.launch(addmodKernel, cld(length(a), sz), a, b, ct.Constant(sz))
res = Array(b)
exp = mod.(cumsum(Array(a), dims=1), 360)
println(res ≈ exp ? "  ✓ PASS" : "  ✗ FAIL")
