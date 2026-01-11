using Test
using CUDA
using cuTile
import cuTile as ct

# Define custom operator
struct MyCounter <: ct.ScanOpMarker end

# Implement required methods
ct.scan_combine(::MyCounter, a::ct.Tile{Float32}, b::ct.Tile{Float32}) = max(a, b)
ct.scan_identity(::MyCounter, ::Type{Float32}) = Float32(-Inf)

function addmodKernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                          tile_size::ct.Constant{Int})
    bid = ct.bid(1)
    tile = ct.load(a, bid, (tile_size[],))
    result = ct.scan(tile, ct.axis(1), MyCounter())
    ct.store(b, bid, result)
    return nothing
end

sz = 32
N = 2^15
a = CUDA.rand(Float32, N)
b = CUDA.zeros(Float32, N)
CUDA.@sync ct.launch(addmodKernel, cld(length(a), sz), a, b, ct.Constant(sz))
res = Array(b)
exp = mod.(cumsum(Array(a), dims=1), 0.5)
println(res ≈ exp ? "  ✓ PASS" : "  ✗ FAIL")
