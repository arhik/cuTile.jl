# mlir_check.jl
# Simple verification that mapreduce expression decomposition works
# Usage: julia> include("mlir_check.jl")

using cuTile
import cuTile as ct

println("\n" * repeat("=", 60))
println("MLIR VERIFICATION TEST")
println("Verifying mapreduce expression decomposition")
println(repeat("=", 60))

# Define ArraySpecs following test/codegen.jl pattern
const spec1d = ct.ArraySpec{1}(16, true)
const spec2d = ct.ArraySpec{2}(16, true)

# Test 1: x + 1
println("\nTEST 1: x -> x + 1")
println("Expected: addf(elem, const)")
function k1(a::ct.TileArray{Float32, 2, spec2d}, b::ct.TileArray{Float32, 1, spec1d})
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    ct.mapreduce(x -> x + 1, +, tile, 2)
end
ct.code_tiled(k1, Tuple{ct.TileArray{Float32, 2, spec2d}, ct.TileArray{Float32, 1, spec1d}})

# Test 2: x^2
println("\nTEST 2: x -> x^2")
println("Expected: mulf(elem, elem)")
function k2(a::ct.TileArray{Float32, 2, spec2d}, b::ct.TileArray{Float32, 1, spec1d})
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    ct.mapreduce(x -> x^2, +, tile, 2)
end
ct.code_tiled(k2, Tuple{ct.TileArray{Float32, 2, spec2d}, ct.TileArray{Float32, 1, spec1d}})

# Test 3: 2 * x
println("\nTEST 3: x -> 2 * x")
println("Expected: mulf(const, elem)")
function k3(a::ct.TileArray{Float32, 2, spec2d}, b::ct.TileArray{Float32, 1, spec1d})
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    ct.mapreduce(x -> 2 * x, *, tile, 2)
end
ct.code_tiled(k3, Tuple{ct.TileArray{Float32, 2, spec2d}, ct.TileArray{Float32, 1, spec1d}})

# Test 4: sin(x) + 1
println("\nTEST 4: x -> sin(x) + 1")
println("Expected: addf(sinf(elem), const)")
function k4(a::ct.TileArray{Float32, 2, spec2d}, b::ct.TileArray{Float32, 1, spec1d})
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    ct.mapreduce(x -> sin(x) + 1, +, tile, 2)
end
ct.code_tiled(k4, Tuple{ct.TileArray{Float32, 2, spec2d}, ct.TileArray{Float32, 1, spec1d}})

# Test 5: abs(x - 1)
println("\nTEST 5: x -> abs(x - 1)")
println("Expected: absf(subf(elem, const))")
function k5(a::ct.TileArray{Float32, 2, spec2d}, b::ct.TileArray{Float32, 1, spec1d})
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    ct.mapreduce(x -> abs(x - 1), max, tile, 2)
end
ct.code_tiled(k5, Tuple{ct.TileArray{Float32, 2, spec2d}, ct.TileArray{Float32, 1, spec1d}})

# Test 6: identity (baseline)
println("\nTEST 6: Baseline: identity +")
println("Expected: simple addf")
function k6(a::ct.TileArray{Float32, 2, spec2d}, b::ct.TileArray{Float32, 1, spec1d})
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    ct.mapreduce(identity, +, tile, 2)
end
ct.code_tiled(k6, Tuple{ct.TileArray{Float32, 2, spec2d}, ct.TileArray{Float32, 1, spec1d}})

println("\n" * repeat("=", 60))
println("Check output for 'reduce' with decomposed operations:")
println("  addf, mulf, subf, sinf, absf")
println("="^60 * "\n")
