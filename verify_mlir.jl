# verify_mlir.jl
# Verify expression decomposition by generating MLIR/Tile IR output
# Usage: julia> include("verify_mlir.jl")

using cuTile
import cuTile as ct

println("\n" * repeat("=", 70))
println("MAPREDUCE EXPRESSION DECOMPOSITION VERIFICATION")
println("Generating MLIR output to verify decomposition works")
println(repeat("=", 70))

# Create fully concrete TileArray types
input_2d_type = ct.TileArray{Float32, 2, ct.ArraySpec{2}(128, 8, true, (0,), (32,))}
output_1d_type = ct.TileArray{Float32, 1, ct.ArraySpec{1}(128, 1, true, (0,), (32,))}

# Define argtypes tuple
argtypes = Tuple{typeof(input_2d_type), typeof(output_1d_type)}

println("\nArgument types: ", argtypes)

# Test 1: x -> x + 1
println("\n" * repeat("=", 60))
println("TEST 1: x -> x + 1")
println("Expected: reduce with addf and constant(1)")
println(repeat("=", 60))

function kernel1(a::typeof(input_2d_type), b::typeof(output_1d_type))
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    result = ct.mapreduce(x -> x + 1, +, tile, 2)
    ct.store(b, pid, result)
    return
end

println("\nGenerated Tile IR:")
ct.code_tiled(kernel1, argtypes)

# Test 2: x -> x^2
println("\n" * repeat("=", 60))
println("TEST 2: x -> x^2")
println("Expected: reduce with mulf(elem, elem)")
println(repeat("=", 60))

function kernel2(a::typeof(input_2d_type), b::typeof(output_1d_type))
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    result = ct.mapreduce(x -> x^2, +, tile, 2)
    ct.store(b, pid, result)
    return
end

println("\nGenerated Tile IR:")
ct.code_tiled(kernel2, argtypes)

# Test 3: x -> 2 * x
println("\n" * repeat("=", 60))
println("TEST 3: x -> 2 * x")
println("Expected: reduce with mulf(const(2), elem)")
println(repeat("=", 60))

function kernel3(a::typeof(input_2d_type), b::typeof(output_1d_type))
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    result = ct.mapreduce(x -> 2 * x, *, tile, 2)
    ct.store(b, pid, result)
    return
end

println("\nGenerated Tile IR:")
ct.code_tiled(kernel3, argtypes)

# Test 4: x -> sin(x) + 1
println("\n" * repeat("=", 60))
println("TEST 4: x -> sin(x) + 1")
println("Expected: reduce with sinf, addf, and constant(1)")
println(repeat("=", 60))

function kernel4(a::typeof(input_2d_type), b::typeof(output_1d_type))
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    result = ct.mapreduce(x -> sin(x) + 1, +, tile, 2)
    ct.store(b, pid, result)
    return
end

println("\nGenerated Tile IR:")
ct.code_tiled(kernel4, argtypes)

# Test 5: x -> abs(x - 1)
println("\n" * repeat("=", 60))
println("TEST 5: x -> abs(x - 1)")
println("Expected: reduce with absf and subf")
println(repeat("=", 60))

function kernel5(a::typeof(input_2d_type), b::typeof(output_1d_type))
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    result = ct.mapreduce(x -> abs(x - 1), max, tile, 2)
    ct.store(b, pid, result)
    return
end

println("\nGenerated Tile IR:")
ct.code_tiled(kernel5, argtypes)

# Test 6: x -> (x + 1) * 2
println("\n" * repeat("=", 60))
println("TEST 6: x -> (x + 1) * 2")
println("Expected: reduce with nested operations")
println(repeat("=", 60))

function kernel6(a::typeof(input_2d_type), b::typeof(output_1d_type))
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    result = ct.mapreduce(x -> (x + 1) * 2, *, tile, 2)
    ct.store(b, pid, result)
    return
end

println("\nGenerated Tile IR:")
ct.code_tiled(kernel6, argtypes)

println("\n" * repeat("=", 70))
println("VERIFICATION COMPLETE")
println(repeat("=", 70))
println("\nCheck the output above for:")
println("  1. cuda_tile.reduce operation")
println("  2. Body with decomposed operations:")
println("     - x + 1       -> addf(elem, const)")
println("     - x^2         -> mulf(elem, elem)")
println("     - 2 * x       -> mulf(const, elem)")
println("     - sin(x) + 1  -> addf(sinf(elem), const)")
println("     - abs(x - 1)  -> absf(subf(elem, const))")
println("     - (x+1) * 2   -> mulf(addf(...), const)")
println("\nIf all tests compile without errors, decomposition is working!")
println(repeat("=", 70) * "\n")
