# simple_verify.jl
# Simple MLIR verification test for mapreduce expression decomposition
# Usage: julia> include("simple_verify.jl")

using cuTile
import cuTile as ct

println("\n" * repeat("=", 70))
println("SIMPLE MLIR VERIFICATION TEST")
println("Testing mapreduce expression decomposition")
println(repeat("=", 70))

# Create concrete TileArray type
const InputType = ct.TileArray{Float32, 2, ct.ArraySpec{2}(128, 8, true, (0,), (32,))}
const OutputType = ct.TileArray{Float32, 1, ct.ArraySpec{1}(128, 1, true, (0,), (32,))}
const ArgTypes = (InputType, OutputType)

# Test 1: x + 1
println("\n" * repeat("=", 50))
println("TEST 1: x -> x + 1")
println(repeat("=", 50))

function kernel1(a::InputType, b::OutputType)
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    ct.mapreduce(x -> x + 1, +, tile, 2)
end

ct.code_tiled(kernel1, ArgTypes)

# Test 2: x^2
println("\n" * repeat("=", 50))
println("TEST 2: x -> x^2")
println(repeat("=", 50))

function kernel2(a::InputType, b::OutputType)
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    ct.mapreduce(x -> x^2, +, tile, 2)
end

ct.code_tiled(kernel2, ArgTypes)

# Test 3: 2 * x
println("\n" * repeat("=", 50))
println("TEST 3: x -> 2 * x")
println(repeat("=", 50))

function kernel3(a::InputType, b::OutputType)
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    ct.mapreduce(x -> 2 * x, *, tile, 2)
end

ct.code_tiled(kernel3, ArgTypes)

# Test 4: sin(x) + 1
println("\n" * repeat("=", 50))
println("TEST 4: x -> sin(x) + 1")
println(repeat("=", 50))

function kernel4(a::InputType, b::OutputType)
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    ct.mapreduce(x -> sin(x) + 1, +, tile, 2)
end

ct.code_tiled(kernel4, ArgTypes)

# Test 5: abs(x - 1)
println("\n" * repeat("=", 50))
println("TEST 5: x -> abs(x - 1)")
println(repeat("=", 50))

function kernel5(a::InputType, b::OutputType)
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    ct.mapreduce(x -> abs(x - 1), max, tile, 2)
end

ct.code_tiled(kernel5, ArgTypes)

println("\n" * repeat("=", 70))
println("COMPLETE")
println("Look for 'reduce' and decomposed ops (addf, mulf, subf, etc.)")
println(repeat("=", 70) * "\n")
