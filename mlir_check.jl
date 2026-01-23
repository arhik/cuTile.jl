# mlir_check.jl
# MLIR verification test for mapreduce functionality
# Tests that mapreduce compiles to correct Tile IR bytecode with named functions
#
# NOTE: This test uses NAMED FUNCTIONS only because anonymous functions (lambdas)
# cannot be compiled to Tile IR. This is a fundamental limitation of Julia's
# closure compilation. See docs/LAMBDA_LIMITATION.md for details.
#
# Usage:
#   julia> include("mlir_check.jl")

using cuTile
import cuTile as ct

println("\n" * repeat("=", 70))
println("MLIR VERIFICATION TEST - MapReduce Bytecode Generation")
println(repeat("=", 70))

println("""
This test verifies that mapreduce compiles to valid Tile IR bytecode.
All tests use NAMED FUNCTIONS (the documented workaround for lambda limitations).
""")

# Define ArraySpecs
const spec1d = ct.ArraySpec{1}(128, true)
const spec2d = ct.ArraySpec{2}(128, true)

# ============================================================================
# SECTION 1: Named Map Functions (built-in)
# ============================================================================

println("\n" * repeat("=", 60))
println("SECTION 1: Named Map Functions")
println(repeat("=", 60))

# Test 1a: identity (baseline - no mapping)
println("\n1a. identity (no-op mapping)")
println("    Expected: Direct element use in reduction")
function test1a(a::ct.TileArray{Float32, 2, spec2d}, b::ct.TileArray{Float32, 1, spec1d})
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    ct.mapreduce(identity, +, tile, 2)
    return
end
ct.code_tiled(test1a, Tuple{ct.TileArray{Float32, 2, spec2d}, ct.TileArray{Float32, 1, spec1d}})

# Test 1b: abs
println("\n1b. abs (absolute value)")
println("    Expected: cuda_tile.absf operation")
function test1b(a::ct.TileArray{Float32, 2, spec2d}, b::ct.TileArray{Float32, 1, spec1d})
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    ct.mapreduce(abs, +, tile, 2)
    return
end
ct.code_tiled(test1b, Tuple{ct.TileArray{Float32, 2, spec2d}, ct.TileArray{Float32, 1, spec1d}})

# Test 1c: abs2 (square)
println("\n1c. abs2 (x*x, square)")
println("    Expected: cuda_tile.mulf(elem, elem)")
function test1c(a::ct.TileArray{Float32, 2, spec2d}, b::ct.TileArray{Float32, 1, spec1d})
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    ct.mapreduce(abs2, +, tile, 2)
    return
end
ct.code_tiled(test1c, Tuple{ct.TileArray{Float32, 2, spec2d}, ct.TileArray{Float32, 1, spec1d}})

# Test 1d: sqrt
println("\n1d. sqrt (square root)")
println("    Expected: cuda_tile.sqrtf operation")
function test1d(a::ct.TileArray{Float32, 2, spec2d}, b::ct.TileArray{Float32, 1, spec1d})
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    ct.mapreduce(sqrt, +, tile, 2)
    return
end
ct.code_tiled(test1d, Tuple{ct.TileArray{Float32, 2, spec2d}, ct.TileArray{Float32, 1, spec1d}})

# Test 1e: sin
println("\n1e. sin (sine)")
println("    Expected: cuda_tile.sinf operation")
function test1e(a::ct.TileArray{Float32, 2, spec2d}, b::ct.TileArray{Float32, 1, spec1d})
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    ct.mapreduce(sin, +, tile, 2)
    return
end
ct.code_tiled(test1e, Tuple{ct.TileArray{Float32, 2, spec2d}, ct.TileArray{Float32, 1, spec1d}})

# ============================================================================
# SECTION 2: User-Defined Named Functions (lambda workaround)
# ============================================================================
println("\n" * repeat("=", 60))
println("SECTION 2: User-Defined Named Functions (Lambda Workaround)")
println(repeat("=", 60))

# Define named functions to replace lambdas
square(x) = x * x
add_one(x) = x + 1.0f0
mul_two(x) = 2.0f0 * x

# Test 2a: square (replaces x -> x * x)
println("\n2a. square(x) = x * x")
println("    Expected: cuda_tile.mulf(elem, elem)")
println("    This replaces the lambda: x -> x * x")
function test2a(a::ct.TileArray{Float32, 2, spec2d}, b::ct.TileArray{Float32, 1, spec1d})
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    ct.mapreduce(square, +, tile, 2)
    return
end
ct.code_tiled(test2a, Tuple{ct.TileArray{Float32, 2, spec2d}, ct.TileArray{Float32, 1, spec1d}})

# Test 2b: add_one (replaces x -> x + 1)
println("\n2b. add_one(x) = x + 1.0")
println("    Expected: cuda_tile.addf(elem, const)")
println("    This replaces the lambda: x -> x + 1")
function test2b(a::ct.TileArray{Float32, 2, spec2d}, b::ct.TileArray{Float32, 1, spec1d})
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    ct.mapreduce(add_one, +, tile, 2)
    return
end
ct.code_tiled(test2b, Tuple{ct.TileArray{Float32, 2, spec2d}, ct.TileArray{Float32, 1, spec1d}})

# Test 2c: mul_two (replaces x -> 2 * x)
println("\n2c. mul_two(x) = 2.0 * x")
println("    Expected: cuda_tile.mulf(const, elem)")
println("    This replaces the lambda: x -> 2 * x")
function test2c(a::ct.TileArray{Float32, 2, spec2d}, b::ct.TileArray{Float32, 1, spec1d})
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    ct.mapreduce(mul_two, *, tile, 2)
    return
end
ct.code_tiled(test2c, Tuple{ct.TileArray{Float32, 2, spec2d}, ct.TileArray{Float32, 1, spec1d}})

# ============================================================================
# SECTION 3: Reduce Operations
# ============================================================================
println("\n" * repeat("=", 60))
println("SECTION 3: Reduce Operations")
println(repeat("=", 60))

# Test 3a: Addition
println("\n3a. identity + add (sum)")
println("    Expected: cuda_tile.addf in reduce body")
function test3a(a::ct.TileArray{Float32, 2, spec2d}, b::ct.TileArray{Float32, 1, spec1d})
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    ct.mapreduce(identity, +, tile, 2)
    return
end
ct.code_tiled(test3a, Tuple{ct.TileArray{Float32, 2, spec2d}, ct.TileArray{Float32, 1, spec1d}})

# Test 3b: Multiplication
println("\n3b. abs2 * multiply (product)")
println("    Expected: cuda_tile.mulf in reduce body")
function test3b(a::ct.TileArray{Float32, 2, spec2d}, b::ct.TileArray{Float32, 1, spec1d})
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    ct.mapreduce(abs2, *, tile, 2)
    return
end
ct.code_tiled(test3b, Tuple{ct.TileArray{Float32, 2, spec2d}, ct.TileArray{Float32, 1, spec1d}})

# Test 3c: Max
println("\n3c. abs + max (maximum)")
println("    Expected: cuda_tile.maxf in reduce body")
function test3c(a::ct.TileArray{Float32, 2, spec2d}, b::ct.TileArray{Float32, 1, spec1d})
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    ct.mapreduce(abs, max, tile, 2)
    return
end
ct.code_tiled(test3c, Tuple{ct.TileArray{Float32, 2, spec2d}, ct.TileArray{Float32, 1, spec1d}})

# Test 3d: Min
println("\n3d. identity + min (minimum)")
println("    Expected: cuda_tile.minf in reduce body")
function test3d(a::ct.TileArray{Float32, 2, spec2d}, b::ct.TileArray{Float32, 1, spec1d})
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    ct.mapreduce(identity, min, tile, 2)
    return
end
ct.code_tiled(test3d, Tuple{ct.TileArray{Float32, 2, spec2d}, ct.TileArray{Float32, 1, spec1d}})

# ============================================================================
# SECTION 4: Different Axes
# ============================================================================
println("\n" * repeat("=", 60))
println("SECTION 4: Different Reduction Axes")
println(repeat("=", 60))

# Test 4a: Axis 1 (first dimension)
println("\n4a. axis=1 (reduce first dimension)")
println("    Expected: (4, 16) -> (16)")
function test4a(a::ct.TileArray{Float32, 2, spec2d}, b::ct.TileArray{Float32, 1, spec1d})
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    ct.mapreduce(identity, +, tile, 1)
    return
end
ct.code_tiled(test4a, Tuple{ct.TileArray{Float32, 2, spec2d}, ct.TileArray{Float32, 1, spec1d}})

# Test 4b: Axis 2 (second dimension)
println("\n4b. axis=2 (reduce second dimension)")
println("    Expected: (4, 16) -> (4)")
function test4b(a::ct.TileArray{Float32, 2, spec2d}, b::ct.TileArray{Float32, 1, spec1d})
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    ct.mapreduce(identity, +, tile, 2)
    return
end
ct.code_tiled(test4b, Tuple{ct.TileArray{Float32, 2, spec2d}, ct.TileArray{Float32, 1, spec1d}})

# ============================================================================
# SECTION 5: Type Variations
# ============================================================================
println("\n" * repeat("=", 60))
println("SECTION 5: Type Variations")
println(repeat("=", 60))

# Test 5a: Float64
println("\n5a. Float64 type")
println("    Expected: Float64 operations in bytecode")
function test5a(a::ct.TileArray{Float64, 2, spec2d}, b::ct.TileArray{Float64, 1, spec1d})
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    ct.mapreduce(identity, +, tile, 2)
    return
end
ct.code_tiled(test5a, Tuple{ct.TileArray{Float64, 2, spec2d}, ct.TileArray{Float64, 1, spec1d}})

# Test 5b: Int32
println("\n5b. Int32 type")
println("    Expected: Integer operations with signedness")
function test5b(a::ct.TileArray{Int32, 2, spec2d}, b::ct.TileArray{Int32, 1, spec1d})
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    ct.mapreduce(identity, +, tile, 2)
    return
end
ct.code_tiled(test5b, Tuple{ct.TileArray{Int32, 2, spec2d}, ct.TileArray{Int32, 1, spec1d}})

# ============================================================================
# SECTION 6: Combined Operations (real-world patterns)
# ============================================================================
println("\n" * repeat("=", 60))
println("SECTION 6: Combined Operations (Real-World Patterns)")
println(repeat("=", 60))

# Real pattern: sum of squared differences
square_diff(x, mean) = (x - mean) * (x - mean)
sqdiff(x) = square_diff(x, 0.0f0)  # Simplified version

println("\n6a. Sum of squares (sqdiff)")
println("    Real pattern: Σ(x²)")
function test6a(a::ct.TileArray{Float32, 2, spec2d}, b::ct.TileArray{Float32, 1, spec1d})
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    ct.mapreduce(sqdiff, +, tile, 2)
    return
end
ct.code_tiled(test6a, Tuple{ct.TileArray{Float32, 2, spec2d}, ct.TileArray{Float32, 1, spec1d}})

# Real pattern: max absolute deviation
abs_dev(x) = abs(x)
println("\n6b. Max absolute deviation (abs_dev)")
println("    Real pattern: max(|x|)")
function test6b(a::ct.TileArray{Float32, 2, spec2d}, b::ct.TileArray{Float32, 1, spec1d})
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    ct.mapreduce(abs_dev, max, tile, 2)
    return
end
ct.code_tiled(test6b, Tuple{ct.TileArray{Float32, 2, spec2d}, ct.TileArray{Float32, 1, spec1d}})

# ============================================================================
# Summary
# ============================================================================

println("\n" * repeat("=", 70))
println("MLIR VERIFICATION COMPLETE")
println(repeat("=", 70))

println("""
Check the output above for the generated Tile IR bytecode.
Look for:
  - cuda_tile.reduce operation (the main reduction)
  - Map operations in body: absf, sqrtf, mulf, addf, etc.
  - Reduce operations in body: addf, mulf, maxf, minf, etc.
  - Correct tile shapes (reduced dimension should be 1)

For lambda limitation details, see: docs/LAMBDA_LIMITATION.md
For mapreduce API details, see: docs/MAPREDUCE_IMPLEMENTATION.md
""")

println(repeat("=", 70) * "\n")
