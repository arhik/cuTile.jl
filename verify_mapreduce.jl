# verify_mapreduce.jl
# Comprehensive verification test for mapreduce functionality
# Tests both working cases (named functions) and documents lambda limitations
#
# Usage:
#   julia> include("verify_mapreduce.jl")

using cuTile
import cuTile as ct

println("\n" * repeat("=", 70))
println("MAPREDUCE COMPREHENSIVE VERIFICATION TEST")
println(repeat("=", 70))

# =============================================================================
# SECTION 1: Named Map Functions (All should work)
# =============================================================================
println("\n--- SECTION 1: Named Map Functions ---\n")

function test_named_map_functions()
    passed = 0

    # Test 1: identity + sum
    try
        tile = ct.Tile{Float32, (4, 16)}()
        result = ct.mapreduce(identity, +, tile, 2)
        @assert result isa ct.Tile
        println("  ✅ identity + sum")
        passed += 1
    catch e
        println("  ❌ identity + sum: $e")
    end

    # Test 2: abs + max
    try
        tile = ct.Tile{Float32, (8, 8)}()
        result = ct.mapreduce(abs, max, tile, 1)
        @assert result isa ct.Tile
        println("  ✅ abs + max")
        passed += 1
    catch e
        println("  ❌ abs + max: $e")
    end

    # Test 3: abs2 + sum
    try
        tile = ct.Tile{Float32, (4, 16)}()
        result = ct.mapreduce(abs2, +, tile, 2)
        @assert result isa ct.Tile
        println("  ✅ abs2 + sum")
        passed += 1
    catch e
        println("  ❌ abs2 + sum: $e")
    end

    # Test 4: sqrt + sum
    try
        tile = ct.Tile{Float32, (4, 16)}()
        result = ct.mapreduce(sqrt, +, tile, 2)
        @assert result isa ct.Tile
        println("  ✅ sqrt + sum")
        passed += 1
    catch e
        println("  ❌ sqrt + sum: $e")
    end

    # Test 5: exp + sum
    try
        tile = ct.Tile{Float32, (4, 16)}()
        result = ct.mapreduce(exp, +, tile, 2)
        @assert result isa ct.Tile
        println("  ✅ exp + sum")
        passed += 1
    catch e
        println("  ❌ exp + sum: $e")
    end

    # Test 6: log + sum
    try
        tile = ct.Tile{Float32, (4, 16)}()
        result = ct.mapreduce(log, +, tile, 2)
        @assert result isa ct.Tile
        println("  ✅ log + sum")
        passed += 1
    catch e
        println("  ❌ log + sum: $e")
    end

    # Test 7: sin + sum
    try
        tile = ct.Tile{Float32, (4, 16)}()
        result = ct.mapreduce(sin, +, tile, 2)
        @assert result isa ct.Tile
        println("  ✅ sin + sum")
        passed += 1
    catch e
        println("  ❌ sin + sum: $e")
    end

    # Test 8: cos + sum
    try
        tile = ct.Tile{Float32, (4, 16)}()
        result = ct.mapreduce(cos, +, tile, 2)
        @assert result isa ct.Tile
        println("  ✅ cos + sum")
        passed += 1
    catch e
        println("  ❌ cos + sum: $e")
    end

    # Test 9: neg (- unary) + sum
    try
        tile = ct.Tile{Float32, (4, 16)}()
        result = ct.mapreduce(-, +, tile, 2)
        @assert result isa ct.Tile
        println("  ✅ neg + sum")
        passed += 1
    catch e
        println("  ❌ neg + sum: $e")
    end

    println("\n  Section 1: $passed/9 tests passed")
    return passed
end

# =============================================================================
# SECTION 2: Reduce Functions (All should work)
# =============================================================================
println("\n--- SECTION 2: Reduce Functions ---\n")

function test_reduce_functions()
    passed = 0

    # Test 1: identity + (add)
    try
        tile = ct.Tile{Float32, (4, 16)}()
        result = ct.mapreduce(identity, +, tile, 2)
        @assert result isa ct.Tile
        println("  ✅ identity + (add)")
        passed += 1
    catch e
        println("  ❌ identity + (add): $e")
    end

    # Test 2: identity * (multiply)
    try
        tile = ct.Tile{Float32, (4, 16)}()
        result = ct.mapreduce(identity, *, tile, 2)
        @assert result isa ct.Tile
        println("  ✅ identity * (multiply)")
        passed += 1
    catch e
        println("  ❌ identity * (multiply): $e")
    end

    # Test 3: abs, max
    try
        tile = ct.Tile{Float32, (8, 8)}()
        result = ct.mapreduce(abs, max, tile, 1)
        @assert result isa ct.Tile
        println("  ✅ abs, max")
        passed += 1
    catch e
        println("  ❌ abs, max: $e")
    end

    # Test 4: abs, min
    try
        tile = ct.Tile{Float32, (8, 8)}()
        result = ct.mapreduce(abs, min, tile, 1)
        @assert result isa ct.Tile
        println("  ✅ abs, min")
        passed += 1
    catch e
        println("  ❌ abs, min: $e")
    end

    # Test 5: identity + (Int32)
    try
        tile = ct.Tile{Int32, (4, 16)}()
        result = ct.mapreduce(identity, +, tile, 2)
        @assert result isa ct.Tile
        println("  ✅ identity + (Int32)")
        passed += 1
    catch e
        println("  ❌ identity + (Int32): $e")
    end

    println("\n  Section 2: $passed/5 tests passed")
    return passed
end

# =============================================================================
# SECTION 3: Different Axes
# =============================================================================
println("\n--- SECTION 3: Different Reduction Axes ---\n")

function test_different_axes()
    passed = 0

    # Test 1: axis=1 (first dimension)
    try
        tile = ct.Tile{Float32, (4, 16)}()
        result = ct.mapreduce(identity, +, tile, 1)
        @assert result isa ct.Tile
        println("  ✅ axis=1 (first dimension)")
        passed += 1
    catch e
        println("  ❌ axis=1: $e")
    end

    # Test 2: axis=2 (second dimension)
    try
        tile = ct.Tile{Float32, (4, 16)}()
        result = ct.mapreduce(identity, +, tile, 2)
        @assert result isa ct.Tile
        println("  ✅ axis=2 (second dimension)")
        passed += 1
    catch e
        println("  ❌ axis=2: $e")
    end

    # Test 3: axis=3 (third dimension)
    try
        tile = ct.Tile{Float32, (2, 4, 8)}()
        result = ct.mapreduce(identity, +, tile, 3)
        @assert result isa ct.Tile
        println("  ✅ axis=3 (third dimension)")
        passed += 1
    catch e
        println("  ❌ axis=3: $e")
    end

    println("\n  Section 3: $passed/3 tests passed")
    return passed
end

# =============================================================================
# SECTION 4: Type Variations
# =============================================================================
println("\n--- SECTION 4: Type Variations ---\n")

function test_type_variations()
    passed = 0

    # Test 1: Float64
    try
        tile = ct.Tile{Float64, (4, 16)}()
        result = ct.mapreduce(identity, +, tile, 2)
        @assert result isa ct.Tile
        println("  ✅ Float64")
        passed += 1
    catch e
        println("  ❌ Float64: $e")
    end

    # Test 2: Int32
    try
        tile = ct.Tile{Int32, (4, 16)}()
        result = ct.mapreduce(identity, +, tile, 2)
        @assert result isa ct.Tile
        println("  ✅ Int32")
        passed += 1
    catch e
        println("  ❌ Int32: $e")
    end

    # Test 3: Int64
    try
        tile = ct.Tile{Int64, (4, 16)}()
        result = ct.mapreduce(identity, +, tile, 2)
        @assert result isa ct.Tile
        println("  ✅ Int64")
        passed += 1
    catch e
        println("  ❌ Int64: $e")
    end

    # Test 4: UInt32
    try
        tile = ct.Tile{UInt32, (4, 16)}()
        result = ct.mapreduce(identity, +, tile, 2)
        @assert result isa ct.Tile
        println("  ✅ UInt32")
        passed += 1
    catch e
        println("  ❌ UInt32: $e")
    end

    println("\n  Section 4: $passed/4 tests passed")
    return passed
end

# =============================================================================
# SECTION 5: Lambda Limitations (Documented)
# =============================================================================
println("\n--- SECTION 5: Lambda Limitations (Documentation) ---\n")

println("""
  NOTE: Anonymous function (lambda) limitations are documented in:
        docs/LAMBDA_LIMITATION.md

  The following demonstrates the recommended workarounds.
""")

function test_workarounds()
    passed = 0
    tile = ct.Tile{Float32, (4, 16)}()

    # Workaround 1: Named function instead of lambda
    try
        add_one(x) = x + 1.0f0
        result = ct.mapreduce(add_one, +, tile, 2)
        @assert result isa ct.Tile
        println("  ✅ Named function workaround: add_one(x)")
        passed += 1
    catch e
        println("  ❌ Named function workaround: $e")
    end

    # Workaround 2: Named function for square
    try
        square(x) = x * x
        result = ct.mapreduce(square, +, tile, 2)
        @assert result isa ct.Tile
        println("  ✅ Named function workaround: square(x)")
        passed += 1
    catch e
        println("  ❌ Named function workaround: $e")
    end

    # Workaround 3: Built-in abs2 (more efficient than square)
    try
        result = ct.mapreduce(abs2, +, tile, 2)
        @assert result isa ct.Tile
        println("  ✅ Built-in workaround: abs2")
        passed += 1
    catch e
        println("  ❌ Built-in workaround: $e")
    end

    println("\n  Section 5: $passed/3 tests passed")
    return passed
end

# =============================================================================
# Main Test Runner
# =============================================================================

# Run all tests
if abspath(PROGRAM_FILE) == @__FILE__
    println("\n" * repeat("=", 70))
    println("MAPREDUCE COMPREHENSIVE VERIFICATION TEST")
    println(repeat("=", 70) * "\n")

    total_passed = 0
    total_tests = 24

    println("Running Section 1: Named Map Functions...")
    total_passed += test_named_map_functions()

    println("\nRunning Section 2: Reduce Functions...")
    total_passed += test_reduce_functions()

    println("\nRunning Section 3: Different Reduction Axes...")
    total_passed += test_different_axes()

    println("\nRunning Section 4: Type Variations...")
    total_passed += test_type_variations()

    println("\nRunning Section 5: Lambda Workarounds...")
    total_passed += test_workarounds()

    println("\n" * repeat("=", 70))
    println("VERIFICATION SUMMARY")
    println(repeat("=", 70))
    println("  Passed:  $total_passed / $total_tests")
    success_rate = round(Int, 100 * total_passed / total_tests)
    println("  Success: $success_rate%")
    println(repeat("=", 70))

    if total_passed == total_tests
        println("\n✅ All tests passed!")
    else
        println("\n⚠️  Some tests failed. Check output above.")
    end

    println("\nFor lambda limitation details, see: docs/LAMBDA_LIMITATION.md")
    println("For mapreduce API details, see: docs/MAPREDUCE_IMPLEMENTATION.md")
    println(repeat("=", 70) * "\n")
end
