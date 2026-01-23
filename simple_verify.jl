
# simple_verify.jl
# Simple verification test for mapreduce functionality
# Usage: julia> include("simple_verify.jl")

using cuTile
import cuTile as ct

println("\n" * repeat("=", 70))
println("SIMPLE MAPREDUCE VERIFICATION TEST")
println(repeat("=", 70))

passed = 0
total = 0

# Test 1: identity + sum
println("\n1. identity + sum")
global total += 1
try
    tile = ct.Tile{Float32, (4, 16)}()
    result = ct.mapreduce(identity, +, tile, 2)
    @assert result isa ct.Tile
    println("   ✅ PASS")
    global passed += 1
catch e
    println("   ❌ FAIL: $e")
end

# Test 2: abs + sum
println("\n2. abs + sum")
global total += 1
try
    tile = ct.Tile{Float32, (4, 16)}()
    result = ct.mapreduce(abs, +, tile, 2)
    @assert result isa ct.Tile
    println("   ✅ PASS")
    global passed += 1
catch e
    println("   ❌ FAIL: $e")
end

# Test 3: abs2 + sum
println("\n3. abs2 + sum")
global total += 1
try
    tile = ct.Tile{Float32, (4, 16)}()
    result = ct.mapreduce(abs2, +, tile, 2)
    @assert result isa ct.Tile
    println("   ✅ PASS")
    global passed += 1
catch e
    println("   ❌ FAIL: $e")
end

# Test 4: sqrt + sum
println("\n4. sqrt + sum")
global total += 1
try
    tile = ct.Tile{Float32, (4, 16)}()
    result = ct.mapreduce(sqrt, +, tile, 2)
    @assert result isa ct.Tile
    println("   ✅ PASS")
    global passed += 1
catch e
    println("   ❌ FAIL: $e")
end

# Test 5: sin + sum
println("\n5. sin + sum")
global total += 1
try
    tile = ct.Tile{Float32, (4, 16)}()
    result = ct.mapreduce(sin, +, tile, 2)
    @assert result isa ct.Tile
    println("   ✅ PASS")
    global passed += 1
catch e
    println("   ❌ FAIL: $e")
end

# Test 6: abs + max
println("\n6. abs + max")
global total += 1
try
    tile = ct.Tile{Float32, (4, 16)}()
    result = ct.mapreduce(abs, max, tile, 1)
    @assert result isa ct.Tile
    println("   ✅ PASS")
    global passed += 1
catch e
    println("   ❌ FAIL: $e")
end

# Test 7: identity + min
println("\n7. identity + min")
global total += 1
try
    tile = ct.Tile{Float32, (4, 16)}()
    result = ct.mapreduce(identity, min, tile, 1)
    @assert result isa ct.Tile
    println("   ✅ PASS")
    global passed += 1
catch e
    println("   ❌ FAIL: $e")
end

# Test 8: identity * product
println("\n8. identity * product")
global total += 1
try
    tile = ct.Tile{Float32, (4, 16)}()
    result = ct.mapreduce(identity, *, tile, 2)
    @assert result isa ct.Tile
    println("   ✅ PASS")
    global passed += 1
catch e
    println("   ❌ FAIL: $e")
end

# Test 9: axis=2 (second dimension)
println("\n9. axis=2 (second dimension)")
global total += 1
try
    tile = ct.Tile{Float32, (4, 16)}()
    result = ct.mapreduce(identity, +, tile, 2)
    @assert result isa ct.Tile
    println("   ✅ PASS")
    global passed += 1
catch e
    println("   ❌ FAIL: $e")
end

# Test 10: axis=1 (first dimension)
println("\n10. axis=1 (first dimension)")
global total += 1
try
    tile = ct.Tile{Float32, (4, 16)}()
    result = ct.mapreduce(identity, +, tile, 1)
    @assert result isa ct.Tile
    println("   ✅ PASS")
    global passed += 1
catch e
    println("   ❌ FAIL: $e")
end

# Test 11: Float64 type
println("\n11. Float64 type")
global total += 1
try
    tile = ct.Tile{Float64, (4, 16)}()
    result = ct.mapreduce(identity, +, tile, 2)
    @assert result isa ct.Tile
    println("   ✅ PASS")
    global passed += 1
catch e
    println("   ❌ FAIL: $e")
end

# Test 12: Int32 type
println("\n12. Int32 type")
global total += 1
try
    tile = ct.Tile{Int32, (4, 16)}()
    result = ct.mapreduce(identity, +, tile, 2)
    @assert result isa ct.Tile
    println("   ✅ PASS")
    global passed += 1
catch e
    println("   ❌ FAIL: $e")
end

# ============================================================================
# Named function workarounds (for lambda limitation)
# ============================================================================

# Define named functions
add_one(x) = x + 1.0f0
square(x) = x * x
double(x) = 2.0f0 * x

println("\n--- NAMED FUNCTION WORKAROUNDS ---")

# Test 13: Named function - add_one
println("\n13. Named: add_one(x) = x + 1")
global total += 1
try
    tile = ct.Tile{Float32, (4, 16)}()
    result = ct.mapreduce(add_one, +, tile, 2)
    @assert result isa ct.Tile
    println("   ✅ PASS")
    global passed += 1
catch e
    println("   ❌ FAIL: $e")
end

# Test 14: Named function - square
println("\n14. Named: square(x) = x * x")
global total += 1
try
    tile = ct.Tile{Float32, (4, 16)}()
    result = ct.mapreduce(square, +, tile, 2)
    @assert result isa ct.Tile
    println("   ✅ PASS")
    global passed += 1
catch e
    println("   ❌ FAIL: $e")
end

# Test 15: Named function - double
println("\n15. Named: double(x) = 2 * x")
global total += 1
try
    tile = ct.Tile{Float32, (4, 16)}()
    result = ct.mapreduce(double, *, tile, 2)
    @assert result isa ct.Tile
    println("   ✅ PASS")
    global passed += 1
catch e
    println("   ❌ FAIL: $e")
end

# Test 16: Built-in abs2 (optimized)
println("\n16. Built-in abs2 (optimized)")
global total += 1
try
    tile = ct.Tile{Float32, (4, 16)}()
    result = ct.mapreduce(abs2, +, tile, 2)
    @assert result isa ct.Tile
    println("   ✅ PASS")
    global passed += 1
catch e
    println("   ❌ FAIL: $e")
end

# ============================================================================
# Summary
# ============================================================================

println("\n" * repeat("=", 70))
println("VERIFICATION SUMMARY")
println(repeat("=", 70))
println("   Passed: $passed / $total")
success_rate = round(Int, 100 * passed / total)
println("   Success: $success_rate%")
println(repeat("=", 70))

if passed == total
    println("\n✅ ALL TESTS PASSED!")
else
    println("\n⚠️  Some tests failed. Review output above.")
end

println("\nFor lambda limitation details, see: docs/LAMBDA_LIMITATION.md")
println("For mapreduce API details, see: docs/MAPREDUCE_IMPLEMENTATION.md")
println(repeat("=", 70) * "\n")
