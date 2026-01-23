"""
    load_test.jl

Simple test to verify package loading and basic mapreduce functionality.
"""
module LoadTest

using cuTile
import cuTile as ct

function test_package_loading()
    println("Testing package loading...")

    # Create test tile
    tile = ct.Tile{Float32, (4, 16)}()

    # Test basic mapreduce
    println("1. Testing basic mapreduce (identity +)")
    result = ct.mapreduce(identity, +, tile, 2)
    println("   OK: identity + works")

    # Test expression decomposition
    println("2. Testing expression decomposition (x + 1)")
    result = ct.mapreduce(x -> x + 1, +, tile, 2)
    println("   OK: x + 1 works")

    println("3. Testing expression decomposition (2 * x)")
    result = ct.mapreduce(x -> 2 * x, *, tile, 1)
    println("   OK: 2 * x works")

    println("4. Testing function composition (sin(x) + 1)")
    result = ct.mapreduce(x -> sin(x) + 1, +, tile, 2)
    println("   OK: sin(x) + 1 works")

    println("5. Testing power expression (x^2)")
    result = ct.mapreduce(x -> x^2, +, tile, 2)
    println("   OK: x^2 works")

    println("6. Testing composite ((x + 1) * 2)")
    result = ct.mapreduce(x -> (x + 1) * 2, *, tile, 1)
    println("   OK: (x + 1) * 2 works")

    println("7. Testing abs composition (abs(x - 1))")
    result = ct.mapreduce(x -> abs(x - 1), max, tile, 1)
    println("   OK: abs(x - 1) works")

    println("\n" * repeat("=", 50))
    println("All tests passed!")
    println(repeat("=", 50) * "\n")
end

# Run tests
test_package_loading()

end  # module
