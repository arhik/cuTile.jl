module TestDecomposition

using cuTile
import cuTile as ct

function test_simple()
    println("\n=== Simple Expressions ===")
    tile = ct.Tile{Float32, (4, 16)}()

    println("1. x + 1: ")
    try
        r = ct.mapreduce(x -> x + 1, +, tile, 2)
        @assert r isa ct.Tile
        println("   OK")
    catch e
        println("   FAIL: $e")
    end

    println("2. 2 * x: ")
    try
        r = ct.mapreduce(x -> 2 * x, *, tile, 1)
        @assert r isa ct.Tile
        println("   OK")
    catch e
        println("   FAIL: $e")
    end

    println("3. x - 5: ")
    try
        r = ct.mapreduce(x -> x - 5, +, tile, 2)
        @assert r isa ct.Tile
        println("   OK")
    catch e
        println("   FAIL: $e")
    end

    println("4. x / 2: ")
    try
        r = ct.mapreduce(x -> x / 2, *, tile, 1)
        @assert r isa ct.Tile
        println("   OK")
    catch e
        println("   FAIL: $e")
    end

    println("5. 1 + x: ")
    try
        r = ct.mapreduce(x -> 1 + x, +, tile, 2)
        @assert r isa ct.Tile
        println("   OK")
    catch e
        println("   FAIL: $e")
    end
end

function test_composite()
    println("\n=== Composite Expressions ===")
    tile = ct.Tile{Float32, (4, 16)}()

    println("1. x^2: ")
    try
        r = ct.mapreduce(x -> x^2, +, tile, 2)
        @assert r isa ct.Tile
        println("   OK")
    catch e
        println("   FAIL: $e")
    end

    println("2. (x + 1) * 2: ")
    try
        r = ct.mapreduce(x -> (x + 1) * 2, *, tile, 1)
        @assert r isa ct.Tile
        println("   OK")
    catch e
        println("   FAIL: $e")
    end

    println("3. 2 * x + 3: ")
    try
        r = ct.mapreduce(x -> 2 * x + 3, +, tile, 2)
        @assert r isa ct.Tile
        println("   OK")
    catch e
        println("   FAIL: $e")
    end

    println("4. (x - 1) / 2: ")
    try
        r = ct.mapreduce(x -> (x - 1) / 2, *, tile, 1)
        @assert r isa ct.Tile
        println("   OK")
    catch e
        println("   FAIL: $e")
    end
end

function test_function_composition()
    println("\n=== Function Compositions ===")
    tile = ct.Tile{Float32, (4, 16)}()

    println("1. sin(x) + 1: ")
    try
        r = ct.mapreduce(x -> sin(x) + 1, +, tile, 2)
        @assert r isa ct.Tile
        println("   OK")
    catch e
        println("   FAIL: $e")
    end

    println("2. abs(x - 1): ")
    try
        r = ct.mapreduce(x -> abs(x - 1), max, tile, 1)
        @assert r isa ct.Tile
        println("   OK")
    catch e
        println("   FAIL: $e")
    end

    println("3. sqrt(x) + cos(x): ")
    try
        r = ct.mapreduce(x -> sqrt(x) + cos(x), +, tile, 2)
        @assert r isa ct.Tile
        println("   OK")
    catch e
        println("   FAIL: $e")
    end

    println("4. exp(x - 1): ")
    try
        r = ct.mapreduce(x -> exp(x - 1), *, tile, 1)
        @assert r isa ct.Tile
        println("   OK")
    catch e
        println("   FAIL: $e")
    end

    println("5. sin(x) * cos(x): ")
    try
        r = ct.mapreduce(x -> sin(x) * cos(x), +, tile, 2)
        @assert r isa ct.Tile
        println("   OK")
    catch e
        println("   FAIL: $e")
    end
end

function test_all()
    println("\n" * repeat("=", 60))
    println("MAPREDUCE EXPRESSION DECOMPOSITION TESTS")
    println(repeat("=", 60))

    test_simple()
    test_composite()
    test_function_composition()

    println("\n" * repeat("=", 60))
    println("TESTS COMPLETE")
    println(repeat("=", 60) * "\n")
end

function quick()
    println("\n[Quick Test] x + 1")
    try
        tile = ct.Tile{Float32, (4, 16)}()
        r = ct.mapreduce(x -> x + 1, +, tile, 2)
        println("OK")
        return true
    catch e
        println("FAIL: $e")
        return false
    end
end

println("\nType TestDecomposition.test_all() or quick()\n")

end  # module
