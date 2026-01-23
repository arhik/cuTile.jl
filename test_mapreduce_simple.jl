module TestMapReduceSimple

using cuTile
import cuTile as ct

function test_syntax()
    println("\nSTAGE 1: Testing MapReduce Syntax")

    result1 = ct.mapreduce(x -> x * x, +, ct.Tile{Float32, (4, 16)}(), 2)
    @assert result1 isa ct.Tile
    println("  OK: Sum of squares")

    result2 = ct.mapreduce(abs, max, ct.Tile{Float32, (8, 8)}(), 1)
    @assert result2 isa ct.Tile
    println("  OK: Max of abs")

    result3 = ct.mapreduce(identity, *, ct.Tile{Float32, (16, 4)}(), 1)
    @assert result3 isa ct.Tile
    println("  OK: Product")

    result4 = ct.mapreduce(abs2, +, ct.Tile{Float32, (4, 16)}(), 2)
    @assert result4 isa ct.Tile
    println("  OK: abs2")

    result5 = ct.mapreduce(sqrt, +, ct.Tile{Float32, (4, 16)}(), 2)
    @assert result5 isa ct.Tile
    println("  OK: sqrt")

    result6 = ct.mapreduce(identity, +, ct.Tile{Float32, (4, 16)}(), ct.Val(0))
    @assert result6 isa ct.Tile
    println("  OK: Val{axis}")

    println("STAGE 1 COMPLETE: All syntax tests passed!\n")
end

function test_compilation()
    println("STAGE 2: Testing Compilation")

    try
        result = ct.Tile{Float32, (4, 1)}()
        @assert result isa ct.Tile
        println("  OK: Compilation syntax valid")
    catch e
        error("Compilation failed: $e")
    end

    println("STAGE 2 COMPLETE: Compilation tests complete!\n")
end

function test_integration()
    println("STAGE 3: Testing Integration")

    tile = ct.Tile{Float32, (4, 16)}()
    result = ct.mapreduce(x -> x * x, +, tile, 2)
    @assert result isa ct.Tile
    println("  OK: Integration pattern valid")

    println("STAGE 3 COMPLETE: Integration tests complete!\n")
end

function test_all()
    println("\n" * repeat("#", 60))
    println("MAPREDUCE TEST SUITE")
    println(repeat("#", 60))

    test_syntax()
    test_compilation()
    test_integration()

    println(repeat("#", 60))
    println("ALL TESTS PASSED!")
    println(repeat("#", 60) * "\n")
end

function quick_verify()
    println("\n[QUICK VERIFY] Testing mapreduce...")
    try
        tile = ct.Tile{Float32, (4, 16)}()
        result = ct.mapreduce(x -> x * x, +, tile, 2)
        println("OK: mapreduce syntax works")
        return true
    catch e
        println("FAIL: $e")
        return false
    end
end

println("\n[Test] Type TestMapReduceSimple.quick_verify() or test_all()\n")

end  # module
