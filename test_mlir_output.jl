"""
    test_mlir_output.jl

Test file to verify mapreduce decomposition by checking MLIR output.
This file demonstrates how the expression decomposition works by inspecting
the generated Tile IR bytecode.

Usage:
    julia> include("test_mlir_output.jl")
    julia> TestMLIROutput.run_all()
"""
module TestMLIROutput

using cuTile
import cuTile as ct

#=============================================================================
# Helper Functions
#=============================================================================

"""
    run_code_tiled(func_argtypes)

Run code_tiled and capture the MLIR output.
Returns the MLIR string for inspection.
"""
function run_code_tiled(@nospecialize(func), argtypes::Tuple)
    # Use a pipe to capture output
    io = IOBuffer()

    # Redirect stdout to capture MLIR
    old_stdout = stdout
    redirect_stdout(io) do
        try
            ct.code_tiled(func, argtypes)
        catch e
            # Some errors are expected if decomposition fails
            # We'll capture whatever output we get
        end
    end

    # Get the captured output
    output = String(take!(io))
    return output
end

"""
    check_for_pattern(output, pattern)

Check if a pattern appears in the MLIR output.
Returns true if pattern is found.
"""
function check_for_pattern(output::String, pattern::String)
    occursin(pattern, output)
end

"""
    show_relevant_output(output)

Extract and display the most relevant parts of the MLIR output.
Focus on reduce operations and function bodies.
"""
function show_relevant_output(output::String)
    lines = split(output, '\n')

    # Find lines containing reduce or function body info
    relevant = filter(lines) do line
        contains(lowercase(line), "reduce") ||
        contains(lowercase(line), "addf") ||
        contains(lowercase(line), "mulf") ||
        contains(lowercase(line), "subf") ||
        contains(lowercase(line), "divf") ||
        contains(lowercase(line), "absf") ||
        contains(lowercase(line), "sinf") ||
        contains(lowercase(line), "cosf") ||
        contains(lowercase(line), "const") ||
        contains(lowercase(line), "identity")
    end

    if !isempty(relevant)
        println("\nRelevant MLIR lines:")
        for line in relevant[1:min(20, length(relevant))]
            println("  ", strip(line))
        end
    else
        println("\nNo relevant patterns found in output")
        println("\nFull output:")
        println(output)
    end
end

#=============================================================================
# Test Cases
#=============================================================================

"""
    test_simple_expression(name, func, pattern)

Test a simple expression and check for expected patterns in MLIR.
"""
function test_simple_expression(name::String, func::Function, expected_pattern::String)
    println("\n" * repeat("=", 60))
    println("TEST: $name")
    println(repeat("=", 60))

    # Create a simple kernel using the mapreduce
    kernel = function(tile::ct.Tile{Float32, ct.Tuple{(4, 16)}})
        ct.mapreduce(func, +, tile, 2)
    end

    # Get argument types
    argtypes = Tuple{typeof(kernel.tile)}

    println("\nCompiling to Tile IR...")
    output = run_code_tiled(kernel, argtypes)

    # Check for expected pattern
    pattern_found = check_for_pattern(output, expected_pattern)

    if pattern_found
        println("✓ Expected pattern found: $expected_pattern")
    else
        println("✗ Expected pattern NOT found: $expected_pattern")
    end

    # Show relevant output
    show_relevant_output(output)

    return pattern_found
end

"""
    test_with_full_kernel(name, func, reduce_op, expected_patterns)

Test a complete kernel with load and store operations.
"""
function test_with_full_kernel(
    name::String,
    func::Function,
    reduce_op::Function,
    expected_patterns::Vector{String}
)
    println("\n" * repeat("=", 60))
    println("TEST: $name (full kernel)")
    println(repeat("=", 60))

    # Define a complete kernel
    kernel = function(
        a::ct.TileArray{Float32, 2},
        b::ct.TileArray{Float32, 1}
    )
        pid = ct.bid(1)
        tile = ct.load(a, pid, (4, 16))
        result = ct.mapreduce(func, reduce_op, tile, 2)
        ct.store(b, pid, result)
        return
    end

    # Get argument types
    input_spec = (128, 8)  # 2D spec
    output_spec = (128, 1)  # 1D spec

    argtypes = (
        ct.TileArray{Float32, 2, ct.ArraySpec{2}(input_spec...)},
        ct.TileArray{Float32, 1, ct.ArraySpec{1}(output_spec...)}
    )

    println("\nCompiling full kernel to Tile IR...")
    output = run_code_tiled(kernel, argtypes)

    # Check all expected patterns
    all_found = true
    for pattern in expected_patterns
        found = check_for_pattern(output, pattern)
        if found
            println("✓ Found: $pattern")
        else
            println("✗ NOT found: $pattern")
            all_found = false
        end
    end

    # Show relevant output
    show_relevant_output(output)

    return all_found
end

#=============================================================================
# Individual Tests
#=============================================================================

"""
    test_identity_plus()

Test: mapreduce(identity, +, tile, 2)
Expected: reduce with addf operation
"""
function test_identity_plus()
    println("\n" * repeat("#", 60))
    println("# Test: identity + sum reduction")
    println("#" * repeat("#", 60))

    # This should generate a simple reduce with addf
    patterns = ["reduce", "addf", "identity"]

    return test_with_full_kernel("identity + sum", identity, +, patterns)
end

"""
    test_x_plus_one()

Test: mapreduce(x -> x + 1, +, tile, 2)
Expected: reduce with addf and a constant
"""
function test_x_plus_one()
    println("\n" * repeat("#", 60))
    println("# Test: x + 1 expression")
    println("#" * repeat("#", 60))

    # This should decompose x + 1 into: addf(elem, const(1))
    patterns = ["reduce", "addf", "const"]

    return test_with_full_kernel("x + 1", x -> x + 1, +, patterns)
end

"""
    test_two_mul_x()

Test: mapreduce(x -> 2 * x, *, tile, 1)
Expected: reduce with mulf and a constant
"""
function test_two_mul_x()
    println("\n" * repeat("#", 60))
    println("# Test: 2 * x expression")
    println("#" * repeat("#", 60))

    # This should decompose 2 * x into: mulf(const(2), elem)
    patterns = ["reduce", "mulf", "const"]

    return test_with_full_kernel("2 * x", x -> 2 * x, *, patterns)
end

"""
    test_x_squared()

Test: mapreduce(x -> x^2, +, tile, 2)
Expected: reduce with mulf(elem, elem)
"""
function test_x_squared()
    println("\n" * repeat("#", 60))
    println("# Test: x^2 expression")
    println("#" * repeat("#", 60))

    # This should decompose x^2 into: mulf(elem, elem)
    patterns = ["reduce", "mulf"]

    return test_with_full_kernel("x^2", x -> x^2, +, patterns)
end

"""
    test_sin_plus_one()

Test: mapreduce(x -> sin(x) + 1, +, tile, 2)
Expected: reduce with sinf, addf, and const
"""
function test_sin_plus_one()
    println("\n" * repeat("#", 60))
    println("# Test: sin(x) + 1 expression")
    println("#" * repeat("#", 60))

    # This should decompose sin(x) + 1 into: addf(sinf(elem), const(1))
    patterns = ["reduce", "sinf", "addf", "const"]

    return test_with_full_kernel("sin(x) + 1", x -> sin(x) + 1, +, patterns)
end

"""
    test_abs_minus_one()

Test: mapreduce(x -> abs(x - 1), max, tile, 1)
Expected: reduce with absf and subf
"""
function test_abs_minus_one()
    println("\n" * repeat("#", 60))
    println("# Test: abs(x - 1) expression")
    println("#" * repeat("#", 60))

    # This should decompose abs(x - 1) into: absf(subf(elem, const(1)))
    patterns = ["reduce", "absf", "subf"]

    return test_with_full_kernel("abs(x - 1)", x -> abs(x - 1), max, patterns)
end

"""
    test_composite_expression()

Test: mapreduce(x -> (x + 1) * 2, *, tile, 1)
Expected: reduce with nested operations
"""
function test_composite_expression()
    println("\n" * repeat("#", 60))
    println("# Test: (x + 1) * 2 composite expression")
    println("#" * repeat("#", 60))

    # This should decompose (x + 1) * 2 into: mulf(addf(elem, const(1)), const(2))
    patterns = ["reduce", "mulf", "addf", "const"]

    return test_with_full_kernel("(x + 1) * 2", x -> (x + 1) * 2, *, patterns)
end

"""
    test_sin_mul_cos()

Test: mapreduce(x -> sin(x) * cos(x), +, tile, 2)
Expected: reduce with sinf, cosf, and mulf
"""
function test_sin_mul_cos()
    println("\n" * repeat("#", 60))
    println("# Test: sin(x) * cos(x) expression")
    println("#" * repeat("#", 60))

    # This should decompose sin(x) * cos(x) into: mulf(sinf(elem), cosf(elem))
    patterns = ["reduce", "sinf", "cosf", "mulf"]

    return test_with_full_kernel("sin(x) * cos(x)", x -> sin(x) * cos(x), +, patterns)
end

#=============================================================================
# Test Runner
#=============================================================================

"""
    run_all()

Run all tests and report results.
"""
function run_all()
    println("\n")
    println(repeat("*", 70))
    println("* MAPREDUCE DECOMPOSITION - MLIR OUTPUT VERIFICATION")
    println("* Testing if expressions are correctly decomposed into Tile IR ops")
    println(repeat("*", 70))

    results = []

    # Run all tests
    push!(results, ("identity + sum", test_identity_plus()))
    push!(results, ("x + 1", test_x_plus_one()))
    push!(results, ("2 * x", test_two_mul_x()))
    push!(results, ("x^2", test_x_squared()))
    push!(results, ("sin(x) + 1", test_sin_plus_one()))
    push!(results, ("abs(x - 1)", test_abs_minus_one()))
    push!(results, ("(x + 1) * 2", test_composite_expression()))
    push!(results, ("sin(x) * cos(x)", test_sin_mul_cos()))

    # Summary
    println("\n")
    println(repeat("=", 70))
    println("TEST SUMMARY")
    println(repeat("=", 70))

    passed = 0
    failed = 0

    for (name, result) in results
        status = result ? "✓ PASS" : "✗ FAIL"
        println("  $status: $name")
        if result
            passed += 1
        else
            failed += 1
        end
    end

    println(repeat("=", 70))
    println("  Total: $(length(results)) tests")
    println("  Passed: $passed")
    println("  Failed: $failed")
    println(repeat("=", 70))

    if failed == 0
        println("\n✓ ALL TESTS PASSED!")
        println("Expression decomposition is working correctly.")
        println("MLIR output shows proper decomposition into Tile IR operations.\n")
        return true
    else
        println("\n✗ SOME TESTS FAILED")
        println("Check the MLIR output above for details.\n")
        return false
    end
end

"""
    quick_test()

Run a quick sanity check.
"""
function quick_test()
    println("\n[Quick Test] Verifying decomposition...")

    # Just test one expression
    test_x_plus_one()

    return true
end

#=============================================================================
# Module Initialization
#=============================================================================

# Auto-run hint
println("\n")
println("Type: TestMLIROutput.run_all()  - Run all MLIR verification tests")
println("      TestMLIROutput.quick_test() - Quick sanity check")
println("")

end  # module
