# Test script for scan bytecode generation
#
# This script validates the bytecode generation for scan operations
# and demonstrates how to integrate it with cuTile's compilation.

using Test
using cuTile
using .cuTile.Bytecode: TypeTable, StringTable, ConstantTable
using .cuTile.Types: SimpleType, CompositeType

# Include our implementation
include("scan_bytecode.jl")

@testset "Scan Bytecode Generation Tests" begin
    @testset "Type System" begin
        # Test that our type system correctly encodes basic types
        type_table = TypeTable()

        # Test simple types
        i32_type = cuTile.Bytecode.I32(type_table)
        f16_type = cuTile.Bytecode.F16(type_table)

        @test i32_type.id == I32_TYPE_ID.id
        @test f16_type.id > 0  # Should be a registered type

        # Test tile types
        tile_type = tile_type!(type_table, i32_type, [32, 32])
        @test tile_type.id > 0  # Should be a new type

        # Test function types
        func_type = function_type!(type_table, [tile_type], [tile_type])
        @test func_type.id > 0  # Should be a new type
    end

    @testset "Bytecode Generator" begin
        # Test that we can create a generator
        generator = ScanBytecodeGenerator()
        @test isa(generator.type_table, TypeTable)
        @test isa(generator.string_table, StringTable)
        @test isa(generator.constant_table, ConstantTable)

        # Test sequential bytecode generation
        element_type = Int32
        tile_size = (4, 4)  # Small size for testing

        seq_bytecode = generate_prefix_sum_bytecode!(generator, element_type, tile_size)
        @test length(seq_bytecode) > 0
        @test seq_bytecode[1] == 0x01  # Should start with function tag

        # Check that the bytecode contains expected structure
        # We can't fully verify without a decoder, but we can check basic structure
        @test 0x10 in seq_bytecode  # Should contain loop tag
        @test 0x20 in seq_bytecode  # Should contain load tag
        @test 0x30 in seq_bytecode  # Should contain add tag
        @test 0x40 in seq_bytecode  # Should contain store tag
        @test 0x60 in seq_bytecode  # Should contain return tag

        # Test parallel bytecode generation
        par_bytecode = generate_parallel_scan_bytecode!(generator, element_type, tile_size)
        @test length(par_bytecode) > 0
        @test par_bytecode[1] == 0x01  # Should start with function tag

        # Parallel scan should include warp primitives
        @test 0x80 in par_bytecode  # Should include warp_prefix_sum
    end

    @testset "Table Serialization" begin
        generator = ScanBytecodeGenerator()

        # Add some entries to tables
        type_table = generator.type_table
        str_table = generator.string_table
        const_table = generator.constant_table

        # Add a string
        test_str_id = str_table["test_string"]

        # Add a constant
        test_const_id = dense_constant!(const_table, UInt8[1, 2, 3, 4])

        # Serialize tables
        buf = UInt8[]
        serialize_tables!(generator, buf)

        @test length(buf) > 0

        # Basic structure validation
        # Type table header
        @test buf[1] >= 0  # Type table length
        # String table header (should include our test string)
        @test buf[2 + 4] >= 1  # String table should have at least 1 entry
    end

    @testset "Demo Function" begin
        # Test that our demo function runs without error
        seq_bytecode, par_bytecode, generator = demo_scan_bytecode()

        @test length(seq_bytecode) > 0
        @test length(par_bytecode) > 0
        @test length(par_bytecode) > length(seq_bytecode)  # Parallel should be larger

        # Print bytecodes for manual inspection
        println("\nSequential scan bytecode preview:")
        for (i, byte) in enumerate(seq_bytecode[1:min(20, length(seq_bytecode))])
            print("0x", string(byte, base=16, pad=2), " ")
        end
        println("...")

        println("\nParallel scan bytecode preview:")
        for (i, byte) in enumerate(par_bytecode[1:min(20, length(par_bytecode))])
            print("0x", string(byte, base=16, pad=2), " ")
        end
        println("...")
    end
end

# Performance comparison
@testset "Performance Benchmarks" begin
    using BenchmarkTools

    generator = ScanBytecodeGenerator()
    element_type = Int32
    tile_size = (32, 32)

    println("\nBenchmarking sequential scan bytecode generation:")
    seq_time = @benchmark generate_prefix_sum_bytecode!($generator, $element_type, $tile_size)
    display(seq_time)

    println("\nBenchmarking parallel scan bytecode generation:")
    par_time = @benchmark generate_parallel_scan_bytecode!($generator, $element_type, $tile_size)
    display(par_time)

    # Parallel generation should take longer due to more complex logic
    @test median(par_time.times) > median(seq_time.times)
end

println("All tests passed!")
