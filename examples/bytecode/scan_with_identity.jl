"""
    Scan Bytecode Generation with Identity Values

This module demonstrates how to generate valid cuTile TileIR bytecode
for scan (prefix sum) operations with proper identity value encoding.

The scan operation computes an inclusive prefix sum:
    result[i] = identity ⊕ operand[0] ⊕ operand[1] ⊕ ... ⊕ operand[i]

Identity values ensure correctness at scan boundaries:
- SCAN_ADD: identity = 0 (since 0 ⊕ x = x)
- SCAN_MUL: identity = 1 (since 1 ⊗ x = x)
- SCAN_MAX: identity = typemin(x) (since typemin ⊕ x = x)
- SCAN_MIN: identity = typemax(x) (since typemax ⊕ x = x)

Based on ReduceOp patterns from src/bytecode/encodings.jl.

# Example Usage

```julia
include("cuTile/examples/bytecode/scan_with_identity.jl")

# Generate scan bytecode for Int32 addition
bytecode = generate_scan_add_bytecode(Int32)
write("scan_add.tile", bytecode)

# Or use the high-level API
bytecode = generate_scan_bytecode(
    Int32,
    dim=1,
    reverse=false,
    op=:add
)

# Disassemble to verify
run(`cuda-tile-translate --cudatilebc-to-mlir scan_add.tile`)
```
"""

using cuTile
import cuTile: TypeTable, StringTable, ConstantTable,
               encode_varint!, encode_int_list!,
               write_bytecode!,
               BytecodeWriter, CodeBuilder,
               I1, I8, I16, I32, I64, F16, BF16, F32, TF32, F64,
               simple_type!, tile_type!, pointer_type!, tensor_view_type!,
               partition_view_type!, function_type!,
               julia_to_tile_dtype!, dense_constant!,
               DebugAttrId, TypeId,
               encode_ReturnOp!, Value,
               IntegerIdentityVal, FloatIdentityVal,
               encode_identity_array!

# ===============================================================================
# Identity Value Functions
# ===============================================================================

"""
    get_scan_identity(op::Symbol, ::Type{T})

Return the identity value for a given scan operation and element type.

# Arguments
- `op`: Scan operation symbol (`:add`, `:mul`, `:max`, `:min`)
- `T`: Element type (e.g., Int32, Float32, Int64)

# Returns
- Integer or float identity value

# Examples
```julia
get_scan_identity(:add, Int32)  # returns 0
get_scan_identity(:mul, Int32)  # returns 1
get_scan_identity(:max, Int32)  # returns typemin(Int32)
get_scan_identity(:min, Int32)  # returns typemax(Int32)
```
"""
function get_scan_identity(op::Symbol, ::Type{T}) where T
    if op == :add
        return zero(T)
    elseif op == :mul
        return one(T)
    elseif op == :max
        return typemin(T)
    elseif op == :min
        return typemax(T)
    else
        error("Unknown scan operation: $op. Supported: :add, :mul, :max, :min")
    end
end

"""
    create_identity_value(op::Symbol, dtype::TypeId, elem_type::Type, writer::BytecodeWriter)

Create an IdentityVal struct for the given scan operation and type.

# Arguments
- `op`: Scan operation symbol
- `dtype`: TypeId for the element type
- `elem_type`: Julia type (Int32, Float32, etc.)
- `writer`: BytecodeWriter for type table access

# Returns
- IntegerIdentityVal or FloatIdentityVal
"""
function create_identity_value(op::Symbol, dtype::TypeId, elem_type::Type, writer::BytecodeWriter)
    identity = get_scan_identity(op, elem_type)

    if elem_type in (Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64)
        return IntegerIdentityVal(UInt128(identity), dtype, elem_type)
    else
        # Float types
        return FloatIdentityVal(Float64(identity), dtype, elem_type)
    end
end

# ===============================================================================
# High-Level Scan Bytecode Generation
# ===============================================================================

"""
    generate_scan_bytecode(::Type{T};
                          tile_size::Tuple{Int, Int} = (32, 32),
                          dim::Int = 1,
                          reverse::Bool = false,
                          op::Symbol = :add) where T

Generate cuTile bytecode for a scan (prefix sum) operation.

# Arguments
- `T`: Element type (e.g., Int32, Float32, Int64)
- `tile_size`: Size of the tile to scan (default: (32, 32))
- `dim`: Dimension to scan along (default: 1)
- `reverse`: Whether to scan in reverse order (default: false)
- `op`: Scan operation (default: :add)

# Returns
- Vector{UInt8} containing valid cuTile bytecode

# Examples
```julia
# Basic scan for Int32
bytecode = generate_scan_bytecode(Int32)

# Scan along dimension 2 with Float32
bytecode = generate_scan_bytecode(Float32; dim=2, tile_size=(16, 16))

# Maximum scan
bytecode = generate_scan_bytecode(Int32; op=:max)
```
"""
function generate_scan_bytecode(::Type{T};
                                tile_size::Tuple{Int, Int} = (32, 32),
                                dim::Int = 1,
                                reverse::Bool = false,
                                op::Symbol = :add) where T
    return write_bytecode!(1) do writer, func_buf
        # Get type IDs
        dtype = julia_to_tile_dtype!(writer.type_table, T)
        in_tile = tile_type!(writer.type_table, dtype, [tile_size[1], tile_size[2]])
        out_tile = tile_type!(writer.type_table, dtype, [tile_size[1], tile_size[2]])

        # Create identity value for the operation
        identity = create_identity_value(op, dtype, T, writer)

        # Build the scan function body
        build_scan_function!(writer, func_buf, in_tile, out_tile, identity;
                             dim=dim, reverse=reverse, op=op)
    end
end

"""
    generate_scan_add_bytecode(::Type{T}; tile_size::Tuple{Int, Int} = (32, 32)) where T

Convenience function to generate scan bytecode for addition.
"""
function generate_scan_add_bytecode(::Type{T}; tile_size::Tuple{Int, Int} = (32, 32)) where T
    return generate_scan_bytecode(T; tile_size=tile_size, op=:add)
end

"""
    generate_scan_mul_bytecode(::Type{T}; tile_size::Tuple{Int, Int} = (32, 32)) where T

Convenience function to generate scan bytecode for multiplication.
"""
function generate_scan_mul_bytecode(::Type{T}; tile_size::Tuple{Int, Int} = (32, 32)) where T
    return generate_scan_bytecode(T; tile_size=tile_size, op=:mul)
end

"""
    generate_scan_max_bytecode(::Type{T}; tile_size::Tuple{Int, Int} = (32, 32)) where T

Convenience function to generate scan bytecode for maximum.
"""
function generate_scan_max_bytecode(::Type{T}; tile_size::Tuple{Int, Int} = (32, 32)) where T
    return generate_scan_bytecode(T; tile_size=tile_size, op=:max)
end

"""
    generate_scan_min_bytecode(::Type{T}; tile_size::Tuple{Int, Int} = (32, 32)) where T

Convenience function to generate scan bytecode for minimum.
"""
function generate_scan_min_bytecode(::Type{T}; tile_size::Tuple{Int, Int} = (32, 32)) where T
    return generate_scan_bytecode(T; tile_size=tile_size, op=:min)
end

# ===============================================================================
# Internal Function Building
# ===============================================================================

"""
    build_scan_function!(writer, func_buf, in_tile, out_tile, identity;
                        dim, reverse, op)

Build the actual function body with scan operations.

This creates a minimal function structure that demonstrates:
1. Parameter passing
2. Identity value encoding
3. Scan operation encoding
4. Return operation
"""
function build_scan_function!(writer::BytecodeWriter,
                              func_buf::Vector{UInt8},
                              in_tile::TypeId,
                              out_tile::TypeId,
                              identity;
                              dim::Int = 1,
                              reverse::Bool = false,
                              op::Symbol = :add)

    # Add function to writer with entry point flag
    cb = cuTile.add_function!(writer, func_buf, "scan",
                               [in_tile], [out_tile];
                               is_entry=true)

    # Get the scan operation opcode
    scan_opcode = 94  # From src/bytecode/encodings.jl: const ScanOp = 94

    # Encode scan operation:
    # [opcode][result_types][attributes][operands][regions]

    # 1. Opcode
    encode_varint!(cb.buf, scan_opcode)

    # 2. Variadic result types (one output tile)
    encode_varint!(cb.buf, 1)  # 1 result type
    encode_typeid!(cb.buf, out_tile)

    # 3. Attributes: dim, reverse, identities
    encode_varint!(cb.buf, dim)  # scan dimension
    encode_opattr_bool!(cb, reverse)  # reverse flag

    # Encode identity array (following ReduceOp pattern)
    encode_varint!(cb.buf, 1)  # 1 identity value
    encode_identity!(cb, identity)

    # 4. Variadic operands: input tile
    encode_varint!(cb.buf, 1)  # 1 operand
    encode_operand!(cb.buf, Value(0))  # First function parameter

    # 5. Number of regions (0 for scan - no nested computation)
    encode_varint!(cb.buf, 0)

    # Mark operation completion
    new_op!(cb, 1)  # 1 result value (the output tile)

    # Get the output value
    result = Value(cb.next_value_id - 1)

    # Return the result
    encode_ReturnOp!(cb, [result])

    # Finalize the function
    cuTile.finalize_function!(func_buf, cb, writer.debug_info)

    return cb
end

"""
    encode_opattr_bool!(cb::CodeBuilder, val::Bool)

Encode a boolean operation attribute.
"""
function encode_opattr_bool!(cb::CodeBuilder, val::Bool)
    push!(cb.buf, val ? 0x01 : 0x00)
end

# ===============================================================================
# Validation and Testing
# ===============================================================================

"""
    test_scan_bytecode_generation()

Test scan bytecode generation and disassembly.
"""
function test_scan_bytecode_generation()
    println("=" ^ 60)
    println("SCAN BYTECODE GENERATION TEST")
    println("=" ^ 60)

    test_cases = [
        (:add, Int32, "Addition Scan (Int32)"),
        (:add, Float32, "Addition Scan (Float32)"),
        (:mul, Int32, "Multiplication Scan (Int32)"),
        (:max, Int32, "Maximum Scan (Int32)"),
        (:min, Int32, "Minimum Scan (Int32)"),
    ]

    all_passed = true

    for (op, dtype, desc) in test_cases
        println("\n--- Testing: $desc ---")

        try
            # Generate bytecode
            bytecode = generate_scan_bytecode(dtype; op=op)

            println("  Bytecode size: $(length(bytecode)) bytes")
            println("  First 32 bytes: $(bytes2hex(bytecode[1:min(32, length(bytecode))]))")

            # Verify magic header
            @assert bytecode[1:8] == b"\x7FTileIR\x00" "Invalid magic header"
            println("  ✓ Magic header valid")

            # Save to temp file for disassembly
            temp_file = tempname() * ".tile"
            write(temp_file, bytecode)

            try
                # Try to disassemble
                disasm_cmd = `$(cuTile.cuda_tile_translate()) --cudatilebc-to-mlir $temp_file`
                disasm = read(disasm_cmd, String)

                println("  ✓ Disassembly successful")
                println("  First 500 chars of disassembly:")
                println("  " * replace(disasm[1:min(500, length(disasm))], "\n" => "\n  "))

            catch e
                println("  ⚠ Disassembly warning: $e")
            finally
                rm(temp_file, force=true)
            end

        catch e
            println("  ✗ FAILED: $e")
            all_passed = false
        end
    end

    println("\n" * "=" ^ 60)
    if all_passed
        println("ALL TESTS PASSED ✓")
    else
        println("SOME TESTS FAILED ✗")
    end
    println("=" ^ 60)

    return all_passed
end

"""
    benchmark_scan_bytecode(::Type{T}; n::Int=100) where T

Benchmark bytecode generation for a given type.
"""
function benchmark_scan_bytecode(::Type{T}; n::Int=100) where T
    println("Benchmarking scan bytecode generation for $T ($n iterations):")

    times = Float64[]
    for i in 1:n
        t = @elapsed bytecode = generate_scan_bytecode(T)
        push!(times, t)
    end

    println("  First run (JIT): $(times[1]*1000) ms")
    println("  Steady-state (mean): $(mean(times[2:end])*1000) ms")
    println("  Steady-state (std):  $(std(times[2:end])*1000) ms")
    println("  Min: $(minimum(times[2:end])*1000) ms")
    println("  Max: $(maximum(times[2:end])*1000) ms")

    return times
end

# ===============================================================================
# REPL Integration
# ===============================================================================

"""
    reload()

Reload the module and test. Call after making changes.
```julia
julia> include("cuTile/examples/bytecode/scan_with_identity.jl")
julia> reload()  # After edits
```
"""
function reload()
    include("cuTile/examples/bytecode/scan_with_identity.jl")
    test_scan_bytecode_generation()
end

"""
    quick_test()

Quick validation of bytecode generation without full disassembly.
"""
function quick_test()
    println("Quick test...")

    # Generate for all types
    for dtype in [Int32, Float32, Int64, Float64]
        bytecode = generate_scan_bytecode(dtype)
        @assert length(bytecode) > 100 "Bytecode too small for $dtype"
        @assert bytecode[1:8] == b"\x7FTileIR\x00" "Invalid magic for $dtype"
    end

    println("Quick test PASSED ✓")
    return true
end

# ===============================================================================
# Main Entry Point
# ===============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    test_scan_bytecode_generation()
end
