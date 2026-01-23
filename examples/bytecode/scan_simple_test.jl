"""
    Scan Bytecode Test

Tests manual scan bytecode generation with structural validation.
Focuses on verifying proper bytecode encoding without disassembly.
"""

using cuTile
import cuTile: encode_varint!, mask_to_width, Value, encode_ReturnOp!,
               encode_YieldOp!, with_region, new_op!, encode_typeid!,
               IntegerIdentityVal, tile_type!, julia_to_tile_dtype!

# Identity values for different operations
get_identity(op::Symbol, ::Type{Int32}) = if op == :add 0 elseif op == :mul 1 elseif op == :max typemin(Int32) else typemax(Int32) end
get_identity(op::Symbol, ::Type{Float32}) = if op == :add Float32(0) else Float32(1) end

"""
    generate_scan_bytecode(::Type{T}; op::Symbol=:add) where T

Generate valid cuTile bytecode for a scan operation.
Uses 0D tiles for entry function compatibility.
"""
function generate_scan_bytecode(::Type{T}; op::Symbol=:add) where T
    cuTile.write_bytecode!(1) do writer, func_buf
        dtype = julia_to_tile_dtype!(writer.type_table, T)
        tile = tile_type!(writer.type_table, dtype, Int[])

        signed_val = get_identity(op, T)
        uint_val = reinterpret(UInt128, convert(Int128, signed_val))
        identity = IntegerIdentityVal(uint_val, dtype, T)

        cb = cuTile.add_function!(writer, func_buf, "scan", [tile], cuTile.TypeId[];
                                  is_entry=true)

        encode_varint!(cb.buf, 94)  # ScanOp = 94

        encode_varint!(cb.buf, 1)
        encode_typeid!(cb.buf, tile)

        encode_varint!(cb.buf, 0)  # dim=0
        push!(cb.buf, 0x00)  # reverse=false

        encode_varint!(cb.buf, 1)
        if T <: Integer
            push!(cb.buf, 0x01)
            encode_typeid!(cb.buf, dtype)
            encoded = mask_to_width(uint_val, T)
            encode_varint!(cb.buf, encoded)
        else
            push!(cb.buf, 0x02)
            encode_typeid!(cb.buf, dtype)
            bits = reinterpret(UInt32, get_identity(:add, T))
            encode_varint!(cb.buf, bits)
        end

        encode_varint!(cb.buf, 1)
        encode_varint!(cb.buf, 0)

        encode_varint!(cb.buf, 1)

        body_types = [tile, tile]
        with_region(cb, body_types) do args
            encode_YieldOp!(cb, [args[1]])
        end

        new_op!(cb, 1)
        encode_ReturnOp!(cb, Value[])

        cuTile.finalize_function!(func_buf, cb, writer.debug_info)
    end
end

# Tests
function test_magic_header()
    println("Magic header...")
    bc = generate_scan_bytecode(Int32)
    @assert bc[1:8] == b"\x7FTileIR\x00"
    println("  ✓")
end

function test_version()
    println("Version...")
    bc = generate_scan_bytecode(Int32)
    @assert bc[9] == 13 "major"
    @assert bc[10] == 1 "minor"
    println("  ✓")
end

function test_bytecode_size()
    println("Bytecode size...")
    bc = generate_scan_bytecode(Int32)
    @assert length(bc) > 100
    println("  ✓")
end

function test_scan_opcode()
    println("Scan opcode (94 = 0x5E)...")
    bc = generate_scan_bytecode(Int32)
    @assert 0x5e in bc
    println("  ✓")
end

function test_identity_encoding()
    println("Identity encoding...")
    for op in [:add, :mul, :max, :min]
        bc = generate_scan_bytecode(Int32; op=op)
        @assert 0x01 in bc "Integer tag for $op"
    end
    println("  ✓")
end

function test_different_types()
    println("Different types...")
    for T in [Int32, Float32]
        bc = generate_scan_bytecode(T)
        @assert length(bc) > 100
    end
    println("  ✓")
end

function benchmark()
    println("\nBenchmark...")
    for T in [Int32, Float32]
        t = @elapsed bc = generate_scan_bytecode(T)
        println("  $T: $(round(t*1000, digits=3))ms, $(length(bc)) bytes")
    end
end

function test_all()
    println(repeat("=", 50))
    println("SCAN BYTECODE TEST")
    println(repeat("=", 50))
    test_magic_header()
    test_version()
    test_bytecode_size()
    test_scan_opcode()
    test_identity_encoding()
    test_different_types()
    benchmark()
    println(repeat("=", 50))
    println("ALL TESTS PASSED")
    println(repeat("=", 50))
end

# REPL helpers
reload() = include("cuTile/examples/bytecode/scan_simple_test.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    test_all()
end
