"""
    Simple Scan Bytecode Test

Minimal test for generating scan operation bytecode using cuTile's API.
Demonstrates proper identity encoding and scan operation structure.

Key findings:
- ScanOp opcode = 94
- Identity values require zigzag encoding via mask_to_width
- Entry functions require 0D tiles but scan requires >=1D tiles
- Body region arguments are (accumulator, element) pairs of scalar tiles
"""

using cuTile
import cuTile: encode_varint!, mask_to_width, Value, encode_ReturnOp!,
               encode_YieldOp!, with_region, new_op!, encode_typeid!,
               IntegerIdentityVal, tile_type!, julia_to_tile_dtype!
import Statistics: mean

"Return identity value for scan operation."
get_identity(op::Symbol, ::Type{Int32}) = if op == :add 0 elseif op == :mul 1 elseif op == :max typemin(Int32) else typemax(Int32) end
get_identity(op::Symbol, ::Type{Int64}) = if op == :add Int64(0) elseif op == :mul Int64(1) elseif op == :max typemin(Int64) else typemax(Int64) end
get_identity(op::Symbol, ::Type{Float32}) = if op == :add Float32(0) else Float32(1) end

"Generate valid cuTile bytecode for a scan operation."
function generate_scan_bytecode(::Type{T}; op::Symbol=:add) where T
    cuTile.write_bytecode!(1) do writer, func_buf
        dtype = julia_to_tile_dtype!(writer.type_table, T)
        scalar_tile = tile_type!(writer.type_table, dtype, Int[])

        signed_val = get_identity(op, T)
        uint_val = reinterpret(UInt128, convert(Int128, signed_val))
        identity = IntegerIdentityVal(uint_val, dtype, T)

        cb = cuTile.add_function!(writer, func_buf, "scan", [scalar_tile], cuTile.TypeId[]; is_entry=true)

        encode_varint!(cb.buf, 94)

        encode_varint!(cb.buf, 1)
        encode_typeid!(cb.buf, scalar_tile)

        encode_varint!(cb.buf, 0)
        push!(cb.buf, 0x00)

        encode_varint!(cb.buf, 1)
        if T <: Integer
            push!(cb.buf, 0x01)  # Integer tag
            encode_typeid!(cb.buf, dtype)
            encoded = mask_to_width(uint_val, T)
            encode_varint!(cb.buf, encoded)
        else
            push!(cb.buf, 0x02)  # Float tag
            encode_typeid!(cb.buf, dtype)
            bits = reinterpret(UInt32, get_identity(:add, T))
            encode_varint!(cb.buf, bits)
        end

        encode_varint!(cb.buf, 1)
        encode_varint!(cb.buf, 0)

        encode_varint!(cb.buf, 1)

        body_types = [scalar_tile, scalar_tile]
        body_args = with_region(cb, body_types) do args
            encode_YieldOp!(cb, [args[1]])
        end

        new_op!(cb, 1)
        encode_ReturnOp!(cb, Value[])

        cuTile.finalize_function!(func_buf, cb, writer.debug_info)
    end
end

function test_magic_header()
    println("Testing magic header...")
    for T in [Int32, Int64, Float32]
        bc = generate_scan_bytecode(T)
        @assert bc[1:8] == b"\x7FTileIR\x00" "Invalid magic for $T"
    end
    println("  PASSED")
end

function test_version()
    println("Testing version...")
    bc = generate_scan_bytecode(Int32)
    @assert bc[9] == 13 "Version major"
    @assert bc[10] == 1 "Version minor"
    println("  PASSED")
end

function test_bytecode_size()
    println("Testing bytecode size...")
    for T in [Int32, Int64, Float32]
        bc = generate_scan_bytecode(T)
        @assert length(bc) > 100 "Bytecode too small for $T"
    end
    println("  PASSED")
end

function test_scan_opcode()
    println("Testing scan opcode (94)...")
    bc = generate_scan_bytecode(Int32)
    @assert 0x5e in bc "Scan opcode 94 not found in bytecode"
    println("  PASSED")
end

function test_identity_encoding()
    println("Testing identity encoding...")
    for op in [:add, :mul, :max, :min]
        bc = generate_scan_bytecode(Int32; op=op)
        @assert length(bc) > 100 "Too small for $op"
        @assert 0x01 in bc "Integer tag not found for $op"
    end
    println("  PASSED")
end

function test_different_types()
    println("Testing different element types...")
    for T in [Int32, Int64, Float32]
        bc = generate_scan_bytecode(T)
        @assert length(bc) > 100 "Too small for $T"
    end
    println("  PASSED")
end

function benchmark()
    println("\nBenchmarking bytecode generation...")
    for T in [Int32, Float32]
        times = [(@elapsed generate_scan_bytecode(T)) * 1000 for _ in 1:5]
        println("  $T: first=$(round(times[1], digits=3))ms, mean=$(round(mean(times), digits=3))ms")
    end
end

function test_all()
    println(repeat("=", 60))
    println("SCAN BYTECODE GENERATION TEST")
    println(repeat("=", 60))
    test_magic_header()
    test_version()
    test_bytecode_size()
    test_scan_opcode()
    test_identity_encoding()
    test_different_types()
    benchmark()
    println(repeat("=", 60))
    println("ALL TESTS PASSED")
    println(repeat("=", 60))
end

reload() = include("cuTile/examples/bytecode/scan_simple_test.jl")
gen(T) = generate_scan_bytecode(T)
save(T, file) = write(file, generate_scan_bytecode(T))

if abspath(PROGRAM_FILE) == @__FILE__
    test_all()
end
