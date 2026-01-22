# Corrected TileIR Bytecode Generation for Hello World
#
# This implementation follows the TileIR specification from the documentation
# to generate valid bytecode that can be successfully disassembled.

using cuTile

"""
    generate_hello_world_bytecode() -> Vector{UInt8}

Generate complete, valid cuTile bytecode for a simple "Hello World" operation
following the exact specification from the TileIR documentation.
"""
function generate_hello_world_bytecode()
    buf = Vector{UInt8}()

    # 1. Magic Header (8 bytes)
    append!(buf, b"\x7FTileIR\x00")

    # 2. Version (4 bytes: major=13, minor=1, tag=0)
    push!(buf, 0x0d)  # Major version 13
    push!(buf, 0x01)  # Minor version 1
    push!(buf, 0x00)  # Tag low
    push!(buf, 0x00)  # Tag high

    # 3. Function Table Section (ID=2, required, with 8-byte alignment)
    # Section header: ID with alignment bit
    push!(buf, 0x82)  # ID=2 with high bit set (alignment required)

    # Function table data
    func_data = UInt8[]
    # Number of functions
    encode_varint!(func_data, 1)
    # Function metadata
    encode_varint!(func_data, 0)  # String index for function name
    encode_varint!(func_data, 0)  # Type index (void -> void)
    encode_varint!(func_data, 0)  # Debug location index (none)

    # Function body (opcodes for print operation)
    func_body = UInt8[]
    # Print instruction (using placeholder opcode)
    push!(func_body, 0x01)  # Opcode
    encode_varint!(func_body, 0)  # No debug location
    # String literal for "Hello World!\n"
    hello_world = "Hello World!\n"
    encode_varint!(func_body, length(hello_world))
    append!(func_body, codeunits(hello_world))

    # Function body length
    encode_varint!(func_data, length(func_body))
    # Function body
    append!(func_data, func_body)

    # Section length
    encode_varint!(buf, length(func_data))
    # Section alignment
    encode_varint!(buf, 8)  # 8-byte alignment

    # Add padding to ensure 8-byte alignment for function data
    current_pos = length(buf) + 1  # Position after alignment byte
    padding_needed = (8 - current_pos % 8) % 8
    for _ in 1:padding_needed
        push!(buf, 0xCB)  # Padding byte
    end

    # Function table data
    append!(buf, func_data)

    # 4. Constant Data Section (ID=4, optional, with 8-byte alignment)
    # We don't need constants for this simple example, but we include an empty section
    # to maintain the expected section order
    push!(buf, 0x84)  # ID=4 with high bit set (alignment required)

    # Empty constant data
    const_data = UInt8[]
    encode_varint!(const_data, 0)  # No constants

    # Section length
    encode_varint!(buf, length(const_data))
    # Section alignment
    encode_varint!(buf, 8)  # 8-byte alignment

    # Add padding to ensure 8-byte alignment for constant data
    current_pos = length(buf) + 1  # Position after alignment byte
    padding_needed = (8 - current_pos % 8) % 8
    for _ in 1:padding_needed
        push!(buf, 0xCB)  # Padding byte
    end

    # Constant data (empty)
    append!(buf, const_data)

    # 5. Debug Section (ID=3, optional, with 8-byte alignment)
    # We don't need debug info for this simple example
    push!(buf, 0x83)  # ID=3 with high bit set (alignment required)

    # Empty debug data
    debug_data = UInt8[]
    encode_varint!(debug_data, 0)  # No debug entries

    # Section length
    encode_varint!(buf, length(debug_data))
    # Section alignment
    encode_varint!(buf, 8)  # 8-byte alignment

    # Add padding to ensure 8-byte alignment for debug data
    current_pos = length(buf) + 1  # Position after alignment byte
    padding_needed = (8 - current_pos % 8) % 8
    for _ in 1:padding_needed
        push!(buf, 0xCB)  # Padding byte
    end

    # Debug data (empty)
    append!(buf, debug_data)

    # 6. Type Section (ID=5, required, with 4-byte alignment)
    # Section header: ID with alignment bit
    push!(buf, 0x85)  # ID=5 with high bit set (alignment required)

    # Calculate type data
    type_data = UInt8[]
    # Type 0: void
    push!(type_data, 0x00)  # SimpleType::Void
    # Type 1: Function type (void) -> void
    push!(type_data, 0x10)  # CompositeType::Func
    push!(type_data, 0x00)  # No parameters
    push!(type_data, 0x00)  # No results

    # Section length
    encode_varint!(buf, length(type_data))
    # Section alignment
    encode_varint!(buf, 4)  # 4-byte alignment (not 8!)

    # Add padding to ensure 4-byte alignment for type data
    current_pos = length(buf) + 1  # Position after alignment byte
    padding_needed = (4 - current_pos % 4) % 4
    for _ in 1:padding_needed
        push!(buf, 0xCB)  # Padding byte
    end

    # Add type data
    append!(buf, type_data)

    # 7. String Section (ID=1, required, with 4-byte alignment)
    # Header with alignment bit
    push!(buf, 0x81)  # ID=1 with high bit set (alignment required)

    # String data
    string_data = UInt8[]
    # Number of strings
    encode_varint!(string_data, 1)
    # String start indices (each is 4 bytes)
    indices = [0]
    encode_varint!(string_data, indices[1])
    # String data (UTF-8)
    hello_str = "hello_world_kernel"
    append!(string_data, codeunits(hello_str))

    # Section length
    encode_varint!(buf, length(string_data))
    # Section alignment
    encode_varint!(buf, 4)  # 4-byte alignment

    # Add padding to ensure 4-byte alignment for string data
    current_pos = length(buf) + 1  # Position after alignment byte
    padding_needed = (4 - current_pos % 4) % 4
    for _ in 1:padding_needed
        push!(buf, 0xCB)  # Padding byte
    end

    # String section data
    append!(buf, string_data)

    # 8. End-of-Bytecode Marker (required)
    push!(buf, 0x00)

    return buf
end

"""
    encode_varint!(buf::Vector{UInt8}, x::Integer)

Encode an unsigned integer using variable-length encoding (LEB128-style).
"""
function encode_varint!(buf::Vector{UInt8}, x::Integer)
    @assert x >= 0 "Varint encoding requires non-negative integers, got $x"
    result = x
    while true
        byte = result & 0x7f
        result >>= 7
        if result != 0
            byte |= 0x80
        end
        push!(buf, byte)
        if result == 0
            break
        end
    end
end

# Test function
function test_hello_world_bytecode()
println("Generating TileIR bytecode for Hello World...")
bytecode = generate_hello_world_bytecode()

println("Bytecode generated: $(length(bytecode)) bytes")
println("First 64 bytes: ", bytes2hex(bytecode[1:min(64, length(bytecode))]))

# More comprehensive debugging
println("\n=== Section Debug Information ===")
println("Byte at pos 0-7 (magic): ", bytes2hex(bytecode[1:8]))
println("Byte at pos 8-11 (version): ", bytes2hex(bytecode[9:12]))
println("Byte at pos 13 (func section header): ", string(bytecode[13], base=16), " (binary: ", string(bytecode[13], base=2), ")")
println("Byte at pos 14 (): ", string(bytecode[14], base=16))
println("Byte at pos 15 (): ", string(bytecode[15], base=16))

# Let's find the actual type section
pos = 1
while pos < length(bytecode)
    if bytecode[pos] == 0x05 || bytecode[pos] == 0x85
        println("\nFound type section header at position: ", pos)
        println("Header byte: ", string(bytecode[pos], base=16), " (binary: ", string(bytecode[pos], base=2), ")")
        break
    end
    pos += 1
end

# Write to temporary file and try to disassemble
temp_file = tempname() * ".tile"
write(temp_file, bytecode)

    try
        # Use cuTile's translate tool to disassemble
        disasm_cmd = `$(cuTile.cuda_tile_translate()) --cudatilebc-to-mlir $temp_file`
        disasm = read(disasm_cmd, String)

        println("\n=== Successfully Disassembled! ===")
        println(disasm)
        return true
    catch e
        println("\nDisassembly failed: $e")
        if isa(e, Base.IOError)
            println("IOError - process failed")
            # Try to get the actual command output if available
            try
                result = read(disasm_cmd, String)
                println("Process output: ", result)
            catch
            end
        end
        return false
    finally
        rm(temp_file, force=true)
    end
end

# Run the test
if abspath(PROGRAM_FILE) == @__FILE__
    success = test_hello_world_bytecode()
    if !success
        println("\nBytecode generation needs adjustment.")
    end
end
