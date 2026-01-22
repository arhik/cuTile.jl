# Debug script to understand section ordering and alignment in cuTile bytecode

using cuTile

"""
    encode_varint!(buf::Vector{UInt8}, x::Integer)

Encode an unsigned integer using variable-length encoding (LEB128-style).
Each byte uses 7 bits for data and 1 bit to indicate continuation.
"""
function encode_varint!(buf::Vector{UInt8}, x::Integer)
    @assert x >= 0 "Varint encoding requires non-negative integers, got $x"
    # Handle zero specially
    if x == 0
        push!(buf, 0x00)
        return buf
    end
    while x > 0x7f
        push!(buf, UInt8((x & 0x7f) | 0x80))
        x >>= 7
    end
    push!(buf, UInt8(x))
    return buf
end

"""
    analyze_bytecode(bytecode::Vector{UInt8})

Analyze a bytecode buffer to identify sections and their properties.
"""
function analyze_bytecode(bytecode::Vector{UInt8})
    println("=== Bytecode Analysis ===")
    println("Total length: $(length(bytecode)) bytes")

    pos = 1

    # Check magic header
    if pos + 7 <= length(bytecode)
        magic = bytecode[pos:pos+7]
        println("\nMagic header: $(String(magic))")
        pos += 8
    else
        println("Error: Not enough bytes for magic header")
        return
    end

    # Check version
    if pos + 3 <= length(bytecode)
        version = bytecode[pos:pos+3]
        println("Version: $(bytes2hex(version))")
        pos += 4
    else
        println("Error: Not enough bytes for version")
        return
    end

    # Analyze sections
    section_num = 1
    while pos <= length(bytecode)
        println("\n--- Section $section_num at position $pos ---")

        # Read section header byte
        if pos > length(bytecode)
            break
        end

        header_byte = bytecode[pos]
        section_id = header_byte & 0x7f  # Low 7 bits
        has_alignment = (header_byte & 0x80) != 0  # High bit
        println("Header byte: $(string(header_byte, base=16)) (binary: $(string(header_byte, base=2)))")
        println("Section ID: $section_id")
        println("Has alignment flag: $has_alignment")
        pos += 1

        # Check for end marker
        if section_id == 0
            println("End of bytecode marker found")
            break
        end

        # Read length as varint
        length_val = 0
        shift = 0
        continue_reading = true

        while continue_reading && pos <= length(bytecode)
            byte_val = bytecode[pos]
            length_val |= ((byte_val & 0x7f) << shift)
            shift += 7
            pos += 1
            continue_reading = (byte_val & 0x80) != 0
        end

        println("Section length: $length_val")

        # Check alignment if present
        alignment = 1
        if has_alignment
            alignment = 0
            shift = 0
            continue_reading = true

            while continue_reading && pos <= length(bytecode)
                byte_val = bytecode[pos]
                alignment |= ((byte_val & 0x7f) << shift)
                shift += 7
                pos += 1
                continue_reading = (byte_val & 0x80) != 0
            end

            println("Section alignment: $alignment")
        end

        # Check padding before data
        if has_alignment && alignment > 1
            padding_bytes = 0
            while pos <= length(bytecode) && bytecode[pos] == 0xCB
                padding_bytes += 1
                pos += 1
            end
            println("Padding bytes: $padding_bytes (0xCB repeated)")

            # Verify alignment
            actual_pos = pos % alignment
            println("Actual position mod alignment: $(actual_pos)")
            if actual_pos != 1  # 1-based indexing
                println("WARNING: Position not aligned correctly!")
            end
        end

        # Skip section data
        data_start = pos
        data_end = min(pos + length_val - 1, length(bytecode))

        if pos <= length(bytecode)
            # Show first few bytes of data
            sample_size = min(16, length_val)
            if data_start + sample_size - 1 <= length(bytecode)
                sample = bytecode[data_start:data_start+sample_size-1]
                println("Data sample (first $sample_size bytes): $(bytes2hex(sample))")
            end

            pos = data_end + 1
        end

        section_num += 1

        # Safety check to avoid infinite loop
        if section_num > 10
            println("WARNING: Too many sections, breaking to avoid infinite loop")
            break
        end
    end
end

"""
    generate_minimal_bytecode()

Generate a minimal but correctly structured bytecode for testing.
"""
function generate_minimal_bytecode()
    buf = Vector{UInt8}()

    # 1. Magic Header (8 bytes)
    append!(buf, b"\x7FTileIR\x00")

    # 2. Version (4 bytes: major=13, minor=1, tag=0)
    push!(buf, 0x0d)  # Major version 13
    push!(buf, 0x01)  # Minor version 1
    push!(buf, 0x00)  # Tag low
    push!(buf, 0x00)  # Tag high

    # 3. Function Table Section (ID=2, required, with 8-byte alignment)
    push!(buf, 0x82)  # ID=2 with high bit set (alignment required)

    # Empty function table
    func_data = UInt8[]
    encode_varint!(func_data, 0)  # No functions

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

    # Function data (empty)
    append!(buf, func_data)

    # 4. Constant Data Section (ID=4, optional, with 8-byte alignment)
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
    push!(buf, 0x85)  # ID=5 with high bit set (alignment required)

    # Minimal type data
    type_data = UInt8[]
    encode_varint!(type_data, 0)  # No types

    # Section length
    encode_varint!(buf, length(type_data))
    # Section alignment
    encode_varint!(buf, 4)  # 4-byte alignment

    # Add padding to ensure 4-byte alignment for type data
    current_pos = length(buf) + 1  # Position after alignment byte
    padding_needed = (4 - current_pos % 4) % 4
    for _ in 1:padding_needed
        push!(buf, 0xCB)  # Padding byte
    end

    # Type data (empty)
    append!(buf, type_data)

    # 7. String Section (ID=1, required, with 4-byte alignment)
    push!(buf, 0x81)  # ID=1 with high bit set (alignment required)

    # Empty string data
    string_data = UInt8[]
    encode_varint!(string_data, 0)  # No strings

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

    # String data (empty)
    append!(buf, string_data)

    # 8. End-of-Bytecode Marker (required)
    push!(buf, 0x00)

    return buf
end

# Test function
function test_minimal_bytecode()
    println("Generating minimal TileIR bytecode...")
    bytecode = generate_minimal_bytecode()

    println("Bytecode generated: $(length(bytecode)) bytes")
    println("First 64 bytes: ", bytes2hex(bytecode[1:min(64, length(bytecode))]))
    println("\nFull bytecode (hex): ", bytes2hex(bytecode))

    # Analyze the bytecode structure
    analyze_bytecode(bytecode)

    # Write to temporary file and try to disassemble
    temp_file = tempname() * ".tile"
    write(temp_file, bytecode)

    try
        # Use cuTile's translate tool to disassemble
        println("\n\n=== Attempting Disassembly ===")
        disasm_cmd = `$(cuTile.cuda_tile_translate()) --cudatilebc-to-mlir $temp_file`
        disasm = read(disasm_cmd, String)

        println("SUCCESS! Disassembled output:")
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
    success = test_minimal_bytecode()
    if !success
        println("\nMinimal bytecode generation needs adjustment.")
    end
end
