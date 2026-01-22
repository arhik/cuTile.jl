# 4. Binary Format — Tile IR

[Skip to main content](#main-content)

Back to top Ctrl+K

[![Tile IR - Home](../_static/nvidia-logo-horiz-rgb-blk-for-screen.svg) ![Tile IR - Home](../_static/nvidia-logo-horiz-rgb-wht-for-screen.svg)

Tile IR](../index.html)

Choose version

Search Ctrl+K

Search Ctrl+K

[![Tile IR - Home](../_static/nvidia-logo-horiz-rgb-blk-for-screen.svg) ![Tile IR - Home](../_static/nvidia-logo-horiz-rgb-wht-for-screen.svg)

Tile IR](../index.html)

Choose version

Table of Contents

-   [1. Introduction](introduction.html)
-   [2. Programming Model](prog_model.html)
-   [3. Syntax](syntax.html)
-   [4. Binary Format](#)
-   [5. Type System](types.html)
-   [6. Semantics](semantics.html)
-   [7. Memory Model](memory_model.html)
-   [8. Operations](operations.html)
-   [9. Debug Info](debug_info.html)
-   [10. Stability](stability.html)
-   [11. Appendix](appendix.html)
-   [12. Release Notes](release_notes.html)

-   [](../index.html)
-   4. Binary Format

# 4. Binary Format[#](#binary-format "Link to this heading")

The **Tile IR** bytecode is a binary representation of **Tile IR** modules including all items (globals, kernels, functions), attributes, and instructions.

The bytecode format is stable, versioned, and provides **Tile IR** portability guarantees. The bytecode format produced by older **Tile IR** compilers/drivers can be read by newer compilers/drivers. A compiler/driver accepts bytecode up to its supported **Tile IR** version.

The remainder of this section describes the **Tile IR** bytecode format and its encoding.

The bytecode has a few top-level design goals: - To represent a finite but expandable over time set of operations. - To use a minimal set of types for the encoding. - To enable manual construction and inspection by humans. - To allow lazy loading of functions. - To support forward and backward compatibility of the encoding.

The encoding of individual operations is described in [Operations](operations.html#section-operations).

## 4.1. Primitives[#](#primitives "Link to this heading")

The bytecode is composed of a handful of primitive types, each with their own encoding described below. The format uses these primitives to encode more complex data structures for representing the full set of **Tile IR** items.

### 4.1.1. Fixed-Width Integers[#](#fixed-width-integers "Link to this heading")

Fixed width integers are unsigned integers of a known size (in bytes). The values are stored in little-endian byte order.

**Note**: All multi-byte values in the **Tile IR** bytecode format use little-endian encoding, including fixed-width integers, array offsets, and type indices.

byte ::= `0x00`...`0xFF`

### 4.1.2. Variable-Width Integers[#](#variable-width-integers "Link to this heading")

Variable width integers, or `VarInt`s, provide a compact representation for integers. Each encoded `VarInt` consists of one to nine bytes, which together represent a single 64-bit value. The **Tile IR** bytecode utilizes the `PrefixVarInt` encoding for `VarInt`s. This encoding is a variant of the `LEB128` ("Little-Endian Base 128") encoding, where each byte of the encoding provides up to `7` bits for the value, with the remaining bit used to store a tag indicating the number of bytes used for the encoding. Small unsigned integers (less than `2^7`) may be stored in one byte, larger unsigned integers (up to `2^14`) may be stored in two bytes, and so on.

varint ::= `0x00`...`0xFF`

## 4.2. File Structure[#](#file-structure "Link to this heading")

The **Tile IR** bytecode file is composed of a stream of bytes; which is composed of a header followed by multiple sections. The header contains the 8-byte "magic number", and a version number.

Each section is expected to appear once within a bytecode file and is identified by a type code, the length, alignment, padding, and the payload. This design allows forward compatibility and selective parsing (tools can skip unknown sections, or unneeded sections).

bytecode {
  magic: "\x7FTileIR\x00",
  version: varint,
  sections: section[]
}

### 4.2.1. Magic Number[#](#magic-number "Link to this heading")

The **Tile IR** bytecode magic number consumes 8 bytes and is `\x7FTileIR\x00`. The magic number must be present at the beginning of the bytecode file to be accepted by the driver.

### 4.2.2. Version[#](#version "Link to this heading")

Following the magic number is the version of the bytecode format. The version allows both forwards and backwards compatibility of the format.

The version is encoded as three little-endian fixed-width fields immediately following the magic number:

version {
  major: uint8_t,    // Major version number
  minor: uint8_t,    // Minor version number
  tag: uint16_t      // Version tag (little-endian)
}

Each file encodes the version which allows parsers to adapt to the specific version. We do not break old versions of the format, if a newer file uses features an older parser can't interpret, it skips unknown optional features but will fail gracefully in cases where new features are unsupported. A new producer can also target older versions of the bytecode directly ensuring compatibility with older drivers.

#### Version Compatibility Guarantees[#](#version-compatibility-guarantees "Link to this heading")

1.  Backward Compatibility (a new deserializer can read old bytecode):
    
    -   The deserializer must support all previous versions of the bytecode format.
        
    -   The deserializer must handle all previously existing opcodes, types, and sections.
        
    -   The deserializer must maintain original program semantics.
        
2.  Forward Compatibility (an old deserializer can read new bytecode):
    
    -   The deserializer must fail only if it encounters unknown required features.
        
    -   The deserializer must only read bytecode up to its version.
        
    -   The deserializer must error with a clear message if there is a version mismatch.
        
3.  Version Targeting (a serializer can target specific older versions):
    
    -   The serializer may target one or more specific older versions.
        
    -   The serializer must validate all features are supported by the requested target version.
        
    -   The serializer must fail if attempting to serialize unsupported features.
        
4.  Optional vs Required Features (required changes are clearly documented):
    
    -   Required changes must be introduced in a new bytecode version and gracefully failure must be designed.
        
    -   Optional changes must be clearly documented and must preserve the above guarantees.
        

**Examples**:

-   For new ops → Add new opcodes
    
-   For changes to existing ops → Typically keep the old format for backward compatibility and add new opcode rather than modifying existing ones
    

### 4.2.3. Sections[#](#sections "Link to this heading")

The remainder of the bytecode stream is composed of sections. Sections are used to group data within the bytecode and allow operations on the stream to be per-section enabling out-of-order processing of data and/or lazy-loading. Each section contains a section ID, whose high bit indicates if the section has alignment requirements, a length and an optional alignment. A section ID is a 7-bit integer with the high bit indicating if the section has alignment requirements. When an alignment is present, a variable number of padding bytes (each byte = `0xCB`) may appear before the section data. The alignment of a section must be a power of 2. The alignment is represented as a `VarInt`. The padding ensures the section data starts at the specified alignment boundary.

section {
  idAndIsAligned: byte   // low 7 bits = section ID, high bit = alignment bit
  length: varint,
  alignment: varint?,    // present only if high bit was set
  padding: byte[],       // bytes of 0xCB as needed
  data: byte[]
}

#### Deserialization[#](#deserialization "Link to this heading")

The following are the implications of the section format on the deserialization process.

1.  The deserializer must read the section ID and length (`idAndIsAligned`) as a single byte.
    
2.  If the section ID has the high bit set, the deserializer must read the alignment (`alignment`) and apply the alignment by skipping `0xCB` bytes as needed.
    
3.  The deserializer must read the length (`length`) bytes of the payload. The length is a `VarInt`.
    
4.  The deserializer must move on to the next section until `EOF` (end of file).
    

#### String Section[#](#string-section "Link to this heading")

The string section holds all textual names used by the module to avoid repeating them inline. The section is encoded with the total number of strings, followed by the start index of each of the individual strings. The remaining encoding contains a single blob containing all the strings concatenated together. This design allows loading a specific string without reading the whole section. Strings in the bytecode are stored as raw byte sequences (in UTF-8 encoding) with an associated size. Finding the i-th string involves jumping to `stringStartIndex[i]` and reading until the next offset or end of the blob.

strings {
  numStrings: varint,
  stringStartIndex: uint32[],
  stringData: byte[]
}

#### Function Table Section[#](#function-table-section "Link to this heading")

The function table section enumerates the module's functions and embeds their code inline. First, `numFunctions` indicates how many functions follow; for each function `i`, `nameIndex[i]` is an index into the string section naming the function and `signatureIndex[i]` is an index into the Type section specifying its parameter, return types, global flags, etc (so multiple functions with identical signatures can share the same type entry). The field `functionLocIndex[i]` is a `VarInt` referencing an entry in the Debug Section that describes the function's definition location (e.g., source file and line). If `functionLocIndex[i]` is zero, there is no associated debug metadata for the function's definition scope. `lengthOfFunction[i]` states how many bytes of code belong to function `i`, and `functionBody[i]` contains exactly those bytes of instruction encodings. This layout avoids a separate code section and makes parsing each function straightforward: once you read the metadata for function `i`, you can either parse its instructions directly or skip them by advancing `lengthOfFunction[i]` bytes. The instruction encoding itself allows each operation to include a slot for its source location, referencing an entry in the debug section.

The instruction encodings (opcodes, operands, etc.) are described later under Operation Opcodes and Encodings Section.

The function table encoding has been updated to include function flags and optional optimization hints:

functionTable {
  numFunctions : varint
  // for each function i in [0..numFunctions-1]:
  nameIndex[i] : varint                    // References the function's name in the StringSec
  signatureIndex[i] : varint               // References the extended function signature in the TypeSec
  entryFlag[i] : byte                      // Function flags (visibility, kind, optimization hints)
  functionLocIndex[i] : varint             // 0 means no debug location
  optimizationHints[i]? : self-contained  // Present only if HasOptimizationHints flag is set
  lengthOfFunction[i] : varint
  functionBody[i] : byte[lengthOfFunction[i]]
}

The `entryFlag` byte encodes the following information:

-   **Bit 0 (0x01)**: Visibility Flag (0 = Public, 1 = Private)
    
-   **Bit 1 (0x02)**: Function Kind Flag (0 = Device Function, 1 = Kernel Entry Point)
    
-   **Bit 2 (0x04)**: Has Optimization Hints Flag (0 = No, 1 = Yes)
    
-   **Bits 3-7**: Reserved for future extensions
    

#### Constant Data Section[#](#constant-data-section "Link to this heading")

The constant data section holds large constants (e.g., dense tensors, large arrays) separately from code.

The current implementation does not impose explicit size limits on individual constants. Constants are stored using `uint64_t` offsets, allowing for very large constant data. However, practical limits may be imposed by available memory and any downstream compiler or runtime limitations (such as CUBIN generation constraints).

As we have done with the string section we move the constants into their own section to avoid bloating the function table section and allow for the lazy loading of specific constants when needed. Individual operations may reference constants by an index into this section. The section is encoded with the total number of constants, followed by the start index of each of the individual constants. The remaining encoding contains a single blob containing all the constants concatenated together.

constant {
  numConstants: varint,
  constantStartIndex: uint64_t[],
  constantData: byte[]
}

#### Type Section[#](#type-section "Link to this heading")

The type section stores all type definitions (scalar types, function signatures, parametric types, etc.) used in the module. The section begins with `numTypes` specifying how many types follow, then an array of offsets (`typeStartIndex`) into the encoded blob (`typeData`). To load the i-th type definition, you use the offset contained in `typeStartIndex[i]` to index into the `typeData` blob and parse the definition from there. The `typeData` blob is a single encoded binary blob containing the type definitions concatenated together.

type {
  numTypes: varint
  typeStartIndex: uint32_t[] // array of offsets, length = numTypes
  typeData: byte[]           // concatenated bytes for all type definitions
}

Each individual type definition will consist of a `typeTag` followed by a predefined payload structure specific to that `typeTag`. This deterministic approach allows parsers to understand the `payload` layout based solely on the `typeTag`.

typeTag : byte
payload : byte[lengthOfPayload]

-   `typeTag` indicates the "kind" of type. This can be a simple scalar, a function signature, or a more complex structure like a tensor.
    
-   `payload` is interpreted based on `typeTag`. For instance, a function signature might store the number of parameters, references to their types, etc, while a tensor type might store dimensions and an element type.
    

Example `typeTag` Values:

`typeTag` | Meaning | Expected Payload

0x01 | i1 (1-bit int) | None

0x02

i32 (32-bit int)

None

0x03

i64 (64-bit int)

None

0x04

tensor

`elementTypeIndex + rank + dimensions[]`

0x10

Function

Extended function signature

…

Future types

Depends on the type

#### Detailed Type Encodings[#](#detailed-type-encodings "Link to this heading")

The following sections describe the specific encoding format for each type tag:

**Integer Types** (`typeTag` = 0x00-0x04)

Integer types require no additional payload data beyond the type tag:

I1:   typeTag = 0x00  // 1-bit boolean
I8:   typeTag = 0x01  // 8-bit integer
I16:  typeTag = 0x02  // 16-bit integer
I32:  typeTag = 0x03  // 32-bit integer
I64:  typeTag = 0x04  // 64-bit integer

**Float Types** (`typeTag` = 0x05-0x0B)

Float types require no additional payload data beyond the type tag:

F16:      typeTag = 0x05  // 16-bit IEEE float
BF16:     typeTag = 0x06  // 16-bit bfloat
F32:      typeTag = 0x07  // 32-bit IEEE float
TF32:     typeTag = 0x08  // TensorFloat-32
F64:      typeTag = 0x09  // 64-bit IEEE float
F8E4M3FN: typeTag = 0x0A  // 8-bit float (E4M3FN format)
F8E5M2:   typeTag = 0x0B  // 8-bit float (E5M2 format)

**Pointer Type** (`typeTag` = 0x0C)

typeTag : byte = 0x0C      // Pointer type
pointeeTypeIndex : varint  // Index of the pointee type

**Tile Type** (`typeTag` = 0x0D)

typeTag : byte = 0x0D      // Tile type
elementTypeIndex : varint  // Index of the element type
shape : int64_t[]          // Shape dimensions (var-length array)

**TensorView Type** (`typeTag` = 0x0E)

typeTag : byte = 0x0E      // TensorView type
elementTypeIndex : varint  // Index of the element type
shape : int64_t[]          // Shape dimensions (var-length array)
strides : int64_t[]        // Stride values (var-length array)
indexTypeTag : byte        // Index type (I32=0x03 or I64=0x04)

**PartitionView Type** (`typeTag` = 0x0F)

typeTag : byte = 0x0F        // PartitionView type
tileShape : int32_t[]        // Tile shape (var-length array)
tensorViewTypeIndex : varint // Index of the TensorView type
dimMap : int32_t[]           // Dimension mapping (var-length array)
masked : byte                // Masked flag (0x00=false, 0x01=true)

**Function Type** (`typeTag` = 0x10)

typeTag : byte = 0x10 // Function Type
numParams : varint    // Number of input parameters
paramTypeIndices : varint[]  // Array of type indices for each parameter
numResults : varint   // Number of results
resultTypeIndices : varint[] // Array of type indices for each return value

The function type encoding stores only the essential type information (input and result types). Argument attributes and other function metadata are stored separately in the function table section.

**Token Type** (`typeTag` = 0x11)

typeTag : byte = 0x11  // Token type (no additional payload)

#### Attribute Encoding[#](#attribute-encoding "Link to this heading")

Attributes in **Tile IR** bytecode can be encoded in two ways:

1.  **Inline encoding** - Simple attributes are encoded directly in the instruction stream
    
2.  **Self-contained encoding** - Complex attributes include a type tag followed by their data
    

Self-contained attributes use the following format:

attributeTag : byte
attributeData : byte[]  // Format depends on attributeTag

The following attribute tags are supported:

**Integer Attribute** (`attributeTag` = 0x01)

attributeTag : byte = 0x01  // Integer attribute
typeIndex : varint          // Type index for the integer type
value : varint              // Integer value (zero-extended)

**Float Attribute** (`attributeTag` = 0x02)

attributeTag : byte = 0x02  // Float attribute
typeIndex : varint          // Type index for the float type
value : byte[]              // APFloat representation (variable length)

**Bool Attribute** (`attributeTag` = 0x03)

attributeTag : byte = 0x03  // Bool attribute
value : byte                // 0x00=false, 0x01=true

**Type Attribute** (`attributeTag` = 0x04)

attributeTag : byte = 0x04  // Type attribute
typeIndex : varint          // Index of the referenced type

**String Attribute** (`attributeTag` = 0x05)

attributeTag : byte = 0x05  // String attribute
stringIndex : varint        // Index into the string section

**Array Attribute** (`attributeTag` = 0x06)

attributeTag : byte = 0x06    // Array attribute
numElements : varint          // Number of elements
elements : self-contained[]   // Array of self-contained attributes

**DenseElements Attribute** (`attributeTag` = 0x07)

attributeTag : byte = 0x07  // DenseElements attribute
typeIndex : varint          // Type index for the shaped type
constantIndex : varint      // Index into constant section (for numeric data)
// OR for string data:
numStrings : varint         // Number of string elements
stringIndices : varint[]    // Indices into string section

**DivBy Attribute** (`attributeTag` = 0x08)

attributeTag : byte = 0x08    // DivBy attribute
divisor : varint              // Divisor value
flags : byte                  // Bit 0: unsignedInt, Bit 1: hasEvery, Bit 2: hasAlong
every : signed_varint?        // Present if Bit 1 set
along : signed_varint?        // Present if Bit 2 set

**SameElements Attribute** (`attributeTag` = 0x09)

attributeTag : byte = 0x09  // SameElements attribute
values : int64_t[]          // Array of int64 values (var-length)

**Dictionary Attribute** (`attributeTag` = 0x0A)

attributeTag : byte = 0x0A      // Dictionary attribute
numEntries : varint             // Number of key-value pairs
entries : dictEntry[]           // Array of dictionary entries

dictEntry {
  keyStringIndex : varint       // Index of key string
  value : self-contained        // Self-contained attribute value
}

**OptimizationHints Attribute** (`attributeTag` = 0x0B)

attributeTag : byte = 0x0B  // OptimizationHints attribute
dictionary : dictionary    // Dictionary attribute (without tag)

**NonNegative Attribute** (`attributeTag` = 0x0C)

attributeTag : byte = 0x0C  // NonNegative attribute
// No additional payload - presence indicates non-negative constraint

#### Global Section[#](#global-section "Link to this heading")

The global section stores module-level global variables. This section is optional and is only present if the module contains `cuda_tile.global` operations.

global {
  numGlobals: varint
  // for each global i in [0..numGlobals-1]:
  symbolNameIndex[i] : varint   // References the global's symbol name in the StringSec
  valueTypeIndex[i] : varint    // Type index for the global's value type
  constantValueIndex[i] : varint // Index into constant section for the global's initial value
}

Each global variable is encoded with:

-   **symbolNameIndex**: Index into the string section for the global's symbol name
    
-   **valueTypeIndex**: Index into the type section for the global's type (typically a shaped type like tensor)
    
-   **constantValueIndex**: Index into the constant section containing the global's initial value
    

#### Debug Section[#](#debug-section "Link to this heading")

The debug section stores the serialized debug information (for more details about debug information see [Debug Info](debug_info.html#section-debug-info)). This section is optional as certain tools may ignore it and serializers may leave it empty for release builds.

debug {
  diOpsNum: varint          // Total number of operations with debug info
  padding: bytes            // Align to 4 bytes
  diIndexOffsets: uint32_t[] // Per op offset into the debug info indices
  diIndicesNum: varint      // Total number of debug info indices
  padding: bytes            // Align to 4 bytes
  diIndices: uint64_t[]     // Array of debug indices to debug info attributes
  diAttrNum: varint         // Total number of debug info attributes
  padding: bytes            // Align to 4 bytes
  diOffsets: uint32_t[]     // Per debug info attribute offset into the debug info data
  diData: bytes             // Data for each debug info attribute
}

The debug section uses a multi-level indirection scheme:

1.  **Operations**: `diOpsNum` operations have debug info, with `diIndexOffsets` pointing into the indices array
    
2.  **Indices**: `diIndicesNum` total indices in `diIndices`, referencing debug info attributes
    
3.  **Attributes**: `diAttrNum` debug info attributes stored in `diData` with offsets in `diOffsets`
    

Each debug info attribute begins with a `debugEntryType` indicating what kind of debug info it is:

-   `0x00` = "Unknown"
    
-   `0x01` = "DICompileUnit"
    
-   `0x02` = "DIFile"
    
-   `0x03` = "DILexicalBlock"
    
-   `0x04` = "DILoc"
    
-   `0x05` = "DISubprogram"
    
-   `0x06` = "CallSite"
    

`debugEntryPayload` describes line, file, variable name, function index, instruction offset, etc. Each `debugEntryType` has a fixed, known structure. If `functionLocIndex[i]` or an instruction's `locationIndex` is non-zero, it references `debugEntryOffset[...]` in this section, whose payload can store file/line info or other metadata.

#### Debug Entry Payload Formats[#](#debug-entry-payload-formats "Link to this heading")

Each debug entry in `diData` begins with a `debugEntryType` byte followed by type-specific data:

debugEntry {
  debugEntryType : byte     // Identifies the debug info type
  entryData : byte[]        // Type-specific payload (format below)
}

The following debug entry types are supported:

**Unknown Debug Info** (`debugEntryType` = 0x00)

debugEntryType : varint = 0x00  // Unknown debug info
// No additional payload

**DICompileUnit** (`debugEntryType` = 0x01)

debugEntryType : varint = 0x01  // DICompileUnit
language : varint               // Source language identifier
fileIndex : varint              // Index of associated DIFile
producer : varint               // String index for compiler producer
optimized : byte                // 0x00=false, 0x01=true
emissionKind : varint           // Emission kind enumeration

**DIFile** (`debugEntryType` = 0x02)

debugEntryType : varint = 0x02  // DIFile
filename : varint               // String index for filename
directory : varint              // String index for directory

**DILexicalBlock** (`debugEntryType` = 0x03)

debugEntryType : varint = 0x03  // DILexicalBlock
line : varint                   // Line number
column : varint                 // Column number
scopeIndex : varint             // Index of parent scope (DIFile or DISubprogram)

**DILoc** (`debugEntryType` = 0x04)

debugEntryType : varint = 0x04  // DILoc (source location)
line : varint                   // Line number
column : varint                 // Column number
scopeIndex : varint             // Index of scope (DISubprogram, DILexicalBlock, etc.)
inlinedAtIndex : varint         // Index of inlined location (0 if not inlined)

**DISubprogram** (`debugEntryType` = 0x05)

debugEntryType : varint = 0x05  // DISubprogram (function debug info)
name : varint                   // String index for function name
linkageName : varint            // String index for linkage name
fileIndex : varint              // Index of associated DIFile
line : varint                   // Line number where function is defined
typeIndex : varint              // Index of function type
scopeLineIndex : varint         // Line number where scope begins
flags : varint                  // Function flags (visibility, etc.)
unitIndex : varint              // Index of associated DICompileUnit

**CallSite** (`debugEntryType` = 0x06)

debugEntryType : varint = 0x06  // CallSite location
calleeIndex : varint            // Index of called location
callerIndex : varint            // Index of calling location

## 4.3. Operation Opcodes and Encodings[#](#operation-opcodes-and-encodings "Link to this heading")

Each instruction in **Tile IR** bytecode is represented as:

opcode : byte
locationIndex : varint
instructionSpecificFields : byte[]

In this representation, `opcode` uniquely identifies the operation. This matches one of the operations defined in the **Tile IR** dialect. `locationIndex` is always present; if the instruction does not carry debug info, this field is 0. A non-zero value refers to an entry in the Debug Section that contains file, line, or other source-level metadata. The instruction-specific fields (operands, attributes, etc.) follow a layout defined by the opcode.

Additionally, we do not store a `resultIndex` for each producing instruction. Instead, we rely on a sequential pass to assign local value indices at parse-time.

Older parsers are expected to skip or reject unknown opcodes. Over time, new operations can be added simply by assigning new opcodes and defining their binary payload formats.

## 4.4. Operation Encoding Details[#](#operation-encoding-details "Link to this heading")

The general structure for operation encoding follows a consistent pattern, but varies based on the operation's characteristics:

**General Operation Structure**

opcode : byte                     // Operation identifier
locationIndex : varint            // Debug location (0 = no debug info)
resultTypes : typeIndex[]?        // Present for variadic result operations
flags : varint?                   // Optional flags for operations with optional fields
attributes : encoded_attr[]       // Operation-specific attributes
operands : operand_encoding       // Operation-specific operand encoding
regions : region_encoding[]?      // Present for operations with regions

**Flags Field Encoding**

For operations with optional attributes or operands, a flags field is used:

flags : varint                    // Bitfield encoding optional presence

The flags field uses individual bits to indicate the presence of optional attributes and operands:

-   **Bits 0-N**: Optional attributes (in declaration order)
    
-   **Bits N+1-M**: Optional operands (in declaration order)
    

**UnitAttr** attributes are encoded only in the flags field - no additional data is written.

**Operand Encoding Patterns**

Operands are encoded differently based on the operation's operand structure:

1.  **Fixed Operands**: Written as sequential operand indices
    
2.  **Variadic Operands**: Prefixed with operand count, then indices
    
3.  **AttrSizedOperandSegments**: Each operand group encoded separately
    

// Fixed operands (e.g., binary operations)
operand1Index : varint
operand2Index : varint

// Variadic operands (e.g., function calls)
numOperands : varint
operandIndices : varint[numOperands]

// AttrSizedOperandSegments (e.g., operations with optional operand groups)
group1 : optional_operand_group
group2 : optional_operand_group
...

**Result Type Encoding**

-   **Fixed Results**: No result types encoded (inferred from operation)
    
-   **Variadic Results**: Number of results encoded, followed by type indices
    

// Variadic results
numResults : varint
resultTypeIndices : varint[numResults]

**Region Encoding**

Operations with regions encode them after operands:

numRegions : varint
regions : region[numRegions]

region {
  numBlocks : varint
  blocks : block[numBlocks]
}

block {
  numArgs : varint
  argTypeIndices : varint[numArgs]
  numOps : varint
  operations : operation[numOps]
}

**Common Operation Examples**

**Arithmetic Operations** (e.g., `cuda_tile.add`)

opcode : byte                     // e.g., 0x15 for add
locationIndex : varint            // Debug location
lhs : varint                      // Left operand index
rhs : varint                      // Right operand index
// Result type inferred from operands

**Memory Operations** (e.g., `cuda_tile.load`)

opcode : byte                     // e.g., 0x20 for load
locationIndex : varint            // Debug location
resultType : varint               // Type index for loaded value
address : varint                  // Address operand index
// Optional attributes encoded via flags

**Control Flow Operations** (e.g., `cuda_tile.if`)

opcode : byte                     // e.g., 0x30 for if
locationIndex : varint            // Debug location
condition : varint                // Condition operand index
numRegions : varint               // Always 2 for if (then, else)
thenRegion : region               // Then block
     elseRegion : region               // Else block (may be empty)

## 4.5. Section Ordering[#](#section-ordering "Link to this heading")

**Tile IR** bytecode readers can handle sections in any order due to their flexible parsing design. The reader first discovers all sections and stores their payloads, then processes them in dependency order.

**Default Writing Order**

Writers typically emit sections in the following order:

1.  **Header** (magic number + version)
    
2.  **Global Section** (Section ID: 0x06) - Optional, only if globals present
    
3.  **Function Table Section** (Section ID: 0x02) - Required
    
4.  **Constant Data Section** (Section ID: 0x04) - Optional, only if constants present
    
5.  **Debug Section** (Section ID: 0x03) - Optional
    
6.  **Type Section** (Section ID: 0x05) - Required
    
7.  **String Section** (Section ID: 0x01) - Required
    
8.  **End-of-Bytecode Marker** (0x00) - Required
    

However, readers are not dependent on this order and can process sections regardless of their arrangement in the file. This flexibility enables future optimizations and different writing strategies.

**Reader Implementation**

The reader implements this flexibility by:

1.  **Discovery Phase**: Reads all section headers and stores their payloads in memory
    
2.  **Processing Phase**: Processes sections in dependency order regardless of file order
    
3.  **Lazy Resolution**: Resolves forward references (e.g., types, strings) on-demand
    

This design allows for efficient random access to any section and supports future file format optimizations.

**Section Dependencies**

-   Function table references: types, strings, constants, debug info
    
-   Global section references: types, strings, constants
    
-   Debug section references: strings
    
-   All sections may reference the string section
    

[

previous

3. Syntax



](syntax.html "previous page")[

next

5. Type System

](types.html "next page")

On this page

-   [4.1. Primitives](#primitives)
    -   [4.1.1. Fixed-Width Integers](#fixed-width-integers)
    -   [4.1.2. Variable-Width Integers](#variable-width-integers)
-   [4.2. File Structure](#file-structure)
    -   [4.2.1. Magic Number](#magic-number)
    -   [4.2.2. Version](#version)
        -   [Version Compatibility Guarantees](#version-compatibility-guarantees)
    -   [4.2.3. Sections](#sections)
        -   [Deserialization](#deserialization)
        -   [String Section](#string-section)
        -   [Function Table Section](#function-table-section)
        -   [Constant Data Section](#constant-data-section)
        -   [Type Section](#type-section)
        -   [Detailed Type Encodings](#detailed-type-encodings)
        -   [Attribute Encoding](#attribute-encoding)
        -   [Global Section](#global-section)
        -   [Debug Section](#debug-section)
        -   [Debug Entry Payload Formats](#debug-entry-payload-formats)
-   [4.3. Operation Opcodes and Encodings](#operation-opcodes-and-encodings)
-   [4.4. Operation Encoding Details](#operation-encoding-details)
-   [4.5. Section Ordering](#section-ordering)

[![NVIDIA](../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg) ![NVIDIA](../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

[Privacy Policy](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/) | [Your Privacy Choices](https://www.nvidia.com/en-us/about-nvidia/privacy-center/) | [Terms of Service](https://www.nvidia.com/en-us/about-nvidia/terms-of-service/) | [Accessibility](https://www.nvidia.com/en-us/about-nvidia/accessibility/) | [Corporate Policies](https://www.nvidia.com/en-us/about-nvidia/company-policies/) | [Product Security](https://www.nvidia.com/en-us/about-nvidia/product-security/) | [Contact](https://www.nvidia.com/en-us/contact/)

Copyright © 2025, NVIDIA.