# 9. Debug Info — Tile IR

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
-   [4. Binary Format](bytecode.html)
-   [5. Type System](types.html)
-   [6. Semantics](semantics.html)
-   [7. Memory Model](memory_model.html)
-   [8. Operations](operations.html)
-   [9. Debug Info](#)
-   [10. Stability](stability.html)
-   [11. Appendix](appendix.html)
-   [12. Release Notes](release_notes.html)

-   [](../index.html)
-   9. Debug Info

# 9. Debug Info[#](#debug-info "Link to this heading")

Debug info is essential for understanding and diagnosing the behavior of a **Tile IR** program. Debug info is metadata that maps generated code back to the original source code, creating a more informative and intuitive debugging experience for the **Tile IR** user and allowing the user to quickly identify and fix issues in a **Tile IR** program. This section explains how to include and use debug info in a **Tile IR** program when using the MLIR dialect. For a description of how to produce this information in the bytecode directly, see the [bytecode section](bytecode.html#section-bytecode).

**Tile IR** supports control flow based debugging of unoptimized code with features such as breakpoints, stepping through code and inspecting stack frames. **Tile IR** debug info exposes scope metadata describing the files, functions and blocks that comprise a **Tile IR** program along with information on how that program was compiled. This scope metadata is included in the **Tile IR** program by fusing it to location information. Details and examples below. **Tile IR** does not support value inspection of user variables, only control flow based debugging as described above is supported at this time.

## 9.1. Location Information[#](#location-information "Link to this heading")

**Tile IR** requires that scope metadata is attached to all operations within a **Tile IR** module to generate debug info of any kind. Scope metadata is required in addition to simple file, line, column location information. Providing simple file, line, column location information alone without scope metadata results in a very poor user experience. Scoped metadata is required to generate DWARF information and DWARF information is required by all developer tools including debuggers and profilers.

Example of a simple file, line, column location: `loc("/tmp/foo.py":1:4)`

Example of a location fused with scope metadata: `#cuda_tile.di_loc<loc("/tmp/foo.py":1:4) in #subprogram`

### 9.1.1. Location Types[#](#location-types "Link to this heading")

Two location types are supported by **Tile IR**:

1.  A **Tile IR** location represented by a `#cuda_tile.di_loc` attribute which combines file, line, column information with scope metadata as shown in the example above. This is the preferred method for generating location information fused with scope metadata.
    
2.  A `CallSiteLoc` where both the callee and caller are supported location types. This respresents a function call e.g. in the case where the **Tile IR** producer supports inlining.
    

Any location containing a supported location such as a `FusedLoc`, `NamedLoc` or `OpaqueLoc` will be converted to that supported location. If no supported location is found, the location will be converted to `UnknownLoc` indicating that the location is unknown or not available. All **Tile IR** global variables will have `UnknownLoc` given that `#cuda_tile.di_loc` supports local scope only and there is no support for `FileLineColLoc`.

Location Type

CudaTile Bytecode Support

CudaTile DILocAttr

Supported

CallSiteLoc

Supported if BOTH callee and caller are supported, else UnknownLoc

FusedLoc

Converted to first supported sub location, else UnknownLoc

NameLoc

Converted to child location if supported, else UnknownLoc

OpaqueLoc

Converted to fallback location if supported, else UnknownLoc

FileLineColLoc

Unsupported; always treated as UnknownLoc

### 9.1.2. Generating Scope Metadata[#](#generating-scope-metadata "Link to this heading")

Below are the options, trade-offs and mechanisms for generating scope metadata in **Tile IR** programs.

### 9.1.3. Options[#](#options "Link to this heading")

**Tile IR** producers have two options for generating location information fused with scope metadata:

1.  The **Tile IR** producer can generate location information fused with scope metadata directly using `#cuda_tile.di_loc` as shown in the example above.
    
2.  The **Tile IR** producer can generate a `loc` with file, line, column location information and then use a **Tile IR** pass (via `--synthesize-debug-info-scopes`) to synthesize scope metadata from that file, line, column location information.
    

Note

File, line, column location information must be converted to a **Tile IR** location prior to bytecode generation as all `FileLineColLoc` attributes are encoded as `UnknownLoc` attributes in the bytecode.

### 9.1.4. Trade-offs[#](#trade-offs "Link to this heading")

When deciding between the two options above, the **Tile IR** producer should consider the following trade-offs:

1.  Providing location information with scope metadata may be harder for the **Tile IR** producer to implement, however, the **Tile IR** producer retains full control to describe its source code to the compiler which may result in a smoother debugging experience for the end user.
    
2.  Providing simple file line, location information is very simple for the **Tile IR** producer to implement but cedes control to the pass which may not be able to accurately synthesize debug info for all cases possibly resulting in a degraded debugging experience for the end user. E.g. the pass may be able to synthesize function name, guess at linkage name, but not do a very good job of getting the line or column number of the function.
    

### 9.1.5. Mechanism[#](#mechanism "Link to this heading")

To synthesize scope metadata, the SynthesizeDebugInfoScopes pass can be added to any **Tile IR** pass pipeline or invoked via Python as follows:

pm = passmanager.PassManager(context=module.context)
full_pass_string = f"cuda_tile.module({"synthesize-debug-info-scopes"})"
pm = pm.parse(full_pass_string, context=module.context)
pm.run(module.operation.regions[0].blocks[0].operations[0])

## 9.2. Scope Metadata[#](#scope-metadata "Link to this heading")

**Tile IR** debug info is comprised of the following scope metadata expressed as attributes and fields:

### 9.2.1. Compile Unit[#](#compile-unit "Link to this heading")

A compile unit represents the root scope of all objects declared within a specific compilation unit. It specifies the source file associated with the compilation unit and encompasses all the other debug information elements such as files, lexical blocks, and subprograms. This scope is fundamental for organizing and correlating debug information with the original source code, facilitating efficient debugging.

Fields:

-   **File**: The source file associated with the compilation unit.
    

The following fields are set by the **Tile IR** compiler and listed for reference:

-   **Is Optimized?**: Set to `false` if compiler option `--opt-level` is `0` else `true`.
    
-   **Emission Kind**: Set based on compiler options: `--line-info` to generate line tables only or `--device-debug` alias `-g` for generating full debug info.
    

### 9.2.2. File[#](#file "Link to this heading")

A file represents a source file used in the compilation process. It specifies the file name and directory of the source file. This information is used to map the generated code back to the original source, allowing developers to trace and debug code effectively.

Fields:

-   **Name**: The name of the file.
    
-   **Directory**: The directory containing the file.
    

### 9.2.3. Lexical Block[#](#lexical-block "Link to this heading")

A lexical block represents a nested scope within a subprogram, specifying the scope, file, line number, and optional column number of the block. Lexical blocks are used to represent various nested scopes in the source code, such as conditional statements, loops, and other code blocks. This detailed scope information helps in understanding the program's structure and control flow during debugging.

Fields:

-   **Scope**: The scope in which the lexical block is defined.
    
-   **File**: The source file containing the lexical block.
    
-   **Line**: The line number where the lexical block starts.
    
-   **Column**: The column number where the lexical block starts.
    

### 9.2.4. Subprogram[#](#subprogram "Link to this heading")

A subprogram represents a function within the source language. It specifies the scope, file, line number, name, and linkage name of the subprogram. Optionally, it can include the line number within the scope. Subprogram information is used for function level debugging, allowing developers to set breakpoints, step through code, and inspect stack frames within specific functions.

Fields:

-   **File**: The source file containing the subprogram.
    
-   **Line**: The line number where the subprogram starts.
    
-   **Name**: The name of the subprogram.
    
-   **Linkage Name**: The linkage name of the subprogram.
    
-   **Compile Unit**: The compilation unit containing the subprogram.
    
-   **Scope Line**: The line number where the scope of the subprogram starts. [Optional]
    

The following fields are set by the **Tile IR** compiler and listed for reference:

-   **Subroutine Type**: Set to `() -> ()` meaning that all functions appear to have no arguments and no return value.
    

## 9.3. Example Usage[#](#example-usage "Link to this heading")

Below is an example showing how to use **Tile IR** debug info:

#file = #cuda_tile.di_file<"foo.py" in "/tmp/">

#compile_unit = #cuda_tile.di_compile_unit<
  file = #file
>

#subprogram = #cuda_tile.di_subprogram<
  file = #file,
  line = 1,
  name = "test_kernel",
  linkageName = "kernel",
  compileUnit = #compile_unit,
  scopeLine = 1
>

#block = #cuda_tile.di_lexical_block<
  scope = #subprogram,
  file = #file,
  line = 1,
  column = 4
>

#loc_fn = #cuda_tile.di_loc<loc("/tmp/foo.py":1:4) in #subprogram>
#loc_return = #cuda_tile.di_loc<loc("/tmp/foo.py":5:4) in #block>

cuda_tile.module @kernels attributes { } {
  entry @kernel() {
    return loc(#loc_return)
  } loc(#loc_fn)
}

[

previous

8. Operations



](operations.html "previous page")[

next

10. Stability

](stability.html "next page")

On this page

-   [9.1. Location Information](#location-information)
    -   [9.1.1. Location Types](#location-types)
    -   [9.1.2. Generating Scope Metadata](#generating-scope-metadata)
    -   [9.1.3. Options](#options)
    -   [9.1.4. Trade-offs](#trade-offs)
    -   [9.1.5. Mechanism](#mechanism)
-   [9.2. Scope Metadata](#scope-metadata)
    -   [9.2.1. Compile Unit](#compile-unit)
    -   [9.2.2. File](#file)
    -   [9.2.3. Lexical Block](#lexical-block)
    -   [9.2.4. Subprogram](#subprogram)
-   [9.3. Example Usage](#example-usage)

[![NVIDIA](../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg) ![NVIDIA](../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

[Privacy Policy](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/) | [Your Privacy Choices](https://www.nvidia.com/en-us/about-nvidia/privacy-center/) | [Terms of Service](https://www.nvidia.com/en-us/about-nvidia/terms-of-service/) | [Accessibility](https://www.nvidia.com/en-us/about-nvidia/accessibility/) | [Corporate Policies](https://www.nvidia.com/en-us/about-nvidia/company-policies/) | [Product Security](https://www.nvidia.com/en-us/about-nvidia/product-security/) | [Contact](https://www.nvidia.com/en-us/contact/)

Copyright © 2025, NVIDIA.