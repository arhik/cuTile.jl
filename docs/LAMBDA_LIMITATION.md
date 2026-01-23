# Lambda Limitation in cuTile MapReduce

## Overview

This document explains why anonymous functions (lambdas) cannot be used with `mapreduce` in cuTile, and provides workarounds.

## The Problem

```julia
# This fails:
mapreduce(x -> x + 1, +, tile, 2)  # ❌ Error: Unknown function call

# This works:
mapreduce(identity, +, tile, 2)    # ✅ Works
```

## Root Cause

### Julia's Closure Compilation

When Julia compiles `x -> x + 1`, it creates an opaque closure object that:
- Has no visible structure for static analysis
- Contains only a function pointer at the IR level
- Cannot be introspected by external tools

### Tile IR Requirements

Tile IR requires the compiler to:
1. Know the exact operation to perform (e.g., `cuda_tile.addf`)
2. Generate a custom body region with specific bytecode
3. Analyze the function structure for optimization

With an anonymous function, Tile IR sees only:
```
%fn = call @julia_closure_123()  # What is this?
```

No body, no structure, no way to generate Tile IR operations.

## Technical Details

### What Julia Generates

```julia
# Source code
x -> x + 1

# Compiled to (simplified):
closure = JITClosures.Closure{
    Tuple{Float32},
    Float32,
    typeof(+)}(parent, env, code)

# At IR level:
%1 = call %closure(%x)
```

### What Tile IR Needs

```mlir
%mapped = cuda_tile.addf(%elem, %const_one)
cuda_tile.yield %mapped
```

### The Gap

| Aspect | Julia Lambda | Tile IR Requirement |
|--------|-------------|---------------------|
| Structure | Opaque closure | Visible operation |
| Analysis | Not possible | Required for codegen |
| Operations | Unknown | Known set (addf, mulf, etc.) |
| Body | Hidden | Explicit bytecode |

## Workarounds

### 1. Use Named Functions

```julia
# Instead of:
mapreduce(x -> x + 1, +, tile, 2)

# Define:
add_one(x) = x + 1
mapreduce(add_one, +, tile, 2)
```

### 2. Use Built-in Functions

```julia
# Instead of:
mapreduce(x -> x * x, +, tile, 2)  # Lambda fails

# Use:
mapreduce(abs2, +, tile, 2)  # ✅ Built-in works
```

### 3. Compose Existing Functions

```julia
# Instead of:
mapreduce(x -> sin(x) + cos(x), +, tile, 2)

# Use:
sin_plus_cos(x) = sin(x) + cos(x)
mapreduce(sin_plus_cos, +, tile, 2)

# Or inline:
mapreduce(sin, +, tile, 2)  # Simple case
```

## Supported Named Functions

### Map Functions
| Function | Description | Bytecode |
|----------|-------------|----------|
| `identity` | No transformation | Direct element |
| `abs` | Absolute value | `cuda_tile.absf` |
| `abs2` | Square (`x * x`) | `cuda_tile.mulf` |
| `sqrt` | Square root | `cuda_tile.sqrtf` |
| `exp` | Exponential | `cuda_tile.exp` |
| `log` | Natural log | `cuda_tile.log` |
| `sin` | Sine | `cuda_tile.sin` |
| `cos` | Cosine | `cuda_tile.cos` |
| `neg` | Unary minus | `cuda_tile.negf` |

### Reduce Functions
| Function | Identity | Bytecode |
|----------|----------|----------|
| `+` | `0` | `cuda_tile.addf` |
| `*` | `1` | `cuda_tile.mulf` |
| `max` | `typemin(T)` | `cuda_tile.maxf` |
| `min` | `typemax(T)` | `cuda_tile.minf` |

## Expression Decomposition

Some expressions can be decomposed automatically:

```julia
# These work via decomposition:
mapreduce(x -> x + 1, +, tile, 2)  # → addf(elem, const)
mapreduce(x -> 2 * x, *, tile, 1)  # → mulf(const, elem)
mapreduce(x -> x^2, +, tile, 2)    # → mulf(elem, elem)
```

**Limitations of decomposition:**
- Only simple arithmetic expressions
- No branches or conditionals
- No function calls within the expression
- Limited to unary operations

## Future Solutions

### Macro-Based Approach (Potential)

```julia
# Could work with a macro that captures lambda at expand time
@mapreduce tile x -> x + 1 + axis=2
```

Requirements:
- Parse lambda at macro expansion time
- Generate appropriate Tile IR calls
- Integration with compiler pipeline

### Julia Compiler Integration

Deep integration with Julia's compiler could:
- Preserve lambda structure through compilation
- Emit structured IR before JIT compilation
- Provide callbacks for custom code generation

## Summary

| Approach | Status | Notes |
|----------|--------|-------|
| Named functions | ✅ Supported | `mapreduce(abs, +, tile, 1)` |
| Built-in functions | ✅ Supported | `mapreduce(abs2, +, tile, 1)` |
| Simple expressions | ⚠️ Partial | Decomposition when possible |
| Anonymous lambdas | ❌ Not supported | Fundamental limitation |

## Recommendation

**Always use named functions for production code:**

```julia
# Define at top level or in a let block
square(x) = x * x
add_offset(x) = x + 1.0f0

# Use in mapreduce
result = mapreduce(square, +, tile, 2)
```

This ensures:
1. Compatibility with Tile IR codegen
2. Clear, debuggable code
3. Potential for further optimization
4. Reusability across your codebase

## References

- [Tile IR Documentation](https://docs.nvidia.com/cuda/tile-ir/latest/)
- [Julia Closures](https://docs.julialang.org/en/v1/manual/functions/#Anonymous-Functions)
- [cuTile MapReduce Implementation](../MAPREDUCE_IMPLEMENTATION.md)