
# MapReduce Implementation for cuTile

This document describes the `mapreduce` function implementation for cuTile's Tile IR compilation pipeline.

## Overview

The `mapreduce` function combines a map (element-wise transformation) operation with a reduction operation in a single pass over tile data. This is more efficient than separate `map` and `reduce` calls because:

1. No intermediate collection needs to be created
2. Elements are processed only once
3. Memory bandwidth is used more efficiently

## API

### Primary Interface

```julia
mapreduce(f, op, tile::Tile{T, S}, axis::Integer; init=nothing) -> Tile{T, reduced_shape}
mapreduce(f, op, tile::Tile{T, S}, ::Val{axis}; init=nothing) -> Tile{T, reduced_shape}
```

### Arguments

| Parameter | Type | Description |
|-----------|------|-------------|
| `f` | `Function` | Unary function applied to each element before reduction |
| `op` | `Function` | Binary function used for reduction (must be associative) |
| `tile` | `Tile{T, S}` | Input tile to process |
| `axis` | `Integer` or `Val` | Axis to reduce along (1-indexed) |
| `init` | Optional | Initial value for reduction (identity element) |

### Returns

A `Tile{T, reduced_shape}` where `reduced_shape` has one fewer dimension than the input (the reduced axis is removed).

### Supported Map Functions

| Function | Description | Bytecode Generated |
|----------|-------------|-------------------|
| `identity` | Return element unchanged | Direct use of element |
| `abs` | Absolute value | `AbsFOp` |
| `abs2` | Square (`x * x`) | `MulFOp(elem, elem)` |
| `sqrt` | Square root | `SqrtOp` |
| `exp` | Exponential (`e^x`) | `ExpOp` |
| `log` | Natural logarithm | `LogOp` |
| `sin` | Sine | `SinOp` |
| `cos` | Cosine | `CosOp` |
| `-` or `neg` | Unary minus | `NegFOp` |

### Supported Reduce Functions

| Function | Description | Identity | Bytecode Generated |
|----------|-------------|----------|-------------------|
| `+` | Addition | `0.0` or `0` | `AddFOp` / `AddIOp` |
| `*` | Multiplication | `1.0` or `1` | `MulFOp` / `MulIOp` |
| `max` | Maximum | `typemin(T)` | `MaxFOp` / `MaxIOp` |
| `min` | Minimum | `typemax(T)` | `MinFOp` / `MinIOp` |

## Examples

### Sum of Squares

```julia
using cuTile

# Compute Σ(x²) along axis 2
tile = ct.Tile{Float32, (4, 16)}()
result = ct.mapreduce(x -> x * x, +, tile, 2)

# In a full kernel:
ct.code_tiled(Tuple{ct.TileArray{Float32,2}, ct.TileArray{Float32,1}}) do a, b
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    # Reduce (4, 16) → (4,) by summing squares
    result = ct.mapreduce(x -> x * x, +, tile, 2)
    ct.store(b, pid, result)
    return
end
```

### Max of Absolute Values

```julia
# Compute max(|x|) along axis 1
tile = ct.Tile{Float32, (8, 8)}()
result = ct.mapreduce(abs, max, tile, 1)

# Output shape: (1, 8) - max along first dimension
```

### Product with Custom Init

```julia
# Compute ∏(x + 1) with init=1
tile = ct.Tile{Float32, (4, 16)}()
result = ct.mapreduce(x -> x + 1, *, tile, 2; init=1.0f0)
```

### Using Named Functions

```julia
# Using abs2 explicitly (more efficient than x -> x*x)
tile = ct.Tile{Float32, (4, 16)}()
result = ct.mapreduce(abs2, +, tile, 2)  # Same as x*x

# Using Val{axis} syntax
result = ct.mapreduce(abs, max, tile, ct.Val(0))  # Along axis 1
```

## Implementation Details

### Bytecode Generation

The `mapreduce` function is compiled to Tile IR using the following pattern:

1. **Input Processing**: The input tile is received as a multi-dimensional tile
2. **ReduceOp Creation**: A `cuda_tile.reduce` operation is emitted with:
   - Input tile operand
   - Reduction axis
   - Identity value for the reduction function
   - Custom body region
3. **Body Region**: The body callback applies:
   ```julia
   # Pseudocode for reduction body
   mapped = f(element)      # Map function
   result = op(accumulator, mapped)  # Reduce function
   ```

### Type System

The implementation handles:
- **Float types**: Float32, Float64 (uses floating-point operations)
- **Integer types**: Int8, Int16, Int32, Int64 (uses integer operations with signedness)
- **Unsigned types**: UInt8, UInt16, UInt32, UInt64

Identity values are computed based on the element type:
- For `+`: `0.0` (floats) or `0` (integers)
- For `*`: `1.0` (floats) or `1` (integers)
- For `max`: `typemin(T)` (smallest representable value)
- For `min`: `typemax(T)` (largest representable value)

### Axis Handling

- **1-indexed**: The public API uses 1-indexed axes (matching Julia's `reduce`)
- **0-indexed**: Internally converted to 0-indexed for Tile IR
- **Val{axis}**: Compile-time axis can be passed as `Val(n)` for optimization

## Comparison with Julia's mapreduce

| Feature | Julia Base | cuTile mapreduce |
|---------|-----------|------------------|
| Map functions | Any function | Named functions only |
| Reduce functions | Any binary function | `+`, `*`, `max`, `min` |
| Associativity | Implementation-defined | GPU-dependent |
| Empty collections | Uses `init` | Requires `init` |
| Performance | May create intermediate arrays | Single-pass, no intermediates |

## Limitations

### Current Limitations

1. **Named functions only**: Anonymous functions (`x -> x^2`) require special handling
2. **Limited reduce ops**: Only `+`, `*`, `max`, `min` are supported
3. **No init with bytecode**: Runtime `init` values not yet supported in bytecode
4. **No user-defined types**: Only primitive numeric types

### Future Enhancements

1. Support for custom user-defined reduction functions
2. Runtime `init` value support via additional parameter passing
3. Multi-output reductions (map to multiple values, then reduce each)
4. Predicate-based filtering during reduction
5. Integration with atomic operations for concurrent reductions

## Testing

### Quick Verification

```julia
julia> include("test_mapreduce_mwe.jl")
julia> TestMapReduceMWE.quick_verify()
```

### Full Test Suite

```julia
julia> include("test_mapreduce_mwe.jl")
julia> TestMapReduceMWE.test_all()
```

### Integration Tests

Tests are located in `test/test_mapreduce.jl` and `test/codegen.jl`.

## Bytecode Structure

For a mapreduce operation like `mapreduce(abs, max, tile, 1)`:

```
cuda_tile.module {
  cuda_tile.func @entry(...) {
    ...
    %result = cuda_tile.reduce(%input, axis=0, identities=[typemin<f32>]) {
    ^body(%acc: f32, %elem: f32):
      %mapped = cuda_tile.absf(%elem)      // Map: abs(elem)
      %result = cuda_tile.maxf(%acc, %mapped)  // Reduce: max(acc, mapped)
      cuda_tile.yield %result
    }
    ...
  }
}
```

## Performance Considerations

1. **Fused operations**: Map and reduce are fused, eliminating intermediate storage
2. **Coalesced access**: Reduction along specific axis enables memory coalescing
3. **Warp-level primitives**: GPU implementation can use warp shuffles for efficient reduction
4. **Identity handling**: Proper identity values ensure correctness for edge cases

## Debugging

### Type Instabilities

Check for type instabilities using:

```julia
julia> @code_warntype ct.mapreduce(abs, +, tile, 1)
```

### Bytecode Inspection

View the generated Tile IR:

```julia
julia> ct.@code_tiled (Tuple{...}) do ...
           # kernel code
       end
```

### Common Errors

1. **"Unsupported map function"**: Function not in supported list
2. **"Reduction axis must be a compile-time constant"**: Axis must be literal or Val
3. **"Cannot resolve input tile"**: Tile type inference failed

## Related Functions

- `reduce_sum(tile, axis)` - Sum reduction (equivalent to `mapreduce(identity, +, tile, axis)`)
- `reduce_max(tile, axis)` - Maximum reduction
- `ct.load(array, pid, shape)` - Load data into tile
- `ct.store(array, pid, tile)` - Store tile to array

## References

- [Tile IR Documentation](https://docs.nvidia.com/cuda/tile-ir/latest/)
- [Julia Base.mapreduce](https://docs.julialang.org/en/v1/base/collections/#Base.mapreduce)
- [cuTile Repository](https://github.com/arhik/cuTile)

```

```

<minimax:tool_call>
<invoke name="edit_file">
<parameter name="display_description">Create mapreduce usage documentation</parameter>
<parameter name="mode">create</parameter>
<parameter name="text"># MapReduce Implementation for cuTile

This document describes the `mapreduce` function implementation for cuTile's Tile IR compilation pipeline.

## Overview

The `mapreduce` function combines a map (element-wise transformation) operation with a reduction operation in a single pass over tile data. This is more efficient than separate `map` and `reduce` calls because:

1. No intermediate collection needs to be created
2. Elements are processed only once
3. Memory bandwidth is used more efficiently

## API

### Primary Interface

```julia
mapreduce(f, op, tile::Tile{T, S}, axis::Integer; init=nothing) -> Tile{T, reduced_shape}
mapreduce(f, op, tile::Tile{T, S}, ::Val{axis}; init=nothing) -> Tile{T, reduced_shape}
```

### Arguments

| Parameter | Type | Description |
|-----------|------|-------------|
| `f` | `Function` | Unary function applied to each element before reduction |
| `op` | `Function` | Binary function used for reduction (must be associative) |
| `tile` | `Tile{T, S}` | Input tile to process |
| `axis` | `Integer` or `Val` | Axis to reduce along (1-indexed) |
| `init` | Optional | Initial value for reduction (identity element) |

### Returns

A `Tile{T, reduced_shape}` where `reduced_shape` has one fewer dimension than the input (the reduced axis is removed).

### Supported Map Functions

| Function | Description | Bytecode Generated |
|----------|-------------|-------------------|
| `identity` | Return element unchanged | Direct use of element |
| `abs` | Absolute value | `AbsFOp` |
| `abs2` | Square (`x * x`) | `MulFOp(elem, elem)` |
| `sqrt` | Square root | `SqrtOp` |
| `exp` | Exponential (`e^x`) | `ExpOp` |
| `log` | Natural logarithm | `LogOp` |
| `sin` | Sine | `SinOp` |
| `cos` | Cosine | `CosOp` |
| `-` or `neg` | Unary minus | `NegFOp` |

### Supported Reduce Functions

| Function | Description | Identity | Bytecode Generated |
|----------|-------------|----------|-------------------|
| `+` | Addition | `0.0` or `0` | `AddFOp` / `AddIOp` |
| `*` | Multiplication | `1.0` or `1` | `MulFOp` / `MulIOp` |
| `max` | Maximum | `typemin(T)` | `MaxFOp` / `MaxIOp` |
| `min` | Minimum | `typemax(T)` | `MinFOp` / `MinIOp` |

## Examples

### Sum of Squares

```julia
using cuTile

# Compute Σ(x²) along axis 2
tile = ct.Tile{Float32, (4, 16)}()
result = ct.mapreduce(x -> x * x, +, tile, 2)

# In a full kernel:
ct.code_tiled(Tuple{ct.TileArray{Float32,2}, ct.TileArray{Float32,1}}) do a, b
    pid = ct.bid(1)
    tile = ct.load(a, pid, (4, 16))
    # Reduce (4, 16) → (4,) by summing squares
    result = ct.mapreduce(x -> x * x, +, tile, 2)
    ct.store(b, pid, result)
    return
end
```

### Max of Absolute Values

```julia
# Compute max(|x|) along axis 1
tile = ct.Tile{Float32, (8, 8)}()
result = ct.mapreduce(abs, max, tile, 1)

# Output shape: (1, 8) - max along first dimension
```

### Product with Custom Init

```julia
# Compute ∏(x + 1) with init=1
tile = ct.Tile{Float32, (4, 16)}()
result = ct.mapreduce(x -> x + 1, *, tile, 2; init=1.0f0)
```

### Using Named Functions

```julia
# Using abs2 explicitly (more efficient than x -> x*x)
tile = ct.Tile{Float32, (4, 16)}()
result = ct.mapreduce(abs2, +, tile, 2)  # Same as x*x

# Using Val{axis} syntax
result = ct.mapreduce(abs, max, tile, ct.Val(0))  # Along axis 1
```

## Implementation Details

### Bytecode Generation

The `mapreduce` function is compiled to Tile IR using the following pattern:

1. **Input Processing**: The input tile is received as a multi-dimensional tile
2. **ReduceOp Creation**: A `cuda_tile.reduce` operation is emitted with:
   - Input tile operand
   - Reduction axis
   - Identity value for the reduction function
   - Custom body region
3. **Body Region**: The body callback applies:
   ```julia
   # Pseudocode for reduction body
   mapped = f(element)      # Map function
   result = op(accumulator, mapped)  # Reduce function
   ```

### Type System

The implementation handles:
- **Float types**: Float32, Float64 (uses floating-point operations)
- **Integer types**: Int8, Int16, Int32, Int64 (uses integer operations with signedness)
- **Unsigned types**: UInt8, UInt16, UInt32, UInt64

Identity values are computed based on the element type:
- For `+`: `0.0` (floats) or `0` (integers)
- For `*`: `1.0` (floats) or `1` (integers)
- For `max`: `typemin(T)` (smallest representable value)
- For `min`: `typemax(T)` (largest representable value)

### Axis Handling

- **1-indexed**: The public API uses 1-indexed axes (matching Julia's `reduce`)
- **0-indexed**: Internally converted to 0-indexed for Tile IR
- **Val{axis}**: Compile-time axis can be passed as `Val(n)` for optimization

## Comparison with Julia's mapreduce

| Feature | Julia Base | cuTile mapreduce |
|---------|-----------|------------------|
| Map functions | Any function | Named functions only |
| Reduce functions | Any binary function | `+`, `*`, `max`, `min` |
| Associativity | Implementation-defined | GPU-dependent |
| Empty collections | Uses `init` | Requires `init` |
| Performance | May create intermediate arrays | Single-pass, no intermediates |

## Limitations

### Current Limitations

1. **Named functions only**: Anonymous functions (`x -> x^2`) require special handling
2. **Limited reduce ops**: Only `+`, `*`, `max`, `min` are supported
3. **No init with bytecode**: Runtime `init` values not yet supported in bytecode
4. **No user-defined types**: Only primitive numeric types

### Future Enhancements

1. Support for custom user-defined reduction functions
2. Runtime `init` value support via additional parameter passing
3. Multi-output reductions (map to multiple values, then reduce each)
4. Predicate-based filtering during reduction
5. Integration with atomic operations for concurrent reductions

## Testing

### Quick Verification

```julia
julia> include("test_mapreduce_mwe.jl")
julia> TestMapReduceMWE.quick_verify()
```

### Full Test Suite

```julia
julia> include("test_mapreduce_mwe.jl")
julia> TestMapReduceMWE.test_all()
```

### Integration Tests

Tests are located in `test/test_mapreduce.jl` and `test/codegen.jl`.

## Bytecode Structure

For a mapreduce operation like `mapreduce(abs, max, tile, 1)`:

```
cuda_tile.module {
  cuda_tile.func @entry(...) {
    ...
    %result = cuda_tile.reduce(%input, axis=0, identities=[typemin<f32>]) {
    ^body(%acc: f32, %elem: f32):
      %mapped = cuda_tile.absf(%elem)      // Map: abs(elem)
      %result = cuda_tile.maxf(%acc, %mapped)  // Reduce: max(acc, mapped)
      cuda_tile.yield %result
    }
    ...
  }
}
```

## Performance Considerations

1. **Fused operations**: Map and reduce are fused, eliminating intermediate storage
2. **Coalesced access**: Reduction along specific axis enables memory coalescing
3. **Warp-level primitives**: GPU implementation can use warp shuffles for efficient reduction
4. **Identity handling**: Proper identity values ensure correctness for edge cases

## Debugging

### Type Instabilities

Check for type instabilities using:

```julia
julia> @code_warntype ct.mapreduce(abs, +, tile, 1)
```

### Bytecode Inspection

View the generated Tile IR:

```julia
julia> ct.@code_tiled (Tuple{...}) do ...
           # kernel code
       end
```

### Common Errors

1. **"Unsupported map function"**: Function not in supported list
2. **"Reduction axis must be a compile-time constant"**: Axis must be literal or Val
3. **"Cannot resolve input tile"**: Tile type inference failed

## Related Functions

- `reduce_sum(tile, axis)` - Sum reduction (equivalent to `mapreduce(identity, +, tile, axis)`)
- `reduce_max(tile, axis)` - Maximum reduction
- `ct.load(array, pid, shape)` - Load data into tile
- `ct.store(array, pid, tile)` - Store tile to array

## References

- [Tile IR Documentation](https://docs.nvidia.com/cuda/tile-ir/latest/)
- [Julia Base.mapreduce](https://docs.julialang.org/en/v1/base/collections/#Base.mapreduce)
- [cuTile Repository](https://github.com/arhik/cuTile)</parameter>
<parameter name="path">cuTile/MAPREDUCE_IMPLEMENTATION.md</parameter>
</invoke>
</minimax:tool_call>