
# MapReduce Function

Implement a generalized `mapreduce` function for cuTile that combines element-wise mapping with reduction operations.

## Summary

Added a new `mapreduce` function that:
1. Applies a unary function `f` to each element of a tile
2. Reduces the result using a binary function `op`
3. Eliminates intermediate collection creation (fused operation)

## Files Changed

1. **src/language/operations.jl**
   - Added `mapreduce` to public exports
   - Added user-facing API with documentation
   - Supports `axis::Integer` and `Val{axis}` syntax

2. **src/compiler/intrinsics/core.jl**
   - Added `emit_mapreduce!()` for bytecode generation
   - Added `determine_reduction_fn()` to identify reduction function
   - Added `emit_map_body()` for map function bytecode
   - Added `emit_reduce_body()` for reduction function bytecode
   - Added `extract_function()` for function identification
   - Added identity operations for `*` (mul) and `min`

3. **test_mapreduce_mwe.jl** (new file)
   - MWE test suite for mapreduce functionality
   - Syntax validation tests
   - Compilation tests

## Supported Functions

### Map Functions
- `identity`, `abs`, `abs2`, `sqrt`, `exp`, `log`, `sin`, `cos`, `neg`

### Reduce Functions
- `+` (add), `*` (mul), `max`, `min`

## Usage Examples

```julia
# Sum of squares
result = mapreduce(x -> x*x, +, tile, 2)

# Max of absolute values
result = mapreduce(abs, max, tile, 1)

# Product with init
result = mapreduce(x -> x + 1, *, tile, 2; init=1.0f0)
```

## Implementation Approach

The `mapreduce` compiles to a `ReduceOp` with a custom body:
```
ReduceOp(input, axis=axis, identity=id) {
  body(acc, elem) {
    mapped = f(elem)           # Apply map function
    result = op(acc, mapped)   # Apply reduce function
    yield result
  }
}
```

## Comparison with Existing Functions

| Function | Map | Reduce | Notes |
|----------|-----|--------|-------|
| `reduce_sum` | `identity` | `+` | Hardcoded |
| `reduce_max` | `identity` | `max` | Hardcoded |
| `mapreduce` | Custom `f` | Custom `op` | Generalized |

## Testing

```julia
include("test_mapreduce_mwe.jl")
TestMapReduceMWE.quick_verify()
TestMapReduceMWE.test_all()
```

## Future Work

1. Support anonymous functions via IR analysis
2. Add more map functions (tanh, floor, ceil, etc.)
3. Support custom reduction functions
4. Runtime `init` value support
5. Integer type support for all operations
