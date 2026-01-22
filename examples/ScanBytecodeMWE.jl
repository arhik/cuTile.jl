# Minimal Working Example for Manual Bytecode Generation in Scan Operation
#
# Goal: Define a custom `scan_add!` function that computes a parallel prefix sum,
# then manually generate and inspect its bytecode using LLVM tools.

"""
    scan_add!(array)

In-place sequential prefix sum (scan) of `array`.
​
This is a naive scalar implementation for demonstration only.
In actual GPU kernels, this would use parallel scans with shared memory
to avoid carrying dependencies across warps/CTAs.
"""
function scan_add!(array)
    # Naive sequential prefix sum: array[i] += array[1..i-1]
    for i in 2:length(array)
        array[i] += array[i-1]
    end
    return array
end

# We'll extend this with LLVM bytecode generation in later stages.
# For this MWE, we show CPU validation first.

# ----------------------------------------------------------------------------
# STAGE 1: Validation & REPL checks
# ----------------------------------------------------------------------------

# Basic correctness test (linear scan)
test_input = [1, 2, 3, 4, 5]
expected   = [1, 3, 6, 10, 15]

@test scan_add!(copy(test_input)) == expected

# Type stability check — ensure `scan_add!` infers correctly
@code_warntype scan_add!(test_input)

# This will build the foundation for manual bytecode manipulation.
```

### Increment 1: File Created & Basic Validation

- **File created:** `cuTile\examples\ScanBytecodeMWE.jl`
- **What changed:** Defined a minimal scan function (`scan_add!`) and basic validation suite.
- **Next steps:** Load into REPL and verify inference, then extend with LLVM bytecode generation tools.

To load and validate in REPL:
```julia
include("cuTile/examples/ScanBytecodeMWE.jl")

# Chairmarks (timing is not the focus yet — correctness first)
@time scan_add!(copy(test_input))

# Stir JIT costs, then measure steady-state
@btime scan_add!($test_input)

# Ensure no inference red flags
@code_typed scan_add!(test_input)
```

Once validated, we can extend this with `LLVM.IRBuilder`, `LLVM.compile`, and `CUDA.@grep_device_code_*` tools to manually shape bytecode for GPU scan patterns like inclination, reduction, or carry propagation.Would you like to proceed with LLVM bytecode generation in the next increment?
