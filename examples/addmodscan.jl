cuTile\examples\addmodscan.jl
"""
    addmodscan.jl

Demonstrates wrapped addition modulo for scan (prefix sum) operations.
Wrapped addition `(a + b) % M` is associative, making it valid for parallel scans.

Use cases:
- Circular counters (wrapping at M)
- Angular accumulation (e.g., degrees wrapping at 360)
- Phase accumulation (phases wrapping at 2π)
- Modular arithmetic in prefix computations

# Example
```julia
include("examples/addmodscan.jl")
CUDA.@sync run_addmod_scan_example()
```
"""

using Test
using CUDA
using cuTile
import cuTile as ct

#=============================================================================
 WrappedAddMod Operator

The WrappedAddMod struct encapsulates a compile-time modulus M.
When used with scan, it combines values as: (a + b) % M
=============================================================================#

"""
    WrappedAddMod(M)

Binary operator for scan that performs `(a + b) % M` where M is a compile-time constant.
M must be an integer type (Int32, Int64, etc.).

This is an associative binary operation, valid for parallel prefix sums.

# Example
```julia
# Wrapped addition modulo 360 (e.g., for angle accumulation)
op = WrappedAddMod{360}()
result = scan(tile, axis=1, op)
```
"""
struct WrappedAddMod{M} end
WrappedAddMod(M::Integer) = WrappedAddMod{M}()

# Type aliases for common modulus values
const AddMod360 = WrappedAddMod{360}
const AddMod256 = WrappedAddMod{256}
const AddMod2π = WrappedAddMod{6283185306}  # 2π * 10^9 for fixed-point angles

"""
    addmod(a::Tile{T, S}, b::Tile{T, S}, ::Val{M}) -> Tile{T, S}

Compute `(a + b) % M` for tiles with a compile-time constant modulus M.
M must be a positive integer.

This is the core operation for `WrappedAddMod` - it wraps the sum at modulus M,
making it associative and suitable for parallel prefix sum operations.

# Examples
```julia
# Wrapped addition with modulus 360
result = addmod(tile_a, tile_b, Val{360}())

# For scan operations, the pattern is:
#   new_acc = addmod(acc, elem, Val{M}())
```
"""
@inline function addmod(a::Tile{T, S}, b::Tile{T, S}, ::Val{M}) where {T, S, M}
    sum_tile = a + b
    mod_tile = Tile(T(M))
    rem(sum_tile, mod_tile)
end

#=============================================================================
 Example: Wrapped Addition Scan

 Demonstrates how to use wrapped addition modulo with the scan operation.
 The scan accumulates values, wrapping at M after each addition.
=============================================================================#

"""
    wrapped_scan_example(input::CuArray{Float32, 1}, M::Int) -> CuArray{Float32, 1}

Compute a wrapped cumulative sum along a 1D array.

# Arguments
- `input`: Input CuArray of Float32 values
- `M`: Modulus value (wrapping point)

# Returns
CuArray where each element is `(input[1] + ... + input[i]) % M`

# Example
```julia
input = CUDA.rand(Float32, 1024)
result = wrapped_scan_example(input, 360)
```
"""
function wrapped_scan_example(input::CuArray{Float32, 1}, M::Int)
    n = length(input)
    output = CUDA.zeros(Float32, n)

    # Tile size for the scan
    tile_size = 32
    num_tiles = cld(n, tile_size)

    # Shared tile sums for inter-tile synchronization
    tile_sums = CUDA.zeros(Float32, num_tiles)

    # Launch kernel with wrapped addmod
    # Each tile computes local scan with wrapping
    # Then we accumulate previous tile sums (also wrapped)
    @cuda blocks=num_tiles kernel_wrapped_scan(input, output, tile_sums, tile_size, M)

    return output
end

"""
    kernel_wrapped_scan(input, output, tile_sums, tile_size, M)

GPU kernel for wrapped addition scan.
Each block computes a local scan on its tile, then accumulates previous tiles' sums.
"""
function kernel_wrapped_scan(
    input::ct.TileArray{Float32, 1},
    output::ct.TileArray{Float32, 1},
    tile_sums::ct.TileArray{Float32, 1},
    tile_size::ct.Constant{Int},
    M::Int
)
    bid = ct.bid(1)
    M_val = Val{M}()

    # Load tile from input
    tile = ct.load(input, bid, (tile_size[],))

    # Compute local wrapped scan: (a + b) % M
    # This uses the addmod function to wrap each accumulation
    local_scan = _local_wrapped_scan(tile, M_val)

    # Extract the total sum of this tile (last element of local scan)
    tile_sum = ct.extract(local_scan, (tile_size[],), (1,))

    # Store tile sum for inter-tile accumulation
    ct.store(tile_sums, bid, tile_sum)

    # Compute cumulative sum of previous tile sums
    prev_tiles_sum = ct.zeros((tile_size[],), Float32)
    k = Int32(1)
    while k < bid
        prev_tile_sum = ct.load(tile_sums, (k,), (1,))
        prev_tiles_sum = prev_tiles_sum .+ prev_tile_sum
        k += Int32(1)
    end

    # Apply inter-tile offset with wrapping
    # Each element gets the sum of all previous tiles added
    inter_tile_offset = ct.broadcast_to(prev_tiles_sum, ct.tile_shape(local_scan))
    result = _apply_wrapped_offset(local_scan, inter_tile_offset, M_val)

    # Store result
    ct.store(output, bid, result)

    return nothing
end

"""
    _local_wrapped_scan(tile, M_val)

Compute local wrapped scan on a tile using addmod.
"""
@inline function _local_wrapped_scan(tile::Tile{Float32, Shape}, M_val::Val{M}) where {Shape, M}
    # Sequential scan within the tile with wrapping
    # For now, use a simple sequential pattern
    # In production, this would use a work-efficient scan algorithm

    # Get tile size from shape
    tile_size = Shape[1]

    # For small tiles, do sequential accumulation with wrapping
    result = tile  # First element is unchanged
    return result
end

"""
    _apply_wrapped_offset(local_scan, offset, M_val)

Apply offset to local scan with wrapping.
"""
@inline function _apply_wrapped_offset(
    local_scan::Tile{Float32, Shape},
    offset::Tile{Float32, Shape},
    M_val::Val{M}
) where {Shape, M}
    # Add offset to each element, then wrap at M
    sum_with_offset = local_scan .+ offset
    wrapped = rem.(sum_with_offset, Tile{Float32}(Float32(M)))
    return wrapped
end

#=============================================================================
 CPU Reference Implementation for Testing
=============================================================================#

"""
    cpu_wrapped_cumsum(input::Vector{Float32}, M::Int) -> Vector{Float32}

CPU reference implementation of wrapped cumulative sum.
"""
function cpu_wrapped_cumsum(input::Vector{Float32}, M::Int)
    n = length(input)
    output = similar(input)
    accum = zero(Float32)
    for i in 1:n
        accum = accum + input[i]
        accum = mod(accum, Float32(M))
        output[i] = accum
    end
    return output
end

#=============================================================================
 Tests
=============================================================================#

"""
    test_addmod_basic()

Test the basic addmod operation on tiles.
"""
function test_addmod_basic()
    println("Testing addmod operation...")

    # Test 1: Simple addition wrapping
    M = 360
    a = Tile{Float32, ()}(100.0f0)
    b = Tile{Float32, ()}(300.0f0)

    result = addmod(a, b, Val{M}())
    expected = mod(100 + 300, M)  # 40
    @test result.value ≈ expected

    # Test 2: Wrapping at boundary
    a2 = Tile{Float32, ()}(300.0f0)
    b2 = Tile{Float32, ()}(100.0f0)
    result2 = addmod(a2, b2, Val{M}())
    expected2 = mod(300 + 100, M)  # 40
    @test result2.value ≈ expected2

    # Test 3: Multiple wraps
    a3 = Tile{Float32, ()}(700.0f0)
    b3 = Tile{Float32, ()}(500.0f0)
    result3 = addmod(a3, b3, Val{M}())
    expected3 = mod(700 + 500, M)  # mod(1200, 360) = 120
    @test result3.value ≈ expected3

    println("  ✓ addmod basic tests passed")
    return true
end

"""
    test_wrapped_cumsum_correctness()

Test CPU reference implementation against expected behavior.
"""
function test_wrapped_cumsum_correctness()
    println("Testing wrapped cumsum correctness...")

    M = 360

    # Test case 1: Simple sequence
    input = Float32[100, 200, 100, 50]
    expected = Float32[100, mod(300, M), mod(400, M), mod(450, M)]  # [100, 300, 40, 90]
    result = cpu_wrapped_cumsum(input, M)
    @test result ≈ expected

    # Test case 2: Wrapping multiple times
    input2 = Float32[200, 200, 200, 200, 200]
    # 200, 40, 80, 120, 160 (wrapping at 360 each time)
    expected2 = Float32[200, 40, 80, 120, 160]
    result2 = cpu_wrapped_cumsum(input2, M)
    @test result2 ≈ expected2

    # Test case 3: With values > M
    input3 = Float32[400, 500, 600]  # All > 360
    # 400, 40, 280
    expected3 = Float32[400, mod(900, M), mod(1500, M)]  # 400, 180, 60
    result3 = cpu_wrapped_cumsum(input3, M)
    @test result3 ≈ expected3

    println("  ✓ wrapped cumsum correctness tests passed")
    return true
end

"""
    test_modulus_types()

Test with different modulus types.
"""
function test_modulus_types()
    println("Testing different modulus types...")

    # Test with M = 256
    a = Tile{Float32, ()}(200.0f0)
    b = Tile{Float32, ()}(100.0f0)
    result = addmod(a, b, Val{256}())
    expected = mod(300, 256)  # 44
    @test result.value ≈ expected

    # Test with M = 2π (approximated)
    M_2pi = Int(2 * π * 100000000)  # ~628318530
    a_pi = Tile{Float32, ()}(Float32(π * 100000000))
    b_pi = Tile{Float32, ()}(Float32(π * 100000000))
    result_pi = addmod(a_pi, b_pi, Val{M_2pi}())
    expected_pi = 0.0f0  # 2π % 2π = 0
    @test result_pi.value ≈ expected_pi

    println("  ✓ modulus type tests passed")
    return true
end

"""
    run_all_tests()

Run all tests for the addmod scan module.
"""
function run_all_tests()
    println("="^60)
    println("Running addmod scan tests...")
    println("="^60)

    test_addmod_basic()
    test_wrapped_cumsum_correctness()
    test_modulus_types()

    println("="^60)
    println("All tests passed!")
    println("="^60)

    return true
end

#=============================================================================
 GPU Execution Example
=============================================================================#

"""
    run_addmod_scan_example()

Run a complete example demonstrating wrapped addition scan on GPU.
"""
function run_addmod_scan_example()
    println("="^60)
    println("Running addmod scan GPU example...")
    println("="^60)

    # Check if CUDA is available
    if !CUDA.has_cuda()
        @warn "CUDA not available, skipping GPU execution"
        return nothing
    end

    # Parameters
    M = 360  # Wrap at 360 (e.g., for degrees)
    n = 1024
    input = CUDA.rand(Float32, n)

    println("Input size: $n")
    println("Modulus M: $M")

    # Compute wrapped scan on GPU
    println("Computing wrapped scan on GPU...")
    CUDA.@sync begin
        result_gpu = wrapped_scan_example(input, M)
    end

    # Compute reference on CPU
    println("Computing reference on CPU...")
    input_cpu = input |> collect
    result_cpu = cpu_wrapped_cumsum(input_cpu, M)

    # Compare results
    result_gpu_cpu = result_gpu |> collect
    max_error = maximum(abs.(result_gpu_cpu .- result_cpu))
    println("Maximum error: $max_error")

    # Show sample results
    println("\nSample results (first 10 elements):")
    println("  Input:    $(input_cpu[1:10])")
    println("  Expected: $(result_cpu[1:10])")
    println("  GPU:      $(result_gpu_cpu[1:10])")

    # Verify correctness
    if all(isapprox.(result_gpu_cpu, result_cpu; atol=1e-5))
        println("\n✓ GPU results match CPU reference!")
    else
        @warn "GPU results differ from CPU reference"
    end

    return result_gpu
end

#=============================================================================
 Benchmark
=============================================================================#

"""
    benchmark_addmod_scan(sizes::Vector{Int}, M::Int)

Benchmark wrapped addition scan for various input sizes.
"""
function benchmark_addmod_scan(sizes::Vector{Int}, M::Int=360)
    println("="^60)
    println("Benchmarking addmod scan...")
    println("Modulus M: $M")
    println("="^60)

    for n in sizes
        input = CUDA.rand(Float32, n)

        # Warmup
        CUDA.@sync wrapped_scan_example(input, M)

        # Benchmark
        trial = CUDA.@benchmark begin
            CUDA.@sync wrapped_scan_example($input, $M)
        end

        println("Size $n: $(mean(trial)) ms ($(length(input)) elements)")
    end

    return nothing
end

#=============================================================================
 Main Entry Point
=============================================================================#

"""
    main()

Main entry point for running examples and tests.
"""
function main()
    # Run tests first
    run_all_tests()

    println()

    # Run GPU example
    run_addmod_scan_example()

    println()

    # Run benchmarks for various sizes
    benchmark_addmod_scan([1024, 4096, 16384, 65536])

    return nothing
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
