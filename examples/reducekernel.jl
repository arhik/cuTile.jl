# EXCLUDE FROM TESTING
# This file is kept for reference only. The comprehensive test suite is in test/reducekernel.jl
# Run tests via: julia --project=test test/reducekernel.jl

using Test
using CUDA
using cuTile
import cuTile as ct

# Example: 1D tile reduction kernel
# Reduces a 1D array along 32-element tiles, writing results to output
elType = Int64
function reducekernel(a::ct.TileArray{elType,1},
                      b::ct.TileArray{elType,1},
                      tileSz::ct.Constant{Int})
    pid = ct.bid(1)                    # 1-indexed block ID
    tile = ct.load(a, pid, (tileSz[],))  # Load tile from input
    reduced = ct.reduce_sum(tile, 1)   # Sum along axis 1 (the only axis)
    ct.store(b, pid, reduced)          # Store result
    return nothing
end

# Launch parameters
tile_size = 32
N = 2^15                              # 32768 elements
grid_size = cld(N, tile_size)         # Ceiling division: 1024 blocks

# Run the kernel
a = CUDA.rand(elType, N)
b = CUDA.zeros(elType, grid_size)
CUDA.@sync ct.launch(reducekernel, grid_size, a, b, ct.Constant(tile_size))

# Verify results
a_cpu = Array(a)
b_cpu = Array(b)
expected = sum(reshape(a_cpu, tile_size, :), dims=1)[:]
@test b_cpu ≈ expected

println("Reduce kernel test passed!")
