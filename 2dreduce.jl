using cuTile
import cuTile as ct
using CUDA

# Axis=2 (reduce rows) kernel with tile size parameter
function reducekernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,1}, tileSize::ct.Constant{Int})
    pid = ct.bid(2)
    tile = ct.load(a, (1, pid), (tileSize[], 1))
    result = ct.reduce_sum(tile, Val(1))
    ct.store(b, pid, result)
    return nothing
end

# Axis=1 (reduce columns) kernel with tile size parameter
function reducekernel_axis1(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,1}, tileSize::ct.Constant{Int})
    pid = ct.bid(1)
    tile = ct.load(a, (pid, 1), (1, tileSize[]))
    result = ct.reduce_sum(tile, Val(2))
    ct.store(b, pid, result)
    return nothing
end

# Test axis=2 (reduce rows) - use full dimension as tile size
M, N = 64, 128
TILE_SIZE_ROWS = M  # 64 - covers all rows in each column

a = CUDA.rand(Float32, M, N)
b = CUDA.zeros(Float32, N)

ct.launch(reducekernel, (1, N), a, b, ct.Constant(TILE_SIZE_ROWS))
CUDA.synchronize()

a_cpu = Array(a)
b_cpu = Array(b)
cpu = sum(a_cpu, dims=1)[:]

d = maximum(abs.(b_cpu - cpu))
println("axis=2 (reduce rows, tile=$TILE_SIZE_ROWS): max_diff = $d")
if d < 1e-3
    println("✓ PASSED!")
else
    println("✗ FAILED")
end

# Test axis=1 (reduce columns) - use full dimension as tile size
TILE_SIZE_COLS = N  # 128 - covers all columns in each row
b2 = CUDA.zeros(Float32, M)
ct.launch(reducekernel_axis1, (M, 1), a, b2, ct.Constant(TILE_SIZE_COLS))
CUDA.synchronize()

b2_cpu = Array(b2)
cpu2 = sum(a_cpu, dims=2)[:]

d2 = maximum(abs.(b2_cpu - cpu2))
println("axis=1 (reduce columns, tile=$TILE_SIZE_COLS): max_diff = $d2")
if d2 < 1e-3
    println("✓ PASSED!")
else
    println("✗ FAILED")
end