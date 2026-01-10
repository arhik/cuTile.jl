
# Scan Example for cuTile.jl
# Demonstrates parallel prefix sum using Tile IR scan operation
#
# Run with: julia --project=. examples/scan.jl

using CUDA
using cuTile
import cuTile as ct

# 1D cumulative sum kernel (1-based indexing, returns nothing)
function cumsum_1d_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                          tile_size::ct.Constant{Int})
    bid = ct.bid(1)
    tile = ct.load(a, bid, (tile_size[],))
    result = ct.cumsum(tile, ct.axis(1))
    ct.store(b, bid, result)
    return
end

# 2D cumulative sum kernel - scan along axis 2 (columns)
function cumsum_2d_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2},
                          tile_x::ct.Constant{Int}, tile_y::ct.Constant{Int})
    bid_x = ct.bid(1)
    bid_y = ct.bid(2)
    tile = ct.load(a, (bid_x, bid_y), (tile_y[], tile_x[]))
    result = ct.cumsum(tile, ct.axis(2))
    ct.store(b, (bid_x, bid_y), result)
    return
end

# 1D cumulative product kernel
function cumprod_1d_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                           tile_size::ct.Constant{Int})
    bid = ct.bid(1)
    tile = ct.load(a, bid, (tile_size[],))
    result = ct.cumprod(tile, ct.axis(1))
    ct.store(b, bid, result)
    return
end

# Show generated Tile IR
function show_scan_ir()
    println("\n=== Generated Tile IR ===")
    input = ct.TileArray(CUDA.zeros(Float32, 1024))
    output = ct.TileArray(CUDA.zeros(Float32, 1024))
    ir = ct.code_tiled(Tuple{typeof(input), typeof(output)}) do a, b
        ct.store(b, ct.bid(1), ct.cumsum(ct.load(a, ct.bid(1), (1024,)), ct.axis(1)))
        return
    end
    println(ir)
    println("=== End IR ===")
    return ir
end

# Main test
function main()
    println("cuTile Scan Example")
    println("===================")
    println()
    println("GPU: $(CUDA.name(CUDA.device()))")
    println("Compute: $(CUDA.capability(CUDA.device()))")
    println()

    # IR test
    println("--- IR Generation ---")
    ir = show_scan_ir()
    println(occursin("scan", ir) ? "✓ scan in IR" : "✗ scan NOT in IR")

    # GPU tests
    println()
    println("--- GPU Tests ---")

    # Test 1: 1D cumsum
    println("\nTest 1: 1D cumsum (1024 elements)")
    n, sz = 1024, 256
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)
    CUDA.@sync ct.launch(cumsum_1d_kernel, cld(n, sz), ct.TileArray(a), ct.TileArray(b), ct.Constant(sz))
    res = Array(b)
    exp = cumsum(Array(a), dims=1)
    println(res ≈ exp ? "  PASS" : "  FAIL")

    # Test 2: 1D cumsum larger
    println("\nTest 2: 1D cumsum (32768 elements)")
    n, sz = 32768, 1024
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)
    CUDA.@sync ct.launch(cumsum_1d_kernel, cld(n, sz), ct.TileArray(a), ct.TileArray(b), ct.Constant(sz))
    res = Array(b)
    exp = cumsum(Array(a), dims=1)
    println(res ≈ exp ? "  PASS" : "  FAIL")

    # Test 3: 2D cumsum along axis 2
    println("\nTest 3: 2D cumsum (256 x 512), axis 2")
    m, n, tx, ty = 256, 512, 32, 32
    a = CUDA.rand(Float32, m, n)
    b = CUDA.zeros(Float32, m, n)
    CUDA.@sync ct.launch(cumsum_2d_kernel, (cld(n, tx), cld(m, ty)),
                         ct.TileArray(a), ct.TileArray(b), ct.Constant(tx), ct.Constant(ty))
    res = Array(b)
    exp = cumsum(Array(a), dims=2)
    println(res ≈ exp ? "  PASS" : "  FAIL")

    # Test 4: 1D cumprod
    println("\nTest 4: 1D cumprod (10000 elements)")
    n, sz = 10000, 256
    a = CUDA.rand(Float32, n) .+ 0.1f0
    b = CUDA.zeros(Float32, n)
    CUDA.@sync ct.launch(cumprod_1d_kernel, cld(n, sz), ct.TileArray(a), ct.TileArray(b), ct.Constant(sz))
    res = Array(b)
    exp = cumprod(Array(a), dims=1)
    println(res ≈ exp ? "  PASS" : "  FAIL")

    println("\nDone!")
end

main()
