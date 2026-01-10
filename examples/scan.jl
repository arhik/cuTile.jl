# Scan Example for cuTile.jl
# Demonstrates parallel prefix sum using Tile IR scan operation
#
# Run with: julia --project=. examples/scan.jl

using CUDA
using cuTile
import cuTile as ct

# 1D cumulative sum kernel
# Uses 1-based indexing (ct.bid(1)), returns nothing
function cumsum_1d_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                          tile_size::ct.Constant{Int})
    bid = ct.bid(1)
    tile = ct.load(a, bid, (tile_size[],))
    result = ct.cumsum(tile, Val(0))
    ct.store(b, bid, result)
    return
end

# 2D cumulative sum kernel - scan along axis 1 (rows)
function cumsum_2d_kernel(a::ct.TileArray{Float32,2}, b::ct.TileArray{Float32,2},
                          tile_x::ct.Constant{Int}, tile_y::ct.Constant{Int})
    bid_x = ct.bid(1)
    bid_y = ct.bid(2)
    tile = ct.load(a, (bid_x, bid_y), (tile_y[], tile_x[]))
    result = ct.cumsum(tile, Val(1))
    ct.store(b, (bid_x, bid_y), result)
    return
end

# 1D cumulative product kernel
function cumprod_1d_kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1},
                           tile_size::ct.Constant{Int})
    bid = ct.bid(1)
    tile = ct.load(a, bid, (tile_size[],))
    result = ct.cumprod(tile, Val(0))
    ct.store(b, bid, result)
    return
end

# Show generated Tile IR for scan operation
function show_scan_ir()
    println("\n=== Generated Tile IR for cumsum ===")

    input = ct.TileArray(CUDA.zeros(Float32, 1024))
    output = ct.TileArray(CUDA.zeros(Float32, 1024))

    ir = ct.code_tiled(Tuple{typeof(input), typeof(output)}) do a, b
        ct.store(b, ct.bid(1), ct.cumsum(ct.load(a, ct.bid(1), (1024,)), Val(0)))
        return
    end

    println(ir)
    println("=== End IR ===")

    return ir
end

# Main test function
function main()
    println("cuTile Scan Example")
    println("===================")
    println()
    println("GPU: $(CUDA.name(CUDA.device()))")
    println("Compute: $(CUDA.capability(CUDA.device()))")
    println()

    # Test 1: Generate and inspect IR (works without GPU)
    println("--- IR Generation Test ---")
    ir = show_scan_ir()

    if occursin("scan", ir) && occursin("addf", ir)
        println("✓ scan and addf operations found in IR")
    else
        println("✗ IR generation issue")
    end

    # Test 2: GPU execution
    println()
    println("--- GPU Execution Tests ---")

    # 1D cumsum test
    println()
    println("Test 1: 1D cumsum (1024 elements)")
    n, tile_size = 1024, 256
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)
    grid = cld(n, tile_size)

    CUDA.@sync ct.launch(cumsum_1d_kernel, grid, ct.TileArray(a), ct.TileArray(b), ct.Constant(tile_size))

    result = Array(b)
    expected = cumsum(Array(a), dims=1)
    if result ≈ expected
        println("  Result: PASS")
    else
        println("  Result: FAIL")
        @printf "  Max error: %.6f\n" maximum(abs.(result .- expected))
    end

    # 1D cumsum larger test
    println()
    println("Test 2: 1D cumsum (32768 elements)")
    n, tile_size = 32768, 1024
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)
    grid = cld(n, tile_size)

    CUDA.@sync ct.launch(cumsum_1d_kernel, grid, ct.TileArray(a), ct.TileArray(b), ct.Constant(tile_size))

    result = Array(b)
    expected = cumsum(Array(a), dims=1)
    if result ≈ expected
        println("  Result: PASS")
    else
        println("  Result: FAIL")
        @printf "  Max error: %.6f\n" maximum(abs.(result .- expected))
    end

    # 2D cumsum test
    println()
    println("Test 3: 2D cumsum (256 x 512, scan along axis 1)")
    m, n, tile_x, tile_y = 256, 512, 32, 32
    a = CUDA.rand(Float32, m, n)
    b = CUDA.zeros(Float32, m, n)
    grid = (cld(n, tile_x), cld(m, tile_y))

    CUDA.@sync ct.launch(cumsum_2d_kernel, grid, ct.TileArray(a), ct.TileArray(b),
                         ct.Constant(tile_x), ct.Constant(tile_y))

    result = Array(b)
    expected = cumsum(Array(a), dims=2)
    if result ≈ expected
        println("  Result: PASS")
    else
        println("  Result: FAIL")
        @printf "  Max error: %.6f\n" maximum(abs.(result .- expected))
    end

    # 1D cumprod test
    println()
    println("Test 4: 1D cumprod (10000 elements)")
    n, tile_size = 10000, 256
    a = CUDA.rand(Float32, n) .+ 0.1f0
    b = CUDA.zeros(Float32, n)
    grid = cld(n, tile_size)

    CUDA.@sync ct.launch(cumprod_1d_kernel, grid, ct.TileArray(a), ct.TileArray(b), ct.Constant(tile_size))

    result = Array(b)
    expected = cumprod(Array(a), dims=1)
    if result ≈ expected
        println("  Result: PASS")
    else
        println("  Result: FAIL")
        @printf "  Max error: %.6f\n" maximum(abs.(result .- expected))
    end

    println()
    println("All tests completed!")
    return
end

main()
