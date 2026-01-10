
# CSDL (Chained Scan with Decoupled Lookback) Scan Example
# SPDX-License-Identifier: Apache-2.0

using CUDA
using cuTile
import cuTile as ct

# 1D cumsum kernel
function cumsum_1d_kernel(a::ct.TileArray{T,1}, b::ct.TileArray{T,1}, tile_size::ct.Constant{Int}) where T
    bid = ct.bid(1)
    tile = ct.load(a, bid, (tile_size[],))
    result = ct.cumsum(tile, Val(0))
    ct.store(b, bid, result)
    return
end

# 2D cumsum kernel (scan along rows)
function cumsum_2d_kernel(a::ct.TileArray{T,2}, b::ct.TileArray{T,2}, tile_x::ct.Constant{Int}, tile_y::ct.Constant{Int}) where T
    bid_x, bid_y = ct.bid(1), ct.bid(2)
    tile = ct.load(a, (bid_x, bid_y), (tile_y[], tile_x[]))
    result = ct.cumsum(tile, Val(1))
    ct.store(b, (bid_x, bid_y), result)
    return
end

# Main test function
function main()
    println("cuTile CSDL Scan Test")

    device = CUDA.device()
    println("GPU: $(CUDA.name(device))")
    println("Compute Capability: $(CUDA.capability(device))")
    println()

    # 1D test
    n, tile_size = 32768, 1024
    a = CUDA.rand(Float32, n)
    b = CUDA.zeros(Float32, n)
    a_tile, b_tile = ct.TileArray(a), ct.TileArray(b)

    grid = cld(n, tile_size)
    CUDA.@sync ct.launch(cumsum_1d_kernel, grid, a_tile, b_tile, ct.Constant(tile_size))

    expected = cumsum(Array(a), dims=1)
    if Array(b) ≈ expected
        println("1D cumsum: PASS")
    else
        println("1D cumsum: FAIL")
    end

    # 2D test
    m, n = 512, 1024
    tile_x, tile_y = 32, 32
    a2 = CUDA.rand(Float32, m, n)
    b2 = CUDA.zeros(Float32, m, n)
    a2_tile, b2_tile = ct.TileArray(a2), ct.TileArray(b2)

    grid2 = (cld(n, tile_x), cld(m, tile_y))
    CUDA.@sync ct.launch(cumsum_2d_kernel, grid2, a2_tile, b2_tile, ct.Constant(tile_x), ct.Constant(tile_y))

    expected2 = cumsum(Array(a2), dims=2)
    if Array(b2) ≈ expected2
        println("2D cumsum: PASS")
    else
        println("2D cumsum: FAIL")
    end

    println("Done!")
end

main()
