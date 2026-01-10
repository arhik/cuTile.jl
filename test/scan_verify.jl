# Minimal Scan Verification - CSDL (32 x 2^15) Tile
# Focus: Verify scan code generation compiles correctly

using Test
using cuTile
import cuTile as ct

# CSDL tile configuration: (32, 2^15)
# 32 = warp width for efficient intra-warp scan
# 2^15 = 32768 = partition length for chained lookback
const TILE_W = 32
const TILE_H = 2^15

@testset "Scan Code Generation" begin
    # Use minimal dimensions for faster compilation
    small_spec = ct.ArraySpec{2}(8, true)

    @testset "Basic cumsum" begin
        ir = ct.code_tiled(
            Tuple{ct.TileArray{Float32,2,small_spec}, ct.TileArray{Float32,2,small_spec}}
        ) do a, b
            pid = ct.bid(1)
            tile = ct.load(a, pid, (8, 64))
            result = ct.cumsum(tile, ct.axis(2))
            ct.store(b, pid, result)
            return
        end
        @test occursin("scan", ir)
        @test occursin("addf", ir)
        @test occursin("dim=1", ir)
        @test occursin("tile<8x64xf32>", ir)
    end

    @testset "CSDL axis 1 scan" begin
        ir = ct.code_tiled(
            Tuple{ct.TileArray{Float32,2,small_spec}, ct.TileArray{Float32,2,small_spec}}
        ) do a, b
            pid = ct.bid(1)
            tile = ct.load(a, pid, (8, 64))
            result = ct.cumsum(tile, ct.axis(2))
            ct.store(b, pid, result)
            return
        end
        @test occursin("dim=1", ir)
    end

    @testset "CSDL reverse scan" begin
        ir = ct.code_tiled(
            Tuple{ct.TileArray{Float32,2,small_spec}, ct.TileArray{Float32,2,small_spec}}
        ) do a, b
            pid = ct.bid(1)
            tile = ct.load(a, pid, (8, 64))
            result = ct.cumsum(tile, ct.axis(2), true)
            ct.store(b, pid, result)
            return
        end
        @test occursin("reverse=true", ir)
    end

    @testset "CSDL cumprod" begin
        ir = ct.code_tiled(
            Tuple{ct.TileArray{Float32,2,small_spec}, ct.TileArray{Float32,2,small_spec}}
        ) do a, b
            pid = ct.bid(1)
            tile = ct.load(a, pid, (8, 64))
            result = ct.cumprod(tile, ct.axis(2))
            ct.store(b, pid, result)
            return
        end
        @test occursin("mulf", ir)
    end
end
