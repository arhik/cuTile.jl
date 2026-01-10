#=============================================================================
 Scan (Prefix Sum) Operation Tests - Tile IR Validation
=============================================================================#

using Test
using cuTile
import cuTile as ct

@testset "Scan" begin
    # Common ArraySpecs for tests
    spec1d = ct.ArraySpec{1}(16, true)
    spec2d = ct.ArraySpec{2}(16, true)
    spec3d = ct.ArraySpec{3}(16, true)

    #=========================================================================
     Tile IR Generation Tests
    =========================================================================#
    @testset "cumsum generates scan operation" begin
        # Test that cumsum generates valid Tile IR with scan operation
        ir = ct.code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,2,spec2d}}) do a, b
            pid = ct.bid(1)
            tile = ct.load(a, pid, (4, 8))
            result = ct.cumsum(tile, ct.axis(1))
            ct.store(b, pid, result)
            return
        end
        @test occursin("scan", ir)
        @test occursin("addf", ir)  # cumsum uses addition in body
    end

    @testset "cumprod generates scan operation" begin
        # Test that cumprod generates valid Tile IR with scan operation
        ir = ct.code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,2,spec2d}}) do a, b
            pid = ct.bid(1)
            tile = ct.load(a, pid, (4, 8))
            result = ct.cumprod(tile, ct.axis(1))
            ct.store(b, pid, result)
            return
        end
        @test occursin("scan", ir)
        @test occursin("mulf", ir)  # cumprod uses multiplication in body
    end

    @testset "direct scan with :add" begin
        ir = ct.code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,2,spec2d}}) do a, b
            pid = ct.bid(1)
            tile = ct.load(a, pid, (4, 8))
            result = ct.scan(tile, ct.axis(1), :add)
            ct.store(b, pid, result)
            return
        end
        @test occursin("scan", ir)
        @test occursin("addf", ir)
    end

    @testset "direct scan with :mul" begin
        ir = ct.code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,2,spec2d}}) do a, b
            pid = ct.bid(1)
            tile = ct.load(a, pid, (4, 8))
            result = ct.scan(tile, ct.axis(1), :mul)
            ct.store(b, pid, result)
            return
        end
        @test occursin("scan", ir)
        @test occursin("mulf", ir)
    end

    #=========================================================================
     Different Element Types
    =========================================================================#
    @testset "Float64 scan" begin
        ir = ct.code_tiled(Tuple{ct.TileArray{Float64,2,spec2d}, ct.TileArray{Float64,2,spec2d}}) do a, b
            pid = ct.bid(1)
            tile = ct.load(a, pid, (4, 8))
            result = ct.cumsum(tile, ct.axis(1))
            ct.store(b, pid, result)
            return
        end
        @test occursin("scan", ir)
        @test occursin("f64", ir)  # Float64 type
    end

    @testset "Int32 scan" begin
        ir = ct.code_tiled(Tuple{ct.TileArray{Int32,2,spec2d}, ct.TileArray{Int32,2,spec2d}}) do a, b
            pid = ct.bid(1)
            tile = ct.load(a, pid, (4, 8))
            result = ct.cumsum(tile, ct.axis(1))
            ct.store(b, pid, result)
            return
        end
        @test occursin("scan", ir)
        @test occursin("i32", ir)  # Int32 type
    end

    #=========================================================================
     Different Axes
    =========================================================================#
    @testset "scan along axis 0 (2D)" begin
        ir = ct.code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,2,spec2d}}) do a, b
            pid = ct.bid(1)
            tile = ct.load(a, pid, (4, 8))
            result = ct.cumsum(tile, ct.axis(1))
            ct.store(b, pid, result)
            return
        end
        @test occursin("scan", ir)
        @test occursin("dim=0", ir)  # axis 0
    end

    @testset "scan along axis 1 (2D)" begin
        ir = ct.code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,2,spec2d}}) do a, b
            pid = ct.bid(1)
            tile = ct.load(a, pid, (4, 8))
            result = ct.cumsum(tile, ct.axis(2))
            ct.store(b, pid, result)
            return
        end
        @test occursin("scan", ir)
        @test occursin("dim=1", ir)  # axis 1
    end

    @testset "scan along axis 0 (3D)" begin
        ir = ct.code_tiled(Tuple{ct.TileArray{Float32,3,spec3d}, ct.TileArray{Float32,3,spec3d}}) do a, b
            pid = ct.bid(1)
            tile = ct.load(a, pid, (2, 4, 8))
            result = ct.cumsum(tile, ct.axis(1))
            ct.store(b, pid, result)
            return
        end
        @test occursin("scan", ir)
        @test occursin("dim=0", ir)
    end

    @testset "scan along axis 1 (3D)" begin
        ir = ct.code_tiled(Tuple{ct.TileArray{Float32,3,spec3d}, ct.TileArray{Float32,3,spec3d}}) do a, b
            pid = ct.bid(1)
            tile = ct.load(a, pid, (2, 4, 8))
            result = ct.cumsum(tile, ct.axis(2))
            ct.store(b, pid, result)
            return
        end
        @test occursin("scan", ir)
        @test occursin("dim=1", ir)
    end

    @testset "scan along axis 2 (3D)" begin
        ir = ct.code_tiled(Tuple{ct.TileArray{Float32,3,spec3d}, ct.TileArray{Float32,3,spec3d}}) do a, b
            pid = ct.bid(1)
            tile = ct.load(a, pid, (2, 4, 8))
            result = ct.cumsum(tile, ct.axis(3))
            ct.store(b, pid, result)
            return
        end
        @test occursin("scan", ir)
        @test occursin("dim=2", ir)
    end

    #=========================================================================
     Shape Preservation Tests
    =========================================================================#
    @testset "2D output shape matches input" begin
        ir = ct.code_tiled(Tuple{ct.TileArray{Float32,2,spec2d}, ct.TileArray{Float32,2,spec2d}}) do a, b
            pid = ct.bid(1)
            tile = ct.load(a, pid, (4, 8))
            result = ct.cumsum(tile, ct.axis(1))
            ct.store(b, pid, result)
            return
        end
        @test occursin("tile<4x8xf32>", ir)  # Output shape matches input
    end

    @testset "3D output shape matches input" begin
        ir = ct.code_tiled(Tuple{ct.TileArray{Float32,3,spec3d}, ct.TileArray{Float32,3,spec3d}}) do a, b
            pid = ct.bid(1)
            tile = ct.load(a, pid, (2, 4, 8))
            result = ct.cumsum(tile, ct.axis(1))
            ct.store(b, pid, result)
            return
        end
        @test occursin("tile<2x4x8xf32>", ir)  # Output shape matches input
    end

    #=========================================================================
     Type Stability Tests
    =========================================================================#
    @testset "Type stability" begin
        tile = ct.Tile{Float32, (4, 5)}()

        @testset "scan return type" begin
            return_types = Base.return_types(ct.scan, (typeof(tile), Val{1}, Symbol))
            @test return_types isa Vector{Any}
            @test !isempty(return_types)
            @test return_types[1] == ct.Tile{Float32, (4, 5)}
        end

        @testset "cumsum return type" begin
            return_types = Base.return_types(ct.cumsum, (typeof(tile), Val{1}))
            @test return_types isa Vector{Any}
            @test !isempty(return_types)
            @test return_types[1] == ct.Tile{Float32, (4, 5)}
        end

        @testset "cumprod return type" begin
            return_types = Base.return_types(ct.cumprod, (typeof(tile), Val{1}))
            @test return_types isa Vector{Any}
            @test !isempty(return_types)
            @test return_types[1] == ct.Tile{Float32, (4, 5)}
        end
    end
end
