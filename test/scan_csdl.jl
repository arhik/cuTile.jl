#=============================================================================
# CSDL (Chained Scan with Decoupled Lookback) Scan Tests
#
# Target: (32, 2^15) tile dimension for single-phase scan
# Focus: Proper Tile IR code generation without GPU execution
#=============================================================================

using Test
using cuTile
import cuTile as ct

# CSDL-specific tile dimensions
const TILE_W = 32   # Warp size for efficient intra-warp scan
const TILE_H = 2^15 # Height for chained lookback across partitions

@testset "CSDL Scan (32 × 32768)" begin
    # ArraySpec for CSDL target dimensions
    # Tile shape: (32, 32768) for efficient single-phase scan
    # The 32-element warp enables efficient intra-warp scan with shuffle
    # The 32768-element column enables chained lookback with O(1) amortized lookback
    csdl_spec = ct.ArraySpec{2}(32, true)  # Using minimal height for testing

    # Shorter variants for faster testing
    csdl_small_spec = ct.ArraySpec{2}(8, true)

    #=========================================================================
    # Tile IR Generation for CSDL Scan
    #========================================================================#
    @testset "CSDL cumsum generates scan operation" begin
        # Basic CSDL scan along the partition dimension (axis 1)
        # The 32-element warp scan uses shuffle, 32768+ uses chained lookback
        ir = ct.code_tiled(Tuple{ct.TileArray{Float32,2,csdl_spec}, ct.TileArray{Float32,2,csdl_spec}}) do a, b
            pid = ct.bid(1)
            # Load CSDL tile: warp (32) × partition (32768)
            tile = ct.load(a, pid, (32, 32768))
            # Scan along partition dimension (axis 1)
            result = ct.cumsum(tile, Val(1))
            ct.store(b, pid, result)
            return
        end
        @test occursin("scan", ir)
        @test occursin("addf", ir)
        @test occursin("dim=1", ir)
    end

    @testset "CSDL cumprod generates scan operation" begin
        ir = ct.code_tiled(Tuple{ct.TileArray{Float32,2,csdl_spec}, ct.TileArray{Float32,2,csdl_spec}}) do a, b
            pid = ct.bid(1)
            tile = ct.load(a, pid, (32, 32768))
            result = ct.cumprod(tile, Val(1))
            ct.store(b, pid, result)
            return
        end
        @test occursin("scan", ir)
        @test occursin("mulf", ir)
    end

    #=========================================================================
    # Different Scan Axes for CSDL
    #========================================================================#
    @testset "CSDL scan along warp axis (axis 0)" begin
        # Scan along warp dimension (32 elements)
        # This is the intra-warp scan phase of CSDL
        ir = ct.code_tiled(Tuple{ct.TileArray{Float32,2,csdl_spec}, ct.TileArray{Float32,2,csdl_spec}}) do a, b
            pid = ct.bid(1)
            tile = ct.load(a, pid, (32, 32768))
            result = ct.cumsum(tile, Val(0))
            ct.store(b, pid, result)
            return
        end
        @test occursin("scan", ir)
        @test occursin("dim=0", ir)
    end

    @testset "CSDL scan along partition axis (axis 1)" begin
        # Scan along partition dimension (32768+ elements)
        # This triggers the chained lookback phase of CSDL
        ir = ct.code_tiled(Tuple{ct.TileArray{Float32,2,csdl_spec}, ct.TileArray{Float32,2,csdl_spec}}) do a, b
            pid = ct.bid(1)
            tile = ct.load(a, pid, (32, 32768))
            result = ct.cumsum(tile, Val(1))
            ct.store(b, pid, result)
            return
        end
        @test occursin("scan", ir)
        @test occursin("dim=1", ir)
    end

    #=========================================================================
    # Reverse Scan for CSDL (trailing segment handling)
    #========================================================================#
    @testset "CSDL reverse scan" begin
        # Reverse scan for trailing segment handling
        # Useful when segments are defined from the end
        ir = ct.code_tiled(Tuple{ct.TileArray{Float32,2,csdl_spec}, ct.TileArray{Float32,2,csdl_spec}}) do a, b
            pid = ct.bid(1)
            tile = ct.load(a, pid, (32, 32768))
            result = ct.cumsum(tile, Val(1); reverse=true)
            ct.store(b, pid, result)
            return
        end
        @test occursin("scan", ir)
        @test occursin("reverse=true", ir)
    end

    #=========================================================================
    # Different Element Types for CSDL
    #========================================================================#
    @testset "CSDL Float64 scan" begin
        ir = ct.code_tiled(Tuple{ct.TileArray{Float64,2,csdl_spec}, ct.TileArray{Float64,2,csdl_spec}}) do a, b
            pid = ct.bid(1)
            tile = ct.load(a, pid, (32, 32768))
            result = ct.cumsum(tile, Val(1))
            ct.store(b, pid, result)
            return
        end
        @test occursin("scan", ir)
        @test occursin("f64", ir)
    end

    @testset "CSDL Int32 scan" begin
        ir = ct.code_tiled(Tuple{ct.TileArray{Int32,2,csdl_spec}, ct.TileArray{Int32,2,csdl_spec}}) do a, b
            pid = ct.bid(1)
            tile = ct.load(a, pid, (32, 32768))
            result = ct.cumsum(tile, Val(1))
            ct.store(b, pid, result)
            return
        end
        @test occursin("scan", ir)
        @test occursin("i32", ir)
    end

    #=========================================================================
    # Shape Preservation for CSDL
    #========================================================================#
    @testset "CSDL output shape matches input" begin
        ir = ct.code_tiled(Tuple{ct.TileArray{Float32,2,csdl_spec}, ct.TileArray{Float32,2,csdl_spec}}) do a, b
            pid = ct.bid(1)
            tile = ct.load(a, pid, (32, 32768))
            result = ct.cumsum(tile, Val(1))
            ct.store(b, pid, result)
            return
        end
        @test occursin("tile<32x32768xf32>", ir)
    end

    @testset "CSDL output shape along axis 0" begin
        ir = ct.code_tiled(Tuple{ct.TileArray{Float32,2,csdl_spec}, ct.TileArray{Float32,2,csdl_spec}}) do a, b
            pid = ct.bid(1)
            tile = ct.load(a, pid, (32, 32768))
            result = ct.cumsum(tile, Val(0))
            ct.store(b, pid, result)
            return
        end
        @test occursin("tile<32x32768xf32>", ir)
    end

    #=========================================================================
    # Type Stability for CSDL
    #========================================================================#
    @testset "CSDL type stability" begin
        # Create CSDL tile type
        tile = ct.Tile{Float32, (TILE_W, TILE_H)}()

        @testset "CSDL cumsum return type" begin
            return_types = Base.return_types(ct.cumsum, (typeof(tile), Val{1}))
            @test return_types isa Vector{Any}
            @test !isempty(return_types)
            @test return_types[1] == ct.Tile{Float32, (TILE_W, TILE_H)}
        end

        @testset "CSDL cumprod return type" begin
            return_types = Base.return_types(ct.cumprod, (typeof(tile), Val{1}))
            @test return_types isa Vector{Any}
            @test !isempty(return_types)
            @test return_types[1] == ct.Tile{Float32, (TILE_W, TILE_H)}
        end

        @testset "CSDL reverse cumsum return type" begin
            return_types = Base.return_types(ct.cumsum, (typeof(tile), Val{1}; reverse=true))
            @test return_types isa Vector{Any}
            @test !isempty(return_types)
            @test return_types[1] == ct.Tile{Float32, (TILE_W, TILE_H)}
        end
    end

    #=========================================================================
    # 3D Extension for CSDL (batch processing)
    #========================================================================#
    @testset "CSDL 3D scan (batch of CSDL tiles)" begin
        # 3D array where each z-slice is a CSDL tile
        csdl_3d_spec = ct.ArraySpec{3}(8, true)

        ir = ct.code_tiled(Tuple{
            ct.TileArray{Float32,3,csdl_3d_spec},
            ct.TileArray{Float32,3,csdl_3d_spec}
        }) do a, b
            pid = ct.bid(1)
            # Load: warp (32) × partition (32768) × batch (8)
            tile = ct.load(a, pid, (32, 32768, 8))
            # Scan along partition dimension
            result = ct.cumsum(tile, Val(1))
            ct.store(b, pid, result)
            return
        end
        @test occursin("scan", ir)
        @test occursin("dim=1", ir)
        @test occursin("tile<32x32768x8xf32>", ir)
    end

    #=========================================================================
    # Direct Scan API Tests
    #========================================================================#
    @testset "CSDL direct scan with :add" begin
        ir = ct.code_tiled(Tuple{ct.TileArray{Float32,2,csdl_spec}, ct.TileArray{Float32,2,csdl_spec}}) do a, b
            pid = ct.bid(1)
            tile = ct.load(a, pid, (32, 32768))
            result = ct.scan(tile, Val(1), :add)
            ct.store(b, pid, result)
            return
        end
        @test occursin("scan", ir)
        @test occursin("addf", ir)
    end

    @testset "CSDL direct scan with :mul" begin
        ir = ct.code_tiled(Tuple{ct.TileArray{Float32,2,csdl_spec}, ct.TileArray{Float32,2,csdl_spec}}) do a, b
            pid = ct.bid(1)
            tile = ct.load(a, pid, (32, 32768))
            result = ct.scan(tile, Val(1), :mul)
            ct.store(b, pid, result)
            return
        end
        @test occursin("scan", ir)
        @test occursin("mulf", ir)
    end

    #=========================================================================
    # Small Tile Tests (for fast validation)
    #========================================================================#
    @testset "Small tile cumsum" begin
        # Smaller tile for faster testing cycles
        ir = ct.code_tiled(Tuple{ct.TileArray{Float32,2,csdl_small_spec}, ct.TileArray{Float32,2,csdl_small_spec}}) do a, b
            pid = ct.bid(1)
            tile = ct.load(a, pid, (8, 256))
            result = ct.cumsum(tile, Val(1))
            ct.store(b, pid, result)
            return
        end
        @test occursin("scan", ir)
        @test occursin("addf", ir)
        @test occursin("tile<8x256xf32>", ir)
    end

    @testset "Small tile Int64 scan" begin
        ir = ct.code_tiled(Tuple{ct.TileArray{Int64,2,csdl_small_spec}, ct.TileArray{Int64,2,csdl_small_spec}}) do a, b
            pid = ct.bid(1)
            tile = ct.load(a, pid, (8, 256))
            result = ct.cumsum(tile, Val(1))
            ct.store(b, pid, result)
            return
        end
        @test occursin("scan", ir)
        @test occursin("i64", ir)
    end
end

#=============================================================================
# CSDL Algorithm Notes (for documentation)
#=============================================================================
#
# The CSDL (Chained Scan with Decoupled Lookback) algorithm is designed for
# efficient parallel prefix sum on GPUs with the following characteristics:
#
# 1. Single-phase operation: No separate global synchronization phases
# 2. Chained lookback: Blocks progressively accumulate previous blocks
#    without device-wide synchronization
# 3. O(1) amortized lookback: Each block looks back at most once on average
# 4. Memory efficient: No auxiliary scan reduction array needed
#
# Tile IR Mapping:
# - scan operation: Base Tile IR scan with custom body
# - addf/mulf: Intra-warp accumulation in scan body
# - identities: 0.0 for sum, 1.0 for product
# - dim parameter: Selects partition dimension for lookback
# - reverse parameter: For trailing segment handling
#
# Expected Tile IR output:
# ```
# %result = scan %input dim=1 reverse=false identities=[0.000000e+00 : f32]
#     : tile<32x32768xf32> -> tile<32x32768xf32>
# (%arg0: tile<f32>, %arg1: tile<f32>) {
#   %2 = addf %arg0, %arg1 : tile<f32>
#   yield %2 : tile<f32>
# }
# ```
