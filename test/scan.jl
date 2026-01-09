
#=============================================================================
 Scan (Prefix Sum) Operation Tests
=============================================================================#

@testset "Scan Operations" begin
    import cuTile: Tile, scan, cumsum, cumprod, tile_shape

    #=========================================================================
     Basic Function Tests
    =========================================================================#
    @testset "Basic scan compilation" begin
        tile = Tile{Float32, (4, 5)}()
        result = scan(tile, Val(1); fn=:add)
        @test eltype(result) == Float32
        @test tile_shape(result) == (4, 5)

        # Test scan with :mul function
        result_mul = scan(tile, Val(1); fn=:mul)
        @test eltype(result_mul) == Float32
        @test tile_shape(result_mul) == (4, 5)
    end

    @testset "cumsum" begin
        # cumsum along axis 1
        tile = Tile{Float32, (4, 5)}()
        result = cumsum(tile, Val(1))
        @test eltype(result) == Float32
        @test tile_shape(result) == (4, 5)

        # cumsum along axis 0
        result_0 = cumsum(tile, Val(0))
        @test eltype(result_0) == Float32
        @test tile_shape(result_0) == (4, 5)

        # cumsum with different types
        tile_64 = Tile{Float64, (3, 4)}()
        result_64 = cumsum(tile_64, Val(1))
        @test eltype(result_64) == Float64
        @test tile_shape(result_64) == (3, 4)
    end

    @testset "cumprod" begin
        # cumprod along axis 1
        tile = Tile{Float32, (4, 5)}()
        result = cumprod(tile, Val(1))
        @test eltype(result) == Float32
        @test tile_shape(result) == (4, 5)

        # cumprod along axis 0
        result_0 = cumprod(tile, Val(0))
        @test eltype(result_0) == Float32
        @test tile_shape(result_0) == (4, 5)
    end

    @testset "Reverse scan" begin
        tile = Tile{Float32, (4, 5)}()

        # Reverse scan with :add
        result_rev_add = scan(tile, Val(1); fn=:add, reverse=true)
        @test eltype(result_rev_add) == Float32
        @test tile_shape(result_rev_add) == (4, 5)

        # Reverse scan with :mul
        result_rev_mul = scan(tile, Val(1); fn=:mul, reverse=true)
        @test eltype(result_rev_mul) == Float32
        @test tile_shape(result_rev_mul) == (4, 5)

        # Reverse cumsum
        result = cumsum(tile, Val(1); reverse=true)
        @test eltype(result) == Float32
        @test tile_shape(result) == (4, 5)

        # Reverse cumprod
        result = cumprod(tile, Val(1); reverse=true)
        @test eltype(result) == Float32
        @test tile_shape(result) == (4, 5)
    end

    #=========================================================================
     Element Types
    =========================================================================#
    @testset "Float32" begin
        tile = Tile{Float32, (3, 4)}()
        result = scan(tile, Val(1); fn=:add)
        @test eltype(result) == Float32
        @test tile_shape(result) == (3, 4)
    end

    @testset "Float64" begin
        tile = Tile{Float64, (3, 4)}()
        result = scan(tile, Val(1); fn=:add)
        @test eltype(result) == Float64
        @test tile_shape(result) == (3, 4)
    end

    @testset "Int32" begin
        tile = Tile{Int32, (3, 4)}()
        result = scan(tile, Val(1); fn=:add)
        @test eltype(result) == Int32
        @test tile_shape(result) == (3, 4)
    end

    @testset "Int64" begin
        tile = Tile{Int64, (3, 4)}()
        result = scan(tile, Val(1); fn=:add)
        @test eltype(result) == Int64
        @test tile_shape(result) == (3, 4)
    end

    #=========================================================================
     Different Axes
    =========================================================================#
    @testset "Axis 0" begin
        tile = Tile{Float32, (2, 3, 4)}()
        result = scan(tile, Val(0); fn=:add)
        @test eltype(result) == Float32
        @test tile_shape(result) == (2, 3, 4)
    end

    @testset "Axis 1" begin
        tile = Tile{Float32, (2, 3, 4)}()
        result = scan(tile, Val(1); fn=:add)
        @test eltype(result) == Float32
        @test tile_shape(result) == (2, 3, 4)
    end

    @testset "Axis 2" begin
        tile = Tile{Float32, (2, 3, 4)}()
        result = scan(tile, Val(2); fn=:add)
        @test eltype(result) == Float32
        @test tile_shape(result) == (2, 3, 4)
    end

    #=========================================================================
     Shape Preservation
    =========================================================================#
    @testset "Shape preservation" begin
        # 2D tile
        tile = Tile{Float32, (4, 5)}()
        result = scan(tile, Val(1); fn=:add)
        @test tile_shape(result) == (4, 5)

        # 3D tile
        tile_3d = Tile{Float32, (2, 3, 4)}()
        result_3d = scan(tile_3d, Val(0); fn=:add)
        @test tile_shape(result_3d) == (2, 3, 4)
    end

    #=========================================================================
     Type Stability
    =========================================================================#
    @testset "Type stability" begin
        tile = Tile{Float32, (4, 5)}()

        # Check return types are concrete
        @testset "scan return type" begin
            return_types = Base.return_types(scan, (typeof(tile), Val{1}))
            @test return_types isa Vector{Any}
            @test !isempty(return_types)
            # First return type should be Tile{Float32, (4, 5)}
            @test return_types[1] == Tile{Float32, (4, 5)}
        end

        @testset "cumsum return type" begin
            return_types = Base.return_types(cumsum, (typeof(tile), Val{1}))
            @test return_types isa Vector{Any}
            @test !isempty(return_types)
            @test return_types[1] == Tile{Float32, (4, 5)}
        end

        @testset "cumprod return type" begin
            return_types = Base.return_types(cumprod, (typeof(tile), Val{1}))
            @test return_types isa Vector{Any}
            @test !isempty(return_types)
            @test return_types[1] == Tile{Float32, (4, 5)}
        end
    end

    #=========================================================================
     Scan Identity Tests
    =========================================================================#
    @testset "Scan identity values" begin
        # Sum scan should use identity 0
        tile = Tile{Float32, (4, 5)}()
        result_add = scan(tile, Val(1); fn=:add)
        @test eltype(result_add) == Float32

        # Product scan should use identity 1
        result_mul = scan(tile, Val(1); fn=:mul)
        @test eltype(result_mul) == Float32
    end
end
