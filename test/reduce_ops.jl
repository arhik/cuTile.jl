using cuTile
import cuTile as ct
using CUDA
using Test

"""
    reduce_ops.jl

Tests for tile reduction operations (min, max, sum, xor, or, and).
Tests 1D tile reductions across different element types.

Note: 2D tile reduction tests are temporarily disabled due to tileiras compilation issues.
These need investigation with the NVIDIA Tile IR compiler.
"""

@testset "Reduce Operations" begin

#======================================================================#
# UInt16 reductions
#======================================================================#

@testset "UInt16 reduce_min" begin
    sz, N = 32, 4096
    a_gpu = CUDA.rand(UInt16, N)
    b_gpu = CUDA.zeros(UInt16, cld(N, sz))
    function kernel(a::ct.TileArray{UInt16,1}, b::ct.TileArray{UInt16,1}, tileSz::ct.Constant{Int})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (tileSz[],))
        ct.store(b, pid, ct.reduce_min(tile, 1))
        return nothing
    end
    CUDA.@sync ct.launch(kernel, cld(N, sz), a_gpu, b_gpu, ct.Constant(sz))
    res = Array(b_gpu)
    a_cpu = Array(a_gpu)
    @test minimum(reshape(a_cpu, sz, :), dims=1)[:] == res
end

@testset "UInt16 reduce_max" begin
    sz, N = 32, 4096
    a_gpu = CUDA.rand(UInt16, N)
    b_gpu = CUDA.zeros(UInt16, cld(N, sz))
    function kernel(a::ct.TileArray{UInt16,1}, b::ct.TileArray{UInt16,1}, tileSz::ct.Constant{Int})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (tileSz[],))
        ct.store(b, pid, ct.reduce_max(tile, 1))
        return nothing
    end
    CUDA.@sync ct.launch(kernel, cld(N, sz), a_gpu, b_gpu, ct.Constant(sz))
    res = Array(b_gpu)
    a_cpu = Array(a_gpu)
    @test maximum(reshape(a_cpu, sz, :), dims=1)[:] == res
end

@testset "UInt16 reduce_sum" begin
    sz, N = 32, 4096
    a_gpu = CUDA.rand(UInt16, N)
    b_gpu = CUDA.zeros(UInt16, cld(N, sz))
    function kernel(a::ct.TileArray{UInt16,1}, b::ct.TileArray{UInt16,1}, tileSz::ct.Constant{Int})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (tileSz[],))
        ct.store(b, pid, ct.reduce_sum(tile, 1))
        return nothing
    end
    CUDA.@sync ct.launch(kernel, cld(N, sz), a_gpu, b_gpu, ct.Constant(sz))
    res = Array(b_gpu)
    a_cpu = Array(a_gpu)
    @test (sum(reshape(a_cpu, sz, :), dims=1)[:] .& typemax(UInt16)) == res
end

@testset "UInt16 reduce_xor" begin
    sz, N = 32, 4096
    a_gpu = CUDA.rand(UInt16, N)
    b_gpu = CUDA.zeros(UInt16, cld(N, sz))
    function kernel(a::ct.TileArray{UInt16,1}, b::ct.TileArray{UInt16,1}, tileSz::ct.Constant{Int})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (tileSz[],))
        ct.store(b, pid, ct.reduce_xor(tile, 1))
        return nothing
    end
    CUDA.@sync ct.launch(kernel, cld(N, sz), a_gpu, b_gpu, ct.Constant(sz))
    res = Array(b_gpu)
    a_cpu = Array(a_gpu)
    @test mapslices(x -> reduce(⊻, x), reshape(a_cpu, sz, :), dims=1)[:] == res
end

@testset "UInt16 reduce_or" begin
    sz, N = 32, 4096
    a_gpu = CUDA.rand(UInt16, N)
    b_gpu = CUDA.zeros(UInt16, cld(N, sz))
    function kernel(a::ct.TileArray{UInt16,1}, b::ct.TileArray{UInt16,1}, tileSz::ct.Constant{Int})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (tileSz[],))
        ct.store(b, pid, ct.reduce_or(tile, 1))
        return nothing
    end
    CUDA.@sync ct.launch(kernel, cld(N, sz), a_gpu, b_gpu, ct.Constant(sz))
    res = Array(b_gpu)
    a_cpu = Array(a_gpu)
    @test mapslices(x -> reduce(|, x), reshape(a_cpu, sz, :), dims=1)[:] == res
end

@testset "UInt16 reduce_and" begin
    sz, N = 32, 4096
    a_gpu = CUDA.rand(UInt16, N)
    b_gpu = CUDA.zeros(UInt16, cld(N, sz))
    function kernel(a::ct.TileArray{UInt16,1}, b::ct.TileArray{UInt16,1}, tileSz::ct.Constant{Int})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (tileSz[],))
        ct.store(b, pid, ct.reduce_and(tile, 1))
        return nothing
    end
    CUDA.@sync ct.launch(kernel, cld(N, sz), a_gpu, b_gpu, ct.Constant(sz))
    res = Array(b_gpu)
    a_cpu = Array(a_gpu)
    @test mapslices(x -> reduce(&, x), reshape(a_cpu, sz, :), dims=1)[:] == res
end

#======================================================================#
# UInt32 reductions
#======================================================================#

@testset "UInt32 reduce_min" begin
    sz, N = 32, 4096
    a_gpu = CUDA.rand(UInt32, N)
    b_gpu = CUDA.zeros(UInt32, cld(N, sz))
    function kernel(a::ct.TileArray{UInt32,1}, b::ct.TileArray{UInt32,1}, tileSz::ct.Constant{Int})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (tileSz[],))
        ct.store(b, pid, ct.reduce_min(tile, 1))
        return nothing
    end
    CUDA.@sync ct.launch(kernel, cld(N, sz), a_gpu, b_gpu, ct.Constant(sz))
    res = Array(b_gpu)
    a_cpu = Array(a_gpu)
    @test minimum(reshape(a_cpu, sz, :), dims=1)[:] == res
end

@testset "UInt32 reduce_max" begin
    sz, N = 32, 4096
    a_gpu = CUDA.rand(UInt32, N)
    b_gpu = CUDA.zeros(UInt32, cld(N, sz))
    function kernel(a::ct.TileArray{UInt32,1}, b::ct.TileArray{UInt32,1}, tileSz::ct.Constant{Int})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (tileSz[],))
        ct.store(b, pid, ct.reduce_max(tile, 1))
        return nothing
    end
    CUDA.@sync ct.launch(kernel, cld(N, sz), a_gpu, b_gpu, ct.Constant(sz))
    res = Array(b_gpu)
    a_cpu = Array(a_gpu)
    @test maximum(reshape(a_cpu, sz, :), dims=1)[:] == res
end

@testset "UInt32 reduce_sum" begin
    sz, N = 32, 4096
    a_gpu = CUDA.rand(UInt32, N)
    b_gpu = CUDA.zeros(UInt32, cld(N, sz))
    function kernel(a::ct.TileArray{UInt32,1}, b::ct.TileArray{UInt32,1}, tileSz::ct.Constant{Int})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (tileSz[],))
        ct.store(b, pid, ct.reduce_sum(tile, 1))
        return nothing
    end
    CUDA.@sync ct.launch(kernel, cld(N, sz), a_gpu, b_gpu, ct.Constant(sz))
    res = Array(b_gpu)
    a_cpu = Array(a_gpu)
    @test (sum(reshape(a_cpu, sz, :), dims=1)[:] .& typemax(UInt32)) == res
end

@testset "UInt32 reduce_xor" begin
    sz, N = 32, 4096
    a_gpu = CUDA.rand(UInt32, N)
    b_gpu = CUDA.zeros(UInt32, cld(N, sz))
    function kernel(a::ct.TileArray{UInt32,1}, b::ct.TileArray{UInt32,1}, tileSz::ct.Constant{Int})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (tileSz[],))
        ct.store(b, pid, ct.reduce_xor(tile, 1))
        return nothing
    end
    CUDA.@sync ct.launch(kernel, cld(N, sz), a_gpu, b_gpu, ct.Constant(sz))
    res = Array(b_gpu)
    a_cpu = Array(a_gpu)
    @test mapslices(x -> reduce(⊻, x), reshape(a_cpu, sz, :), dims=1)[:] == res
end

@testset "UInt32 reduce_or" begin
    sz, N = 32, 4096
    a_gpu = CUDA.rand(UInt32, N)
    b_gpu = CUDA.zeros(UInt32, cld(N, sz))
    function kernel(a::ct.TileArray{UInt32,1}, b::ct.TileArray{UInt32,1}, tileSz::ct.Constant{Int})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (tileSz[],))
        ct.store(b, pid, ct.reduce_or(tile, 1))
        return nothing
    end
    CUDA.@sync ct.launch(kernel, cld(N, sz), a_gpu, b_gpu, ct.Constant(sz))
    res = Array(b_gpu)
    a_cpu = Array(a_gpu)
    @test mapslices(x -> reduce(|, x), reshape(a_cpu, sz, :), dims=1)[:] == res
end

@testset "UInt32 reduce_and" begin
    sz, N = 32, 4096
    a_gpu = CUDA.rand(UInt32, N)
    b_gpu = CUDA.zeros(UInt32, cld(N, sz))
    function kernel(a::ct.TileArray{UInt32,1}, b::ct.TileArray{UInt32,1}, tileSz::ct.Constant{Int})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (tileSz[],))
        ct.store(b, pid, ct.reduce_and(tile, 1))
        return nothing
    end
    CUDA.@sync ct.launch(kernel, cld(N, sz), a_gpu, b_gpu, ct.Constant(sz))
    res = Array(b_gpu)
    a_cpu = Array(a_gpu)
    @test mapslices(x -> reduce(&, x), reshape(a_cpu, sz, :), dims=1)[:] == res
end

#======================================================================#
# UInt64 reductions
#======================================================================#

@testset "UInt64 reduce_min" begin
    sz, N = 32, 4096
    a_gpu = CUDA.rand(UInt64, N)
    b_gpu = CUDA.zeros(UInt64, cld(N, sz))
    function kernel(a::ct.TileArray{UInt64,1}, b::ct.TileArray{UInt64,1}, tileSz::ct.Constant{Int})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (tileSz[],))
        ct.store(b, pid, ct.reduce_min(tile, 1))
        return nothing
    end
    CUDA.@sync ct.launch(kernel, cld(N, sz), a_gpu, b_gpu, ct.Constant(sz))
    res = Array(b_gpu)
    a_cpu = Array(a_gpu)
    @test minimum(reshape(a_cpu, sz, :), dims=1)[:] == res
end

@testset "UInt64 reduce_max" begin
    sz, N = 32, 4096
    a_gpu = CUDA.rand(UInt64, N)
    b_gpu = CUDA.zeros(UInt64, cld(N, sz))
    function kernel(a::ct.TileArray{UInt64,1}, b::ct.TileArray{UInt64,1}, tileSz::ct.Constant{Int})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (tileSz[],))
        ct.store(b, pid, ct.reduce_max(tile, 1))
        return nothing
    end
    CUDA.@sync ct.launch(kernel, cld(N, sz), a_gpu, b_gpu, ct.Constant(sz))
    res = Array(b_gpu)
    a_cpu = Array(a_gpu)
    @test maximum(reshape(a_cpu, sz, :), dims=1)[:] == res
end

@testset "UInt64 reduce_sum" begin
    sz, N = 32, 4096
    a_gpu = CUDA.rand(UInt64, N)
    b_gpu = CUDA.zeros(UInt64, cld(N, sz))
    function kernel(a::ct.TileArray{UInt64,1}, b::ct.TileArray{UInt64,1}, tileSz::ct.Constant{Int})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (tileSz[],))
        ct.store(b, pid, ct.reduce_sum(tile, 1))
        return nothing
    end
    CUDA.@sync ct.launch(kernel, cld(N, sz), a_gpu, b_gpu, ct.Constant(sz))
    res = Array(b_gpu)
    a_cpu = Array(a_gpu)
    @test (sum(reshape(a_cpu, sz, :), dims=1)[:] .& typemax(UInt64)) == res
end

@testset "UInt64 reduce_xor" begin
    sz, N = 32, 4096
    a_gpu = CUDA.rand(UInt64, N)
    b_gpu = CUDA.zeros(UInt64, cld(N, sz))
    function kernel(a::ct.TileArray{UInt64,1}, b::ct.TileArray{UInt64,1}, tileSz::ct.Constant{Int})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (tileSz[],))
        ct.store(b, pid, ct.reduce_xor(tile, 1))
        return nothing
    end
    CUDA.@sync ct.launch(kernel, cld(N, sz), a_gpu, b_gpu, ct.Constant(sz))
    res = Array(b_gpu)
    a_cpu = Array(a_gpu)
    @test mapslices(x -> reduce(⊻, x), reshape(a_cpu, sz, :), dims=1)[:] == res
end

@testset "UInt64 reduce_or" begin
    sz, N = 32, 4096
    a_gpu = CUDA.rand(UInt64, N)
    b_gpu = CUDA.zeros(UInt64, cld(N, sz))
    function kernel(a::ct.TileArray{UInt64,1}, b::ct.TileArray{UInt64,1}, tileSz::ct.Constant{Int})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (tileSz[],))
        ct.store(b, pid, ct.reduce_or(tile, 1))
        return nothing
    end
    CUDA.@sync ct.launch(kernel, cld(N, sz), a_gpu, b_gpu, ct.Constant(sz))
    res = Array(b_gpu)
    a_cpu = Array(a_gpu)
    @test mapslices(x -> reduce(|, x), reshape(a_cpu, sz, :), dims=1)[:] == res
end

@testset "UInt64 reduce_and" begin
    sz, N = 32, 4096
    a_gpu = CUDA.rand(UInt64, N)
    b_gpu = CUDA.zeros(UInt64, cld(N, sz))
    function kernel(a::ct.TileArray{UInt64,1}, b::ct.TileArray{UInt64,1}, tileSz::ct.Constant{Int})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (tileSz[],))
        ct.store(b, pid, ct.reduce_and(tile, 1))
        return nothing
    end
    CUDA.@sync ct.launch(kernel, cld(N, sz), a_gpu, b_gpu, ct.Constant(sz))
    res = Array(b_gpu)
    a_cpu = Array(a_gpu)
    @test mapslices(x -> reduce(&, x), reshape(a_cpu, sz, :), dims=1)[:] == res
end

#======================================================================#
# Float16 reductions
#======================================================================#

@testset "Float16 reduce_min" begin
    sz, N = 32, 4096
    a_gpu = CUDA.rand(Float16, N)
    b_gpu = CUDA.zeros(Float16, cld(N, sz))
    function kernel(a::ct.TileArray{Float16,1}, b::ct.TileArray{Float16,1}, tileSz::ct.Constant{Int})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (tileSz[],))
        ct.store(b, pid, ct.reduce_min(tile, 1))
        return nothing
    end
    CUDA.@sync ct.launch(kernel, cld(N, sz), a_gpu, b_gpu, ct.Constant(sz))
    res = Array(b_gpu)
    a_cpu = Array(a_gpu)
    @test isapprox(minimum(reshape(a_cpu, sz, :), dims=1)[:], res)
end

@testset "Float16 reduce_max" begin
    sz, N = 32, 4096
    a_gpu = CUDA.rand(Float16, N)
    b_gpu = CUDA.zeros(Float16, cld(N, sz))
    function kernel(a::ct.TileArray{Float16,1}, b::ct.TileArray{Float16,1}, tileSz::ct.Constant{Int})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (tileSz[],))
        ct.store(b, pid, ct.reduce_max(tile, 1))
        return nothing
    end
    CUDA.@sync ct.launch(kernel, cld(N, sz), a_gpu, b_gpu, ct.Constant(sz))
    res = Array(b_gpu)
    a_cpu = Array(a_gpu)
    @test isapprox(maximum(reshape(a_cpu, sz, :), dims=1)[:], res)
end

@testset "Float16 reduce_sum" begin
    sz, N = 32, 4096
    a_gpu = CUDA.rand(Float16, N)
    b_gpu = CUDA.zeros(Float16, cld(N, sz))
    function kernel(a::ct.TileArray{Float16,1}, b::ct.TileArray{Float16,1}, tileSz::ct.Constant{Int})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (tileSz[],))
        ct.store(b, pid, ct.reduce_sum(tile, 1))
        return nothing
    end
    CUDA.@sync ct.launch(kernel, cld(N, sz), a_gpu, b_gpu, ct.Constant(sz))
    res = Array(b_gpu)
    a_cpu = Array(a_gpu)
    @test isapprox(sum(reshape(a_cpu, sz, :), dims=1)[:], res)
end

#======================================================================#
# Float32 reductions
#======================================================================#

@testset "Float32 reduce_min" begin
    sz, N = 32, 4096
    a_gpu = CUDA.rand(Float32, N)
    b_gpu = CUDA.zeros(Float32, cld(N, sz))
    function kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1}, tileSz::ct.Constant{Int})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (tileSz[],))
        ct.store(b, pid, ct.reduce_min(tile, 1))
        return nothing
    end
    CUDA.@sync ct.launch(kernel, cld(N, sz), a_gpu, b_gpu, ct.Constant(sz))
    res = Array(b_gpu)
    a_cpu = Array(a_gpu)
    @test isapprox(minimum(reshape(a_cpu, sz, :), dims=1)[:], res)
end

@testset "Float32 reduce_max" begin
    sz, N = 32, 4096
    a_gpu = CUDA.rand(Float32, N)
    b_gpu = CUDA.zeros(Float32, cld(N, sz))
    function kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1}, tileSz::ct.Constant{Int})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (tileSz[],))
        ct.store(b, pid, ct.reduce_max(tile, 1))
        return nothing
    end
    CUDA.@sync ct.launch(kernel, cld(N, sz), a_gpu, b_gpu, ct.Constant(sz))
    res = Array(b_gpu)
    a_cpu = Array(a_gpu)
    @test isapprox(maximum(reshape(a_cpu, sz, :), dims=1)[:], res)
end

@testset "Float32 reduce_sum" begin
    sz, N = 32, 4096
    a_gpu = CUDA.rand(Float32, N)
    b_gpu = CUDA.zeros(Float32, cld(N, sz))
    function kernel(a::ct.TileArray{Float32,1}, b::ct.TileArray{Float32,1}, tileSz::ct.Constant{Int})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (tileSz[],))
        ct.store(b, pid, ct.reduce_sum(tile, 1))
        return nothing
    end
    CUDA.@sync ct.launch(kernel, cld(N, sz), a_gpu, b_gpu, ct.Constant(sz))
    res = Array(b_gpu)
    a_cpu = Array(a_gpu)
    @test isapprox(sum(reshape(a_cpu, sz, :), dims=1)[:], res)
end

#======================================================================#
# Float64 reductions
#======================================================================#

@testset "Float64 reduce_min" begin
    sz, N = 32, 4096
    a_gpu = CUDA.rand(Float64, N)
    b_gpu = CUDA.zeros(Float64, cld(N, sz))
    function kernel(a::ct.TileArray{Float64,1}, b::ct.TileArray{Float64,1}, tileSz::ct.Constant{Int})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (tileSz[],))
        ct.store(b, pid, ct.reduce_min(tile, 1))
        return nothing
    end
    CUDA.@sync ct.launch(kernel, cld(N, sz), a_gpu, b_gpu, ct.Constant(sz))
    res = Array(b_gpu)
    a_cpu = Array(a_gpu)
    @test isapprox(minimum(reshape(a_cpu, sz, :), dims=1)[:], res)
end

@testset "Float64 reduce_max" begin
    sz, N = 32, 4096
    a_gpu = CUDA.rand(Float64, N)
    b_gpu = CUDA.zeros(Float64, cld(N, sz))
    function kernel(a::ct.TileArray{Float64,1}, b::ct.TileArray{Float64,1}, tileSz::ct.Constant{Int})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (tileSz[],))
        ct.store(b, pid, ct.reduce_max(tile, 1))
        return nothing
    end
    CUDA.@sync ct.launch(kernel, cld(N, sz), a_gpu, b_gpu, ct.Constant(sz))
    res = Array(b_gpu)
    a_cpu = Array(a_gpu)
    @test isapprox(maximum(reshape(a_cpu, sz, :), dims=1)[:], res)
end

@testset "Float64 reduce_sum" begin
    sz, N = 32, 4096
    a_gpu = CUDA.rand(Float64, N)
    b_gpu = CUDA.zeros(Float64, cld(N, sz))
    function kernel(a::ct.TileArray{Float64,1}, b::ct.TileArray{Float64,1}, tileSz::ct.Constant{Int})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (tileSz[],))
        ct.store(b, pid, ct.reduce_sum(tile, 1))
        return nothing
    end
    CUDA.@sync ct.launch(kernel, cld(N, sz), a_gpu, b_gpu, ct.Constant(sz))
    res = Array(b_gpu)
    a_cpu = Array(a_gpu)
    @test isapprox(sum(reshape(a_cpu, sz, :), dims=1)[:], res)
end

#======================================================================#
# Int32 reductions
#======================================================================#

@testset "Int32 reduce_min" begin
    sz, N = 32, 4096
    a_gpu = CUDA.rand(Int32, N)
    b_gpu = CUDA.zeros(Int32, cld(N, sz))
    function kernel(a::ct.TileArray{Int32,1}, b::ct.TileArray{Int32,1}, tileSz::ct.Constant{Int})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (tileSz[],))
        ct.store(b, pid, ct.reduce_min(tile, 1))
        return nothing
    end
    CUDA.@sync ct.launch(kernel, cld(N, sz), a_gpu, b_gpu, ct.Constant(sz))
    res = Array(b_gpu)
    a_cpu = Array(a_gpu)
    @test minimum(reshape(a_cpu, sz, :), dims=1)[:] == res
end

@testset "Int32 reduce_max" begin
    sz, N = 32, 4096
    a_gpu = CUDA.rand(Int32, N)
    b_gpu = CUDA.zeros(Int32, cld(N, sz))
    function kernel(a::ct.TileArray{Int32,1}, b::ct.TileArray{Int32,1}, tileSz::ct.Constant{Int})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (tileSz[],))
        ct.store(b, pid, ct.reduce_max(tile, 1))
        return nothing
    end
    CUDA.@sync ct.launch(kernel, cld(N, sz), a_gpu, b_gpu, ct.Constant(sz))
    res = Array(b_gpu)
    a_cpu = Array(a_gpu)
    @test maximum(reshape(a_cpu, sz, :), dims=1)[:] == res
end

@testset "Int32 reduce_sum" begin
    sz, N = 32, 4096
    a_gpu = CUDA.rand(Int32, N)
    b_gpu = CUDA.zeros(Int32, cld(N, sz))
    function kernel(a::ct.TileArray{Int32,1}, b::ct.TileArray{Int32,1}, tileSz::ct.Constant{Int})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (tileSz[],))
        ct.store(b, pid, ct.reduce_sum(tile, 1))
        return nothing
    end
    CUDA.@sync ct.launch(kernel, cld(N, sz), a_gpu, b_gpu, ct.Constant(sz))
    res = Array(b_gpu)
    a_cpu = Array(a_gpu)
    @test sum(reshape(a_cpu, sz, :), dims=1)[:] == res
end

@testset "Int32 reduce_and" begin
    sz, N = 32, 4096
    a_gpu = CUDA.rand(Int32, N)
    b_gpu = CUDA.zeros(Int32, cld(N, sz))
    function kernel(a::ct.TileArray{Int32,1}, b::ct.TileArray{Int32,1}, tileSz::ct.Constant{Int})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (tileSz[],))
        ct.store(b, pid, ct.reduce_and(tile, 1))
        return nothing
    end
    CUDA.@sync ct.launch(kernel, cld(N, sz), a_gpu, b_gpu, ct.Constant(sz))
    res = Array(b_gpu)
    a_cpu = Array(a_gpu)
    @test mapslices(x -> reduce(&, x), reshape(a_cpu, sz, :), dims=1)[:] == res
end

@testset "Int32 reduce_or" begin
    sz, N = 32, 4096
    a_gpu = CUDA.rand(Int32, N)
    b_gpu = CUDA.zeros(Int32, cld(N, sz))
    function kernel(a::ct.TileArray{Int32,1}, b::ct.TileArray{Int32,1}, tileSz::ct.Constant{Int})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (tileSz[],))
        ct.store(b, pid, ct.reduce_or(tile, 1))
        return nothing
    end
    CUDA.@sync ct.launch(kernel, cld(N, sz), a_gpu, b_gpu, ct.Constant(sz))
    res = Array(b_gpu)
    a_cpu = Array(a_gpu)
    @test mapslices(x -> reduce(|, x), reshape(a_cpu, sz, :), dims=1)[:] == res
end

@testset "Int32 reduce_xor" begin
    sz, N = 32, 4096
    a_gpu = CUDA.rand(Int32, N)
    b_gpu = CUDA.zeros(Int32, cld(N, sz))
    function kernel(a::ct.TileArray{Int32,1}, b::ct.TileArray{Int32,1}, tileSz::ct.Constant{Int})
        pid = ct.bid(1)
        tile = ct.load(a, pid, (tileSz[],))
        ct.store(b, pid, ct.reduce_xor(tile, 1))
        return nothing
    end
    CUDA.@sync ct.launch(kernel, cld(N, sz), a_gpu, b_gpu, ct.Constant(sz))
    res = Array(b_gpu)
    a_cpu = Array(a_gpu)
    @test mapslices(x -> reduce(⊻, x), reshape(a_cpu, sz, :), dims=1)[:] == res
end

end  # @testset "Reduce Operations"
