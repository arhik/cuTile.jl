# Tile IR intrinsics
#
# Organized according to https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html

module Intrinsics

using ..cuTile: Tile, TileArray, Constant, TensorView, PartitionView
using ..cuTile: Signedness, SignednessSigned, SignednessUnsigned
using ..cuTile: ComparisonPredicate, CmpLessThan, CmpLessThanOrEqual, CmpGreaterThan, CmpGreaterThanOrEqual, CmpEqual, CmpNotEqual

# Segmented scan intrinsics - returns Tile with same element type as values
# Takes values tile, flags tile (both same shape), axis, fn symbol, and reverse flag
segmented_scan(values::Tile{T,S}, flags::Tile{Bool,S}, axis::Val{A}, fn::Symbol, reverse::Bool) where {T,S,A} = error("Not implemented")
exclusive_segmented_scan(values::Tile{T,S}, flags::Tile{Bool,S}, axis::Val{A}, fn::Symbol, reverse::Bool) where {T,S,A} = error("Not implemented")

# Custom scan with binary operator (e.g., WrappedAddMod for modulo arithmetic)
# The op is a type parameter M for compile-time modulus
scan_with_op(values::Tile{T,S}, axis::Val{A}, op::Type{Val{M}}, reverse::Bool) where {T,S,A,M} = error("Not implemented")


end

# NOTE: Due to JuliaLang/julia#60583, intrinsics may be called during constant evaluation.
#       Because of that, such intrinsics (such as basic arithmetic) need to provide an
#       implementation that actually computes a valid result using Julia intrinsics.

emit_intrinsic!(ctx::CGCtx, @nospecialize(func), args) = missing

include("intrinsics/core.jl")
include("intrinsics/conversions.jl")
include("intrinsics/arithmetic.jl")
include("intrinsics/math.jl")
include("intrinsics/memory.jl")
include("intrinsics/atomics.jl")
include("intrinsics/views.jl")
include("intrinsics/misc.jl")

include("intrinsics/julia.jl")
