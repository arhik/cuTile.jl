# core Tile IR intrinsics

"""
    validate_tile_shape(shape, context::String)

Validate that all tile dimensions are powers of 2.
Tile IR requires all tile dimensions to be powers of 2.
Throws an error with a clear message if validation fails.
"""
function validate_tile_shape(shape, context::String)
    for (i, dim) in enumerate(shape)
        if dim <= 0
            error("$context: tile dimension $i must be positive, got $dim")
        end
        if !ispow2(dim)
            error("$context: tile dimension $i must be a power of 2, got $dim")
        end
    end
end

# cuda_tile.broadcast
@eval Intrinsics begin
    """
        broadcast(tile, shape_val)

    Explicitly broadcast a tile to a target shape.
    Compiled to cuda_tile.broadcast.
    """
    @noinline function broadcast(tile::Tile{T}, ::Val{Shape}) where {T, Shape}
        Tile{T, Shape}()
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.broadcast), args)
    cb = ctx.cb
    tt = ctx.tt

    # Get source tile
    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve source operand for broadcast()")

    # Get source element type
    source_type = unwrap_type(source.jltype)
    source_elem = source_type.parameters[1]

    # Extract target shape from the constant tuple argument
    target_shape_tuple = get_constant(ctx, args[2])
    target_shape_tuple isa Tuple || error("broadcast() shape must be a compile-time constant tuple")
    target_shape = collect(Int, target_shape_tuple)
    validate_tile_shape(target_shape, "broadcast")

    # If already the right shape, return unchanged
    if source.shape == target_shape
        return source
    end

    # Use the existing broadcast helper
    dtype = julia_to_tile_dtype!(tt, source_elem)
    result_v = broadcast_tile_to_shape!(cb, tt, source, target_shape, dtype)
    result_type_id = tile_type!(tt, dtype, target_shape)

    CGVal(result_v, result_type_id, Tile{source_elem, Tuple(target_shape)}, target_shape)
end

"""
    broadcast_tile_to_shape!(cb, tt, tv::CGVal, target_shape::Vector{Int}, dtype::TypeId) -> Value

Broadcast a tile to a target shape by inserting ReshapeOp (for leading 1s) and BroadcastOp.
Returns the value after broadcasting, or the original value if shapes already match.
"""
function broadcast_tile_to_shape!(cb::CodeBuilder, tt::TypeTable, tv::CGVal,
                                   target_shape::Vector{Int}, dtype::TypeId)
    src_shape = tv.shape

    # Already the right shape?
    if src_shape == target_shape
        return tv.v
    end

    current_val = tv.v
    current_shape = src_shape

    # Step 1: Add leading 1s via ReshapeOp if needed (dimension mismatch)
    if length(current_shape) < length(target_shape)
        # Prepend 1s to match target ndim
        n_extra = length(target_shape) - length(current_shape)
        new_shape = vcat(fill(1, n_extra), current_shape)
        reshaped_type = tile_type!(tt, dtype, new_shape)
        current_val = encode_ReshapeOp!(cb, reshaped_type, current_val)
        current_shape = new_shape
    end

    # Step 2: Broadcast dimensions that are 1 to target size
    if current_shape != target_shape
        broadcast_type = tile_type!(tt, dtype, target_shape)
        current_val = encode_BroadcastOp!(cb, broadcast_type, current_val)
    end

    current_val
end

# cuda_tile.cat
@eval Intrinsics begin
    """
        cat(tiles, axis_val)

    Concatenate two tiles along 0-indexed axis.
    Compiled to cuda_tile.cat.
    """
    @noinline function cat(tiles::Tuple{Tile{T, S1}, Tile{T, S2}}, ::Val{Axis}) where {T, S1, S2, Axis}
        ndims = length(S1)
        axis = Axis < 0 ? ndims + Axis : Axis
        result_shape = ntuple(ndims) do i
            if i == axis + 1  # 0-indexed axis, 1-indexed tuple access
                S1[i] + S2[i]
            else
                S1[i]
            end
        end
        Tile{T, result_shape}()
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.cat), args)
    cb = ctx.cb
    tt = ctx.tt

    # Emit tuple value to get CGVal with component refs in .tuple
    tuple_tv = emit_value!(ctx, args[1])
    tuple_tv === nothing && error("cat() cannot resolve tuple argument")

    # Extract component refs from .tuple field
    tuple_tv.tuple !== nothing || error("cat() requires tuple with tracked components")
    length(tuple_tv.tuple) == 2 || error("cat() expects exactly 2 tiles, got $(length(tuple_tv.tuple))")

    # Emit tiles from refs (looks up ctx.values, not stmts!)
    lhs = emit_value!(ctx, tuple_tv.tuple[1])
    rhs = emit_value!(ctx, tuple_tv.tuple[2])
    (lhs === nothing || rhs === nothing) && error("Cannot resolve tile operands for cat()")

    # Get axis from Val{Axis}
    axis_val = get_constant(ctx, args[2])
    axis_val isa Integer || error("cat() axis must be a compile-time constant integer")

    # Handle negative axis
    lhs_shape = lhs.shape
    ndims = length(lhs_shape)
    axis = axis_val < 0 ? ndims + axis_val : axis_val

    # Compute output shape - concatenate along the axis
    rhs_shape = rhs.shape
    output_shape = collect(Int, lhs_shape)
    output_shape[axis + 1] += rhs_shape[axis + 1]  # 1-based indexing
    validate_tile_shape(output_shape, "cat")

    # Get element type
    elem_type = unwrap_type(lhs.jltype).parameters[1]

    # Create output tile type
    dtype = julia_to_tile_dtype!(tt, elem_type)
    output_tile_type = tile_type!(tt, dtype, output_shape)

    # Emit CatOp (axis is 0-indexed for bytecode)
    result = encode_CatOp!(cb, output_tile_type, lhs.v, rhs.v, axis)

    CGVal(result, output_tile_type, Tile{elem_type, Tuple(output_shape)}, output_shape)
end

# cuda_tile.constant
@eval Intrinsics begin
    """
        constant(shape, value, T)

    Create a tile filled with a constant value.
    Compiled to cuda_tile.constant.
    """
    @noinline function constant(shape::NTuple{N, Int}, value, ::Type{T}) where {N, T}
        Tile{T, shape}()
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.constant), args)
    cb = ctx.cb
    tt = ctx.tt

    # Extract shape
    shape = get_constant(ctx, args[1])
    shape isa Tuple || error("full() shape must be a compile-time constant tuple")
    tile_shape = collect(Int, shape)
    validate_tile_shape(tile_shape, "full")

    # Extract value
    value = @something get_constant(ctx, args[2]) error("full() value must be a compile-time constant")

    # Extract dtype from Type{T} argument
    elem_type = @something get_constant(ctx, args[3]) error("constant() requires a compile-time element type")

    dtype = julia_to_tile_dtype!(tt, elem_type)
    tile_type = tile_type!(tt, dtype, tile_shape)

    # Create constant directly at target shape
    value_bytes = constant_to_bytes(value, elem_type)
    result = encode_ConstantOp!(cb, tile_type, value_bytes)

    CGVal(result, tile_type, Tile{elem_type, Tuple(tile_shape)}, tile_shape)
end

# TODO: cuda_tile.entry

# cuda_tile.extract
@eval Intrinsics begin
    """
        extract(tile, index_val, shape_val)

    Extract a sub-tile from tile at 0-indexed slice indices.
    Compiled to cuda_tile.extract.
    """
    @noinline function extract(tile::Tile{T}, ::Val{Index}, ::Val{Shape}) where {T, Index, Shape}
        Tile{T, Shape}()
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.extract), args)
    cb = ctx.cb
    tt = ctx.tt

    # Get source tile
    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve source operand for extract()")

    # Extract index from Val{Index} argument
    index_tuple = get_constant(ctx, args[2])
    index_tuple isa Tuple || error("extract() index must be a compile-time constant tuple")

    # Extract shape from Val{Shape} argument
    shape_tuple = get_constant(ctx, args[3])
    shape_tuple isa Tuple || error("extract() shape must be a compile-time constant tuple")
    output_shape = collect(Int, shape_tuple)
    validate_tile_shape(output_shape, "extract")

    # Get element type
    elem_type = unwrap_type(source.jltype).parameters[1]

    # Create output tile type
    dtype = julia_to_tile_dtype!(tt, elem_type)
    output_tile_type = tile_type!(tt, dtype, output_shape)

    # Create constant index values (0D i32 tiles)
    scalar_i32 = tile_type!(tt, I32(tt), Int[])
    index_vals = Value[]
    for idx in index_tuple
        idx_bytes = collect(reinterpret(UInt8, [Int32(idx)]))
        idx_val = encode_ConstantOp!(cb, scalar_i32, idx_bytes)
        push!(index_vals, idx_val)
    end

    # Emit ExtractOp
    result = encode_ExtractOp!(cb, output_tile_type, source.v, index_vals)

    CGVal(result, output_tile_type, Tile{elem_type, Tuple(output_shape)}, output_shape)
end

# TODO: cuda_tile.get_global

# cuda_tile.get_num_tile_blocks
@eval Intrinsics begin
    """
        get_num_tile_blocks(axis)::Int32

    Get the grid size along the given axis (0=x, 1=y, 2=z).
    Compiled to cuda_tile.get_num_tile_blocks.
    """
    @noinline get_num_tile_blocks(axis::Integer) = compilerbarrier(:const, zero(Int32))
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.get_num_tile_blocks), args)
    axis = @something get_constant(ctx, args[1]) error("get_num_tile_blocks() axis must be a compile-time constant")
    axis in (0, 1, 2) || error("get_num_tile_blocks() axis must be 0, 1, or 2, got $axis")

    res_type = tile_type!(ctx.tt, I32(ctx.tt), Int[])
    nb_x, nb_y, nb_z = encode_GetNumTileBlocksOp!(ctx.cb, res_type, res_type, res_type)

    CGVal((nb_x, nb_y, nb_z)[axis + 1], res_type, Int32)
end

# cuda_tile.get_tile_block_id
@eval Intrinsics begin
    """
        get_tile_block_id(axis)::Int32

    Get the block ID along the given axis (0=x, 1=y, 2=z).
    Compiled to cuda_tile.get_tile_block_id.
    """
    @noinline get_tile_block_id(axis::Integer) = compilerbarrier(:const, zero(Int32))
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.get_tile_block_id), args)
    axis = @something get_constant(ctx, args[1]) error("get_tile_block_id() axis must be a compile-time constant")
    axis in (0, 1, 2) || error("get_tile_block_id() axis must be 0, 1, or 2, got $axis")

    res_type = tile_type!(ctx.tt, I32(ctx.tt), Int[])
    bid_x, bid_y, bid_z = encode_GetTileBlockIdOp!(ctx.cb, res_type, res_type, res_type)
    result = (bid_x, bid_y, bid_z)[axis + 1]

    CGVal(result, res_type, Int32)
end

# TODO: cuda_tile.global

# cuda_tile.iota
@eval Intrinsics begin
    """
        iota(shape, T)

    Create a 1D tile with values [0, 1, 2, ..., shape[1]-1] (0-indexed).
    Compiled to cuda_tile.iota.
    """
    @noinline function iota(shape::NTuple{1, Int}, ::Type{T}) where {T}
        Tile{T, shape}()
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.iota), args)
    cb = ctx.cb
    tt = ctx.tt

    # Extract shape
    shape = get_constant(ctx, args[1])
    shape isa Tuple || error("iota() shape must be a compile-time constant tuple")
    tile_shape = collect(Int, shape)
    validate_tile_shape(tile_shape, "arange")

    # Extract dtype from Type{T} argument
    elem_type = @something get_constant(ctx, args[2]) error("iota() requires a compile-time element type")

    dtype = julia_to_tile_dtype!(tt, elem_type)
    tile_type = tile_type!(tt, dtype, tile_shape)

    # Emit IotaOp
    result = encode_IotaOp!(cb, tile_type)

    CGVal(result, tile_type, Tile{elem_type, Tuple(tile_shape)}, tile_shape)
end

# cuda_tile.mmaf, cuda_tile.mmai
@eval Intrinsics begin
    """
        mma(a, b, acc)

    Matrix-multiply-accumulate: result = a @ b + acc.
    Compiled to cuda_tile.mmaf or cuda_tile.mmai.
    """
    @noinline function mma(a::Tile{T1}, b::Tile{T2}, acc::Tile{T3, SC}) where {T1, T2, T3, SC}
        Tile{T3, SC}()
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.mma), args)
    cb = ctx.cb

    lhs = emit_value!(ctx, args[1])
    rhs = emit_value!(ctx, args[2])
    acc = emit_value!(ctx, args[3])

    (lhs === nothing || rhs === nothing || acc === nothing) && error("Cannot resolve operands for mma()")

    result = encode_MmaFOp!(cb, acc.type_id, lhs.v, rhs.v, acc.v)

    CGVal(result, acc.type_id, acc.jltype, acc.shape)
end

# TODO: cuda_tile.module

# cuda_tile.offset
@eval Intrinsics begin
    """
        offset(base, offsets)

    Compute base_ptr + offsets for each element of offsets tile (element-scaled).
    Returns a tile of pointers. Compiled to cuda_tile.offset.
    """
    @noinline function offset(base::Ptr{T}, offsets::Tile{I, S}) where {T, I <: Integer, S}
        Tile{Ptr{T}, S}()
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.offset), args)
    cb = ctx.cb
    tt = ctx.tt

    # Get base pointer (arg 1)
    base_ptr_tv = emit_value!(ctx, args[1])
    base_ptr_tv === nothing && error("offset: cannot resolve base pointer")
    base_ptr = base_ptr_tv.v

    # Get offsets tile (arg 2)
    offsets_tv = emit_value!(ctx, args[2])
    offsets_tv === nothing && error("offset: cannot resolve offsets tile")
    offsets = offsets_tv.v
    tile_shape = offsets_tv.shape

    # Get pointer element type from base pointer type (Ptr{T})
    base_ptr_type = unwrap_type(base_ptr_tv.jltype)
    ptr_elem_type = eltype(base_ptr_type)  # T from Ptr{T}
    elem_dtype = julia_to_tile_dtype!(tt, ptr_elem_type)
    ptr_dtype = pointer_type!(tt, elem_dtype)
    ptr_tile_type = tile_type!(tt, ptr_dtype, tile_shape)

    # Broadcast base pointer to tile shape
    ndims = length(tile_shape)
    if ndims > 0
        ones_shape = fill(1, ndims)
        reshaped_ptr_type = tile_type!(tt, ptr_dtype, ones_shape)
        base_ptr_reshaped = encode_ReshapeOp!(cb, reshaped_ptr_type, base_ptr)
        base_ptr_tile = encode_BroadcastOp!(cb, ptr_tile_type, base_ptr_reshaped)
    else
        base_ptr_tile = base_ptr
    end

    # Compute offset pointers: base_ptr + offsets (element offset)
    pointers = encode_OffsetOp!(cb, ptr_tile_type, base_ptr_tile, offsets)

    result_jltype = Tile{Ptr{ptr_elem_type}, Tuple(tile_shape)}
    CGVal(pointers, ptr_tile_type, result_jltype, tile_shape)
end

# TODO: cudatile.pack

# cuda_tile.permute
@eval Intrinsics begin
    """
        permute(tile, perm_val)

    Permute tile dimensions according to 0-indexed permutation.
    Compiled to cuda_tile.permute.
    """
    @noinline function permute(tile::Tile{T, S}, ::Val{Perm}) where {T, S, Perm}
        # Compute permuted shape: for each position i in output, take S[Perm[i]+1]
        permuted_shape = ntuple(i -> S[Perm[i] + 1], length(Perm))
        Tile{T, permuted_shape}()
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.permute), args)
    cb = ctx.cb
    tt = ctx.tt

    # Get source tile
    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve source operand for permute()")

    input_shape = source.shape
    isempty(input_shape) && error("Cannot determine tile shape for permute()")

    # Extract permutation from Val{Perm} argument
    perm_tuple = get_constant(ctx, args[2])
    perm_tuple isa Tuple || error("permute() permutation must be a compile-time constant tuple")

    # Convert to 0-indexed vector for bytecode
    permutation = collect(Int, perm_tuple)

    # Compute output shape based on permutation
    # permutation[i] tells us which input dimension goes to output position i
    output_shape = [input_shape[p + 1] for p in permutation]

    # Get element type
    elem_type = unwrap_type(source.jltype).parameters[1]

    # Create output tile type
    dtype = julia_to_tile_dtype!(tt, elem_type)
    output_tile_type = tile_type!(tt, dtype, output_shape)

    # Emit PermuteOp
    result = encode_PermuteOp!(cb, output_tile_type, source.v, permutation)

    CGVal(result, output_tile_type, Tile{elem_type, Tuple(output_shape)}, output_shape)
end

# cuda_tile.transpose
@eval Intrinsics begin
    """
        transpose(tile)

    Transpose a 2D tile, swapping its dimensions.
    Compiled to cuda_tile.permute with perm=(1, 0).
    """
    @noinline function transpose(tile::Tile{T, S}) where {T, S}
        Tile{T, reverse(S)}()
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.transpose), args)
    cb = ctx.cb
    tt = ctx.tt

    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve operand for transpose()")

    input_shape = source.shape
    isempty(input_shape) && error("Cannot determine tile shape for transpose()")

    output_shape = reverse(input_shape)

    elem_type = unwrap_type(source.jltype).parameters[1]

    dtype = julia_to_tile_dtype!(tt, elem_type)
    output_tile_type = tile_type!(tt, dtype, output_shape)

    ndim = length(output_shape)
    permutation = collect(ndim-1:-1:0)

    result = encode_PermuteOp!(cb, output_tile_type, source.v, permutation)

    CGVal(result, output_tile_type, Tile{elem_type, Tuple(output_shape)}, output_shape)
end


## XXX: cuda_tile.reduce
@eval Intrinsics begin
    """
        reduce_sum(tile, axis_val)

    Sum reduction along 0-indexed axis.
    Compiled to cuda_tile.reduce with ADD.
    """
    @noinline function reduce_sum(tile::Tile{T, S}, ::Val{axis}) where {T, S, axis}
        reduced_shape = ntuple(i -> S[i < axis + 1 ? i : i + 1], length(S) - 1)
        Tile{T, reduced_shape}()
    end

    """
        reduce_max(tile, axis_val)

    Maximum reduction along 0-indexed axis.
    Compiled to cuda_tile.reduce with MAX.
    """
    @noinline function reduce_max(tile::Tile{T, S}, ::Val{axis}) where {T, S, axis}
        reduced_shape = ntuple(i -> S[i < axis + 1 ? i : i + 1], length(S) - 1)
        Tile{T, reduced_shape}()
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.reduce_sum), args)
    emit_reduce!(ctx, args, :add)
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.reduce_max), args)
    emit_reduce!(ctx, args, :max)
end
function emit_reduce!(ctx::CGCtx, args, reduce_fn::Symbol)
    cb = ctx.cb
    tt = ctx.tt

    # Get input tile
    input_tv = emit_value!(ctx, args[1])
    input_tv === nothing && error("Cannot resolve input tile for reduction")

    # Get reduction axis
    axis = @something get_constant(ctx, args[2]) error("Reduction axis must be a compile-time constant")

    # Get element type and shapes
    input_type = unwrap_type(input_tv.jltype)
    elem_type = input_type.parameters[1]
    input_shape = input_tv.shape
    isempty(input_shape) && error("Cannot reduce scalar tile")

    # Compute output shape (dimension at axis is removed)
    output_shape = Int[input_shape[i] for i in eachindex(input_shape) if i != axis + 1]

    dtype = julia_to_tile_dtype!(tt, elem_type)

    # Output tile type
    output_tile_type = tile_type!(tt, dtype, output_shape)

    # Scalar type for reduction body (0D tile)
    scalar_tile_type = tile_type!(tt, dtype, Int[])

    # Create identity value via dispatch on reduction function and element type
    identity = operation_identity(Val(reduce_fn), dtype, elem_type)

    # Emit ReduceOp
    results = encode_ReduceOp!(cb, [output_tile_type], [input_tv.v], axis, [identity], [scalar_tile_type]) do block_args
        acc, elem = block_args[1], block_args[2]

        res = encode_reduce_body(cb, scalar_tile_type, acc, elem, reduce_fn, elem_type)
        encode_YieldOp!(cb, [res])
    end

    CGVal(results[1], output_tile_type, Tile{elem_type, Tuple(output_shape)}, output_shape)
end

#============================================================================
 Reduce Identity Values via Dispatch
============================================================================#

"""
    operation_identity(fn, dtype, elem_type) -> IdentityVal
    to_uint128(value)

Convert an integer value to UInt128 for storage in IntegerIdentityVal.
For signed types, this returns the two's complement bit representation.
"""
# Unsigned types: directly convert
to_uint128(value::UInt64) = UInt128(value)
to_uint128(value::UInt32) = UInt128(value)
to_uint128(value::UInt16) = UInt128(value)
to_uint128(value::UInt8) = UInt128(value)
# Signed types: reinterpret as unsigned first, then convert
to_uint128(value::Int64) = UInt128(reinterpret(UInt64, value))
to_uint128(value::Int32) = UInt128(reinterpret(UInt32, value))
to_uint128(value::Int16) = UInt128(reinterpret(UInt16, value))
to_uint128(value::Int8) = UInt128(reinterpret(UInt8, value))

"""
    operation_identity(fn, dtype, elem_type) -> IdentityVal

Return the identity value for a binary operation (reduce, scan, etc.).
Identity must satisfy: identity ⊕ x = x for the operation.
"""

# Addition identity: 0 + x = x
operation_identity(::Val{:add}, dtype, ::Type{T}) where T <: AbstractFloat =
    FloatIdentityVal(zero(T), dtype, T)
operation_identity(::Val{:add}, dtype, ::Type{T}) where T <: Integer =
    IntegerIdentityVal(to_uint128(zero(T)), dtype, T)

# Maximum identity: max(typemin(T), x) = x
operation_identity(::Val{:max}, dtype, ::Type{T}) where T <: AbstractFloat =
    FloatIdentityVal(typemin(T), dtype, T)
operation_identity(::Val{:max}, dtype, ::Type{T}) where T <: Integer =
    IntegerIdentityVal(to_uint128(typemin(T)), dtype, T)

# Multiplication identity: 1 * x = x
operation_identity(::Val{:mul}, dtype, ::Type{T}) where T <: AbstractFloat =
    FloatIdentityVal(one(T), dtype, T)
operation_identity(::Val{:mul}, dtype, ::Type{T}) where T <: Integer =
    IntegerIdentityVal(to_uint128(one(T)), dtype, T)

# Minimum identity: min(typemax(T), x) = x
operation_identity(::Val{:min}, dtype, ::Type{T}) where T <: AbstractFloat =
    FloatIdentityVal(typemax(T), dtype, T)
operation_identity(::Val{:min}, dtype, ::Type{T}) where T <: Integer =
    IntegerIdentityVal(to_uint128(typemax(T)), dtype, T)

#============================================================================
 Reduce Body Operations
============================================================================#
function encode_reduce_body(cb, type, acc, elem, op::Symbol, ::Type{T}) where T
    if T <: AbstractFloat
        if op == :add
            encode_AddFOp!(cb, type, acc, elem)
        elseif op == :max
            encode_MaxFOp!(cb, type, acc, elem)
        end
    else  # Integer
        signedness = T <: Signed ? SignednessSigned : SignednessUnsigned
        if op == :add
            encode_AddIOp!(cb, type, acc, elem)
        elseif op == :max
            encode_MaxIOp!(cb, type, acc, elem; signedness)
        end
    end
end


# cuda_tile.reshape
@eval Intrinsics begin
    """
        reshape(tile, shape_val)

    Reshape a tile to a new shape (same total elements).
    Compiled to cuda_tile.reshape.
    """
    @noinline function reshape(tile::Tile{T}, ::Val{Shape}) where {T, Shape}
        Tile{T, Shape}()
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.reshape), args)
    cb = ctx.cb
    tt = ctx.tt

    # Get source tile
    source = emit_value!(ctx, args[1])
    source === nothing && error("Cannot resolve source operand for reshape()")

    # Extract target shape from Val{Shape} argument
    target_shape_tuple = get_constant(ctx, args[2])
    target_shape_tuple isa Tuple || error("reshape() shape must be a compile-time constant tuple")
    target_shape = collect(Int, target_shape_tuple)
    validate_tile_shape(target_shape, "reshape")

    # Get element type and source shape
    source_type = unwrap_type(source.jltype)
    elem_type = source_type.parameters[1]
    source_shape = collect(Int, source_type.parameters[2])

    dtype = julia_to_tile_dtype!(tt, elem_type)

    # Tile IR's ReshapeOp uses row-major element ordering, but Julia uses column-major.
    # To achieve Julia's column-major reshape semantics, we need to:
    # 1. Permute source to row-major order (reverse dims) if ndim > 1
    # 2. Reshape with reversed target shape
    # 3. Permute result back to column-major order (reverse dims) if ndim > 1

    current_val = source.v
    current_shape = source_shape

    # Step 1: Permute source if >1 dimension (column-major → row-major)
    if length(current_shape) > 1
        perm = collect(length(current_shape)-1:-1:0)  # 0-indexed reverse
        permuted_shape = reverse(current_shape)
        perm_type_id = tile_type!(tt, dtype, permuted_shape)
        current_val = encode_PermuteOp!(cb, perm_type_id, current_val, perm)
        current_shape = permuted_shape
    end

    # Step 2: ReshapeOp with reversed target shape
    reversed_target = reverse(target_shape)
    reshape_type_id = tile_type!(tt, dtype, reversed_target)
    current_val = encode_ReshapeOp!(cb, reshape_type_id, current_val)
    current_shape = reversed_target

    # Step 3: Permute result back if >1 dimension (row-major → column-major)
    if length(target_shape) > 1
        perm = collect(length(target_shape)-1:-1:0)  # 0-indexed reverse
        result_type_id = tile_type!(tt, dtype, target_shape)
        current_val = encode_PermuteOp!(cb, result_type_id, current_val, perm)
    else
        result_type_id = tile_type!(tt, dtype, target_shape)
    end

    CGVal(current_val, result_type_id, Tile{elem_type, Tuple(target_shape)}, target_shape)
end

# TODO: cuda_tile.scan

# cuda_tile.select
@eval Intrinsics begin
    """
        select(cond, x, y)

    Element-wise conditional selection.
    Compiled to cuda_tile.select.
    """
    @noinline function select(cond::Tile{Bool, S}, x::Tile{T, S}, y::Tile{T, S}) where {T, S}
        Tile{T, S}()
    end
end
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.select), args)
    cb = ctx.cb

    cond_tv = emit_value!(ctx, args[1])
    x_tv = emit_value!(ctx, args[2])
    y_tv = emit_value!(ctx, args[3])

    (cond_tv === nothing || x_tv === nothing || y_tv === nothing) &&
        error("Cannot resolve operands for select()")

    result = encode_SelectOp!(cb, x_tv.type_id, cond_tv.v, x_tv.v, y_tv.v)

    CGVal(result, x_tv.type_id, x_tv.jltype, x_tv.shape)
end

# TODO: cuda_tile.unpack

#=============================================================================
 MapReduce operations
=============================================================================#

"""
    emit_mapreduce!(ctx, args) -> CGVal

Emit bytecode for mapreduce operation.
Args: [f, op, tile, axis]

The mapreduce operation applies f to each element, then reduces with op.
"""
function emit_mapreduce!(ctx::CGCtx, args::Vector{Any})
    cb = ctx.cb
    tt = ctx.tt

    # Get arguments
    f_arg = args[1]
    op_arg = args[2]
    tile_tv = emit_value!(ctx, args[3])
    tile_tv === nothing && error("Cannot resolve input tile for mapreduce")

    # Get axis from Val argument
    axis_val = @something get_constant(ctx, args[4]) error("Reduction axis must be a compile-time constant")
    axis = axis_val

    # Get element type and shapes
    input_type = unwrap_type(tile_tv.jltype)
    elem_type = input_type.parameters[1]
    input_shape = tile_tv.shape
    isempty(input_shape) && error("Cannot reduce scalar tile with mapreduce")

    # Compute output shape (dimension at axis is removed)
    output_shape = Int[input_shape[i] for i in eachindex(input_shape) if i != axis + 1]

    # Create types
    dtype = julia_to_tile_dtype!(tt, elem_type)
    output_tile_type = tile_type!(tt, dtype, output_shape)
    scalar_tile_type = tile_type!(tt, dtype, Int[])

    # Determine reduction function and identity
    reduce_fn, identity = determine_reduction_fn(op_arg, dtype, elem_type)

    # Encode the reduction with custom body
    results = encode_ReduceOp!(cb, [output_tile_type], [tile_tv.v], axis, [identity], [scalar_tile_type]) do block_args
        acc, elem = block_args[1], block_args[2]

        # Body: result = op(acc, f(elem))
        # Step 1: Apply map function f to element
        mapped = emit_map_body!(ctx, f_arg, elem, scalar_tile_type, elem_type)

        # Step 2: Apply reduction function op to (acc, mapped)
        result = emit_reduce_body!(ctx, op_arg, acc, mapped, scalar_tile_type, elem_type)

        encode_YieldOp!(cb, [result])
    end

    CGVal(results[1], output_tile_type, Tile{elem_type, Tuple(output_shape)}, output_shape)
end

"""
    determine_reduction_fn(op_arg, dtype, elem_type) -> (Symbol, IdentityVal)

Identify the reduction function from the op argument.
Returns (function_name, identity_value).
"""
function determine_reduction_fn(@nospecialize(op_arg), dtype, ::Type{T}) where T
    # Try to extract the function from GlobalRef or direct value
    if op_arg isa GlobalRef
        name = op_arg.name
        if name === :+
            return :add, operation_identity(Val(:add), dtype, T)
        elseif name === :*
            return :mul, operation_identity(Val(:mul), dtype, T)
        elseif name === :max
            return :max, operation_identity(Val(:max), dtype, T)
        elseif name === :min
            return :min, operation_identity(Val(:min), dtype, T)
        end
    elseif op_arg isa Function
        # Direct function value
        if op_arg === (+)
            return :add, operation_identity(Val(:add), dtype, T)
        elseif op_arg === (*)
            return :mul, operation_identity(Val(:mul), dtype, T)
        elseif op_arg === max
            return :max, operation_identity(Val(:max), dtype, T)
        elseif op_arg === min
            return :min, operation_identity(Val(:min), dtype, T)
        end
    elseif op_arg isa Core.Builtin
        if op_arg === Core.add
            return :add, operation_identity(Val(:add), dtype, T)
        elseif op_arg === Core.mul
            return :mul, operation_identity(Val(:mul), dtype, T)
        end
    end

    error("Unsupported reduction function. Supported: +, *, max, min")
end

"""
    extract_function(ctx, arg) -> Union{Function, Nothing}

Extract a Function value from various IR forms.
"""
function extract_function(ctx::CGCtx, @nospecialize(arg))
    if arg isa Core.Builtin
        # Handle built-in functions
        if arg === Core.add
            return (+)
        elseif arg === Core.mul
            return (*)
        end
        return nothing
    elseif arg isa GlobalRef
        # Try to evaluate the global reference
        try
            return Core.eval(arg.mod, arg.name)
        catch
            return nothing
        end
    elseif arg isa SSAValue
        # Look up in context values
        tv = get(ctx.values, arg.id, nothing)
        if tv !== nothing && tv.constant !== nothing
            return something(tv.constant)
        end
        return nothing
    end
    return nothing
end

"""
    emit_map_body!(ctx, f_arg, elem, scalar_type, elem_type) -> Value

Emit bytecode for the map function f applied to element.
Supports decomposition of composite expressions into sequences of Tile IR ops.
"""
function emit_map_body!(ctx::CGCtx, @nospecialize(f_arg), elem::Value, scalar_type::TypeId, ::Type{T}) where T
    cb = ctx.cb
    tt = ctx.tt

    # First, try direct single-op functions
    direct_fn = extract_direct_function(ctx, f_arg)
    if direct_fn !== nothing
        if direct_fn === :identity
            return elem
        elseif direct_fn === :abs
            return encode_AbsFOp!(cb, scalar_type, elem)
        elseif direct_fn === :sqrt
            return encode_SqrtOp!(cb, scalar_type, elem)
        elseif direct_fn === :exp
            return encode_ExpOp!(cb, scalar_type, elem)
        elseif direct_fn === :log
            return encode_LogOp!(cb, scalar_type, elem)
        elseif direct_fn === :sin
            return encode_SinOp!(cb, scalar_type, elem)
        elseif direct_fn === :cos
            return encode_CosOp!(cb, scalar_type, elem)
        elseif direct_fn === :neg
            return encode_NegFOp!(cb, scalar_type, elem)
        elseif direct_fn === :abs2
            return encode_MulFOp!(cb, scalar_type, elem, elem)
        end
    end

    # Try to decompose composite expressions
    decomposed = try_decompose_expr(ctx, f_arg, elem, scalar_type, T)
    if decomposed !== nothing
        return decomposed
    end

    error("Unsupported map function. Supported: identity, abs, abs2, sqrt, exp, log, sin, cos, neg, and decomposable expressions like x+c, x*c, (x+c)*d, sin(x)+1, etc.")
end

"""
    extract_direct_function(ctx, f_arg) -> Union{Symbol, Nothing}

Extract a direct function name from f_arg.
"""
function extract_direct_function(ctx::CGCtx, @nospecialize(f_arg))
    # Check GlobalRef for common functions
    if f_arg isa GlobalRef
        name = f_arg.name
        if name === :identity || name === :abs || name === :sqrt ||
           name === :exp || name === :log || name === :sin || name === :cos ||
           name === :neg || name === :abs2
            return name
        end
    end

    # Try to evaluate the function
    fn = extract_function(ctx, f_arg)
    if fn !== nothing
        if fn === identity
            return :identity
        elseif fn === abs
            return :abs
        elseif fn === sqrt
            return :sqrt
        elseif fn === exp
            return :exp
        elseif fn === log
            return :log
        elseif fn === sin
            return :sin
        elseif fn === cos
            return :cos
        elseif fn === abs
            return :abs
        elseif fn === Base.:-
            return :neg
        end
    end

    return nothing
end

"""
    try_decompose_expr(ctx, f_arg, elem, scalar_type, elem_type) -> Union{Value, Nothing}

Try to decompose a composite expression into Tile IR operations.
Returns the final value or nothing if decomposition fails.
"""
function try_decompose_expr(ctx::CGCtx, @nospecialize(f_arg), elem::Value, scalar_type::TypeId, ::Type{T}) where T
    cb = ctx.cb

    # Try to analyze the function
    if f_arg isa Expr && f_arg.head === :call
        fn_name = f_arg.args[1]
        args = f_arg.args[2:end]

        # Case: x + c (addition with constant)
        if fn_name === :+ && length(args) == 2
            # Check if one argument is a constant
            if is_constant_arg(args[1])
                constant = get_constant_value(args[1])
                const_val = encode_constant!(ctx, constant, scalar_type, T)
                return encode_AddFOp!(cb, scalar_type, elem, const_val)
            elseif is_constant_arg(args[2])
                constant = get_constant_value(args[2])
                const_val = encode_constant!(ctx, constant, scalar_type, T)
                return encode_AddFOp!(cb, scalar_type, elem, const_val)
            end
        end

        # Case: x - c (subtraction with constant)
        if fn_name === :- && length(args) == 2
            if is_constant_arg(args[1])
                # c - x
                constant = get_constant_value(args[1])
                const_val = encode_constant!(ctx, constant, scalar_type, T)
                return encode_SubFOp!(cb, scalar_type, const_val, elem)
            elseif is_constant_arg(args[2])
                # x - c
                constant = get_constant_value(args[2])
                const_val = encode_constant!(ctx, constant, scalar_type, T)
                return encode_SubFOp!(cb, scalar_type, elem, const_val)
            end
        end

        # Case: x * c (multiplication with constant)
        if fn_name === :* && length(args) == 2
            if is_constant_arg(args[1])
                constant = get_constant_value(args[1])
                const_val = encode_constant!(ctx, constant, scalar_type, T)
                return encode_MulFOp!(cb, scalar_type, const_val, elem)
            elseif is_constant_arg(args[2])
                constant = get_constant_value(args[2])
                const_val = encode_constant!(ctx, constant, scalar_type, T)
                return encode_MulFOp!(cb, scalar_type, elem, const_val)
            end
        end

        # Case: x / c (division with constant)
        if fn_name === :/ && length(args) == 2
            if is_constant_arg(args[2])
                constant = get_constant_value(args[2])
                const_val = encode_constant!(ctx, constant, scalar_type, T)
                return encode_DivFOp!(cb, scalar_type, elem, const_val)
            end
        end

        # Case: abs(x) - handled by direct, but for completeness
        if fn_name === :abs && length(args) == 1
            return encode_AbsFOp!(cb, scalar_type, elem)
        end

        # Case: sqrt(x) - handled by direct
        if fn_name === :sqrt && length(args) == 1
            return encode_SqrtOp!(cb, scalar_type, elem)
        end

        # Case: x^2 (power of 2)
        if fn_name === :^ && length(args) == 3 && args[3] == 2
            return encode_MulFOp!(cb, scalar_type, elem, elem)
        end

        # Case: composite: f(g(x)) where both are decomposable
        if length(args) == 1
            inner = args[1]
            if inner isa Expr && inner.head === :call
                # Try to decompose inner first, then apply outer
                inner_result = try_decompose_expr(ctx, inner, elem, scalar_type, T)
                if inner_result !== nothing
                    # Now apply outer function
                    return apply_function(ctx, fn_name, inner_result, scalar_type, T)
                end
            end
        end
    end

    # Try to handle more complex expressions
    return try_complex_expr_decomposition(ctx, f_arg, elem, scalar_type, T)
end

"""
    apply_function(ctx, fn_name, value, scalar_type, elem_type) -> Value

Apply a single-argument function to a value.
"""
function apply_function(ctx::CGCtx, fn_name, value::Value, scalar_type::TypeId, ::Type{T}) where T
    cb = ctx.cb

    if fn_name === :abs
        return encode_AbsFOp!(cb, scalar_type, value)
    elseif fn_name === :sqrt
        return encode_SqrtOp!(cb, scalar_type, value)
    elseif fn_name === :exp
        return encode_ExpOp!(cb, scalar_type, value)
    elseif fn_name === :log
        return encode_LogOp!(cb, scalar_type, value)
    elseif fn_name === :sin
        return encode_SinOp!(cb, scalar_type, value)
    elseif fn_name === :cos
        return encode_CosOp!(cb, scalar_type, value)
    elseif fn_name === :neg || fn_name === :-
        return encode_NegFOp!(cb, scalar_type, value)
    else
        error("Unsupported function: $fn_name")
    end
end

"""
    is_constant_arg(arg) -> Bool

Check if arg is a constant literal.
"""
function is_constant_arg(@nospecialize(arg))
    if arg isa Number
        return true
    elseif arg isa Expr
        # Check for literal expressions like :(1.0)
        return arg.head === :lit
    end
    return false
end

"""
    get_constant_value(arg) -> Number

Extract the numeric value from a constant argument.
"""
function get_constant_value(@nospecialize(arg))
    if arg isa Number
        return arg
    elseif arg isa Expr && arg.head === :lit
        return arg.args[1]
    end
    error("Not a constant: $arg")
end

"""
    encode_constant!(ctx, value, scalar_type, elem_type) -> Value

Encode a constant value as a Tile IR constant operation.
"""
function encode_constant!(ctx::CGCtx, value, scalar_type::TypeId, ::Type{T}) where T
    cb = ctx.cb

    # Convert to the appropriate type
    typed_value = T(value)

    # Encode as bytes
    bytes = reinterpret(UInt8, [typed_value])

    # Use ConstantOp to create the constant
    return encode_ConstantOp!(ctx.cb, scalar_type, collect(bytes))
end

"""
    try_complex_expr_decomposition(ctx, f_arg, elem, scalar_type, elem_type) -> Union{Value, Nothing}

Handle more complex expressions that require multiple operations.
Examples: x^2 + 1, sin(x) + cos(x), (x + 1) * 2, etc.
"""
function try_complex_expr_decomposition(ctx::CGCtx, @nospecialize(f_arg), elem::Value, scalar_type::TypeId, ::Type{T}) where T
    cb = ctx.cb

    if f_arg isa Expr
        # Pattern: x^2 (special case of x*x)
        if f_arg.head === :call && f_arg.args[1] === :^ && length(f_arg.args) == 3
            if f_arg.args[3] == 2
                return encode_MulFOp!(cb, scalar_type, elem, elem)
            end
        end

        # Pattern: f(x) op c where op is +, -, *, /
        if f_arg.head === :call && length(f_arg.args) == 3
            return try_binary_expr_decomposition(ctx, f_arg, elem, scalar_type, T)
        end

        # Pattern: f(g(x)) - nested function calls
        if f_arg.head === :call && length(f_arg.args) == 2
            inner_fn = f_arg.args[1]
            inner_arg = f_arg.args[2]

            if inner_arg isa Symbol || (inner_arg isa Expr && inner_arg.head === :call)
                # Recursively decompose
                inner_result = try_decompose_expr(ctx, inner_arg, elem, scalar_type, T)
                if inner_result !== nothing
                    return apply_function(ctx, inner_fn, inner_result, scalar_type, T)
                end
            end
        end

        # Pattern: f(x) + g(x) (binary operation of two function applications)
        if f_arg.head === :call && length(f_arg.args) == 3
            left = f_arg.args[1]
            op = f_arg.args[2]
            right = f_arg.args[3]

            # Check if it's a composite expression like sin(x) + cos(x)
            if op isa Symbol && (op === :+ || op === :- || op === :* || op === :/)
                # Try to decompose both sides
                left_val = try_decompose_expr(ctx, left, elem, scalar_type, T)
                right_val = try_decompose_expr(ctx, right, elem, scalar_type, T)

                if left_val !== nothing && right_val !== nothing
                    if op === :+
                        return encode_AddFOp!(cb, scalar_type, left_val, right_val)
                    elseif op === :-
                        return encode_SubFOp!(cb, scalar_type, left_val, right_val)
                    elseif op === :*
                        return encode_MulFOp!(cb, scalar_type, left_val, right_val)
                    elseif op === :/
                        return encode_DivFOp!(cb, scalar_type, left_val, right_val)
                    end
                end
            end
        end
    end

    return nothing
end

"""
    try_binary_expr_decomposition(ctx, f_arg, elem, scalar_type, elem_type) -> Union{Value, Nothing}

Decompose binary expressions like (x + c) or (c * x).
"""
function try_binary_expr_decomposition(ctx::CGCtx, @nospecialize(f_arg), elem::Value, scalar_type::TypeId, ::Type{T}) where T
    cb = ctx.cb

    left = f_arg.args[1]
    op = f_arg.args[2]
    right = f_arg.args[3]

    # Case 1: (x op c) where op is +, -, *, /
    if left isa Symbol  # x is a variable
        if is_constant_arg(right)
            constant = get_constant_value(right)
            const_val = encode_constant!(ctx, constant, scalar_type, T)

            if op === :+
                return encode_AddFOp!(cb, scalar_type, elem, const_val)
            elseif op === :-
                return encode_SubFOp!(cb, scalar_type, elem, const_val)
            elseif op === :*
                return encode_MulFOp!(cb, scalar_type, elem, const_val)
            elseif op === :/
                return encode_DivFOp!(cb, scalar_type, elem, const_val)
            end
        end
    end

    # Case 2: (c op x) where x is variable
    if right isa Symbol  # x is a variable
        if is_constant_arg(left)
            constant = get_constant_value(left)
            const_val = encode_constant!(ctx, constant, scalar_type, T)

            if op === :+
                return encode_AddFOp!(cb, scalar_type, const_val, elem)
            elseif op === :*
                return encode_MulFOp!(cb, scalar_type, const_val, elem)
            elseif op === :-
                return encode_SubFOp!(cb, scalar_type, const_val, elem)
            elseif op === :/
                return encode_DivFOp!(cb, scalar_type, const_val, elem)
            end
        end
    end

    # Case 3: (f(x) op c) where f(x) needs decomposition
    left_decomposed = try_decompose_expr(ctx, left, elem, scalar_type, T)
    if left_decomposed !== nothing && is_constant_arg(right)
        constant = get_constant_value(right)
        const_val = encode_constant!(ctx, constant, scalar_type, T)

        if op === :+
            return encode_AddFOp!(cb, scalar_type, left_decomposed, const_val)
        elseif op === :-
            return encode_SubFOp!(cb, scalar_type, left_decomposed, const_val)
        elseif op === :*
            return encode_MulFOp!(cb, scalar_type, left_decomposed, const_val)
        elseif op === :/
            return encode_DivFOp!(cb, scalar_type, left_decomposed, const_val)
        end
    end

    # Case 4: (c op f(x))
    right_decomposed = try_decompose_expr(ctx, right, elem, scalar_type, T)
    if right_decomposed !== nothing && is_constant_arg(left)
        constant = get_constant_value(left)
        const_val = encode_constant!(ctx, constant, scalar_type, T)

        if op === :+
            return encode_AddFOp!(cb, scalar_type, const_val, right_decomposed)
        elseif op === :-
            return encode_SubFOp!(cb, scalar_type, const_val, right_decomposed)
        elseif op === :*
            return encode_MulFOp!(cb, scalar_type, const_val, right_decomposed)
        elseif op === :/
            return encode_DivFOp!(cb, scalar_type, const_val, right_decomposed)
        end
    end

    return nothing
end

"""
    emit_reduce_body!(ctx, op_arg, acc, elem, scalar_type, elem_type) -> Value

Emit bytecode for the reduction function op applied to (acc, elem).
"""
function emit_reduce_body!(ctx::CGCtx, @nospecialize(op_arg), acc::Value, elem::Value, scalar_type::TypeId, ::Type{T}) where T
    cb = ctx.cb

    # Try to identify the reduction function
    fn = extract_function(ctx, op_arg)

    # Handle known functions
    if fn !== nothing
        if fn === (+)
            return encode_AddFOp!(cb, scalar_type, acc, elem)
        elseif fn === (*)
            return encode_MulFOp!(cb, scalar_type, acc, elem)
        elseif fn === max
            return encode_MaxFOp!(cb, scalar_type, acc, elem)
        elseif fn === min
            return encode_MinFOp!(cb, scalar_type, acc, elem)
        end
    end

    # Check GlobalRef for common functions
    if op_arg isa GlobalRef
        name = op_arg.name
        if name === :+
            return encode_AddFOp!(cb, scalar_type, acc, elem)
        elseif name === :*
            return encode_MulFOp!(cb, scalar_type, acc, elem)
        elseif name === :max
            return encode_MaxFOp!(cb, scalar_type, acc, elem)
        elseif name === :min
            return encode_MinFOp!(cb, scalar_type, acc, elem)
        end
    end

    error("Unsupported reduction function. Supported: +, *, max, min")
end

# Define mapreduce intrinsic
@eval Intrinsics begin
    """
        mapreduce(f, op, tile, axis_val)

    Apply f to each element, then reduce with op.
    Axis is 0-indexed (converted from 1-indexed in public API).
    """
    @noinline function mapreduce(f::Function, op::Function, tile::Tile{T, S}, ::Val{axis}) where {T, S, axis}
        reduced_shape = ntuple(i -> S[i < axis + 1 ? i : i + 1], length(S) - 1)
        Tile{T, reduced_shape}()
    end
end

function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.mapreduce), args)
    emit_mapreduce!(ctx, args)
end
