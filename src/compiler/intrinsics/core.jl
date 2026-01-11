# core Tile IR intrinsics


## cuda_tile.broadcast

@eval Intrinsics begin
    """
        broadcast(tile, shape_val)

    Explicitly broadcast a tile to a target shape.
    Compiled to cuda_tile.broadcast.
    """
    @noinline function broadcast(tile::Tile{T, S}, ::Val{Shape}) where {T, S, Shape}
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
    source_elem = source_type <: Tile ? source_type.parameters[1] : source_type

    # Extract target shape from the constant tuple argument
    target_shape_tuple = get_constant(ctx, args[2])
    target_shape_tuple isa Tuple || error("broadcast() shape must be a compile-time constant tuple")
    target_shape = collect(Int, target_shape_tuple)

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



## cuda_tile.cat

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

    # Get element type
    elem_type = unwrap_type(lhs.jltype)
    if elem_type <: Tile
        elem_type = elem_type.parameters[1]
    end

    # Create output tile type
    dtype = julia_to_tile_dtype!(tt, elem_type)
    output_tile_type = tile_type!(tt, dtype, output_shape)

    # Emit CatOp (axis is 0-indexed for bytecode)
    result = encode_CatOp!(cb, output_tile_type, lhs.v, rhs.v, axis)

    CGVal(result, output_tile_type, Tile{elem_type, Tuple(output_shape)}, output_shape)
end


## cuda_tile.constant

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

    # Extract value
    value = @something get_constant(ctx, args[2]) error("full() value must be a compile-time constant")

    # Extract dtype from Type{T} argument
    elem_type = @something get_constant(ctx, args[3]) error("constant() requires a compile-time element type")

    dtype = julia_to_tile_dtype!(tt, elem_type)
    tile_type = tile_type!(tt, dtype, tile_shape)

    # Create scalar constant
    scalar_type = tile_type!(tt, dtype, Int[])
    value_bytes = constant_to_bytes(value, elem_type)
    scalar_val = encode_ConstantOp!(cb, scalar_type, value_bytes)

    # Reshape and broadcast
    ndims = length(tile_shape)
    if ndims > 0
        ones_shape = fill(1, ndims)
        reshaped_type = tile_type!(tt, dtype, ones_shape)
        reshaped_val = encode_ReshapeOp!(cb, reshaped_type, scalar_val)
    else
        reshaped_val = scalar_val
    end

    result = encode_BroadcastOp!(cb, tile_type, reshaped_val)

    CGVal(result, tile_type, Tile{elem_type, Tuple(tile_shape)}, tile_shape)
end


## TODO: cuda_tile.entry


## cuda_tile.extract

@eval Intrinsics begin
    """
        extract(tile, index_val, shape_val)

    Extract a sub-tile from tile at 0-indexed slice indices.
    Compiled to cuda_tile.extract.
    """
    @noinline function extract(tile::Tile{T, S}, ::Val{Index}, ::Val{Shape}) where {T, S, Index, Shape}
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

    # Get element type
    elem_type = unwrap_type(source.jltype)
    if elem_type <: Tile
        elem_type = elem_type.parameters[1]
    end

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


## TODO: cuda_tile.get_global


## cuda_tile.get_num_tile_blocks

@eval Intrinsics begin
    """
        get_num_tile_blocks(axis)::Int32

    Get the grid size along the given axis (0=x, 1=y, 2=z).
    Compiled to cuda_tile.get_num_tile_blocks.
    """
    @noinline get_num_tile_blocks(axis::Integer) = Base.compilerbarrier(:const, zero(Int32))
end

function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.get_num_tile_blocks), args)
    axis = @something get_constant(ctx, args[1]) error("get_num_tile_blocks() axis must be a compile-time constant")
    axis in (0, 1, 2) || error("get_num_tile_blocks() axis must be 0, 1, or 2, got $axis")

    res_type = tile_type!(ctx.tt, I32(ctx.tt), Int[])
    nb_x, nb_y, nb_z = encode_GetNumTileBlocksOp!(ctx.cb, res_type, res_type, res_type)

    CGVal((nb_x, nb_y, nb_z)[axis + 1], res_type, Int32)
end


## cuda_tile.get_tile_block_id

@eval Intrinsics begin
    """
        get_tile_block_id(axis)::Int32

    Get the block ID along the given axis (0=x, 1=y, 2=z).
    Compiled to cuda_tile.get_tile_block_id.
    """
    @noinline get_tile_block_id(axis::Integer) = Base.compilerbarrier(:const, zero(Int32))
end

function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.get_tile_block_id), args)
    axis = @something get_constant(ctx, args[1]) error("get_tile_block_id() axis must be a compile-time constant")
    axis in (0, 1, 2) || error("get_tile_block_id() axis must be 0, 1, or 2, got $axis")

    res_type = tile_type!(ctx.tt, I32(ctx.tt), Int[])
    bid_x, bid_y, bid_z = encode_GetTileBlockIdOp!(ctx.cb, res_type, res_type, res_type)
    result = (bid_x, bid_y, bid_z)[axis + 1]

    CGVal(result, res_type, Int32)
end


## TODO: cuda_tile.global


## cuda_tile.iota

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

    # Extract dtype from Type{T} argument
    elem_type = @something get_constant(ctx, args[2]) error("iota() requires a compile-time element type")

    dtype = julia_to_tile_dtype!(tt, elem_type)
    tile_type = tile_type!(tt, dtype, tile_shape)

    # Emit IotaOp
    result = encode_IotaOp!(cb, tile_type)

    CGVal(result, tile_type, Tile{elem_type, Tuple(tile_shape)}, tile_shape)
end


## cuda_tile.mmaf, cuda_tile.mmai

@eval Intrinsics begin
    """
        mma(a, b, acc)

    Matrix-multiply-accumulate: result = a @ b + acc.
    Compiled to cuda_tile.mmaf or cuda_tile.mmai.
    """
    @noinline function mma(a::Tile{T1, SA}, b::Tile{T2, SB}, acc::Tile{T3, SC}) where {T1, T2, T3, SA, SB, SC}
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


## TODO: cuda_tile.module


## cuda_tile.offset

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


## TODO: cudatile.pack


## cuda_tile.permute

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
    elem_type = unwrap_type(source.jltype)
    if elem_type <: Tile
        elem_type = elem_type.parameters[1]
    end

    # Create output tile type
    dtype = julia_to_tile_dtype!(tt, elem_type)
    output_tile_type = tile_type!(tt, dtype, output_shape)

    # Emit PermuteOp
    result = encode_PermuteOp!(cb, output_tile_type, source.v, permutation)

    CGVal(result, output_tile_type, Tile{elem_type, Tuple(output_shape)}, output_shape)
end


## XXX: cuda_tile.transpose

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

    elem_type = unwrap_type(source.jltype)
    if elem_type <: Tile
        elem_type = elem_type.parameters[1]
    end

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
    @noinline function reduce_sum(tile::Tile{T, S}, ::Val{axis}) where {T <: AbstractFloat, S, axis}
        reduced_shape = ntuple(i -> S[i < axis + 1 ? i : i + 1], length(S) - 1)
        Tile{T, reduced_shape}()
    end

    """
        reduce_max(tile, axis_val)

    Maximum reduction along 0-indexed axis.
    Compiled to cuda_tile.reduce with MAX.
    """
    @noinline function reduce_max(tile::Tile{T, S}, ::Val{axis}) where {T <: AbstractFloat, S, axis}
        reduced_shape = ntuple(i -> S[i < axis + 1 ? i : i + 1], length(S) - 1)
        Tile{T, reduced_shape}()
    end

    """
        scan(tile, axis_val, fn_type; reverse=false)

    Parallel prefix scan along specified dimension.
    fn_type=:add for cumulative sum, :mul for cumulative product.
    reverse=false for forward scan, true for reverse scan.
    Compiled to cuda_tile.scan.
    """
    @noinline function scan(tile::Tile{T, S}, ::Val{axis}, fn::Symbol, reverse::Bool=false) where {T, S, axis}
        # Scan preserves shape - result has same dimensions as input
        Tile{T, S}()
    end

    """
        scan_with_op(tile, axis_val, M_val; reverse=false)

    Parallel prefix scan with wrapped addition modulo M.
    Computes (acc + elem) % M at each step.
    Compiled to cuda_tile.scan with custom body.
    """
    @noinline function scan_with_op(tile::Tile{T, S}, ::Val{axis}, ::Type{Val{M}}, reverse::Bool=false) where {T, S, axis, M}
        # Scan preserves shape - result has same dimensions as input
        Tile{T, S}()
    end

    """
        scan_with_custom_op(tile, axis_val, op_type, reverse=false)

    Parallel prefix scan with custom binary operator.
    Takes an operator type that implements scan_combine and scan_identity.
    Compiled to cuda_tile.scan with custom body.
    """
    @noinline function scan_with_custom_op(tile::Tile{T, S}, ::Val{axis}, op_type::Type, reverse::Bool=false) where {T, S, axis}
        # Scan preserves shape - result has same dimensions as input
        Tile{T, S}()
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
    elem_type = input_type <: Tile ? input_type.parameters[1] : input_type
    input_shape = input_tv.shape
    isempty(input_shape) && error("Cannot reduce scalar tile")

    # Compute output shape (dimension at axis is removed)
    output_shape = Int[input_shape[i] for i in eachindex(input_shape) if i != axis + 1]

    dtype = julia_to_tile_dtype!(tt, elem_type)

    # Output tile type
    output_tile_type = tile_type!(tt, dtype, output_shape)

    # Scalar type for reduction body (0D tile)
    scalar_tile_type = tile_type!(tt, dtype, Int[])

    # Create identity value - use simple dtype (f32), not tile type
    identity_val = reduce_fn == :add ? -0.0 : (reduce_fn == :max ? -Inf : 0.0)
    identity = FloatIdentity(identity_val, dtype, elem_type)

    # Emit ReduceOp
    results = encode_ReduceOp!(cb, [output_tile_type], [input_tv.v], axis, [identity], [scalar_tile_type]) do block_args
        acc, elem = block_args[1], block_args[2]
        res = encode_reduce_body(cb, scalar_tile_type, acc, elem, Val(reduce_fn), elem_type)
        encode_YieldOp!(cb, [res])
    end

    CGVal(results[1], output_tile_type, Tile{elem_type, Tuple(output_shape)}, output_shape)
end

# Dispatch helpers for reduce body operations - dispatch on Val{fn} and elem_type
encode_reduce_body(cb, type, acc, elem, ::Val{:add}, ::Type{T}) where T <: AbstractFloat =
    encode_AddFOp!(cb, type, acc, elem)
encode_reduce_body(cb, type, acc, elem, ::Val{:max}, ::Type{T}) where T <: AbstractFloat =
    encode_MaxFOp!(cb, type, acc, elem)
encode_reduce_body(cb, type, acc, elem, ::Val{:add}, ::Type{T}) where T <: Integer =
    encode_AddIOp!(cb, type, acc, elem)
encode_reduce_body(cb, type, acc, elem, ::Val{:max}, ::Type{T}) where T <: Integer =
    encode_MaxIOp!(cb, type, acc, elem)

function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.scan), args)
    emit_scan!(ctx, args)
end

function emit_scan!(ctx::CGCtx, args)
    cb = ctx.cb
    tt = ctx.tt

    # Get input tile
    input_tv = emit_value!(ctx, args[1])
    input_tv === nothing && error("Cannot resolve input tile for scan")

    # Get scan axis
    axis = @something get_constant(ctx, args[2]) error("Scan axis must be a compile-time constant")

    # Get scan function type
    fn_type = @something get_constant(ctx, args[3]) error("Scan function type must be a compile-time constant")
    fn_type == :add || fn_type == :mul || error("Scan function must be :add or :mul")

    # Get reverse flag (optional, defaults to false)
    reverse = false
    if length(args) >= 4
        reverse_val = get_constant(ctx, args[4])
        reverse = reverse_val === true
    end

    # Get element type and shapes
    input_type = unwrap_type(input_tv.jltype)
    elem_type = input_type <: Tile ? input_type.parameters[1] : input_type
    input_shape = input_tv.shape

    # For scan, output shape is same as input shape
    output_shape = copy(input_shape)

    dtype = julia_to_tile_dtype!(tt, elem_type)

    # Output tile type (same shape as input)
    output_tile_type = tile_type!(tt, dtype, output_shape)

    # Scalar type for scan body (0D tile)
    scalar_tile_type = tile_type!(tt, dtype, Int[])

    # Create identity value
    # For cumsum: identity is 0.0 (represented as -0.0 for float)
    # For cumprod: identity is 1.0
    if fn_type == :add
        identity_val = -0.0  # Negative zero works as additive identity
    else  # :mul
        identity_val = 1.0
    end

    # Choose identity type based on element type
    if elem_type <: AbstractFloat
        # Use float identity for float types
        identity = ScanFloatIdentity(identity_val, dtype, elem_type)
    elseif elem_type <: Integer
        # Use integer identity for integer types
        identity_val_int = fn_type == :add ? Int64(0) : Int64(1)
        is_signed = elem_type <: Signed
        identity = ScanIntegerIdentity(identity_val_int, dtype, elem_type, is_signed)
    else
        error("Unsupported element type for scan: $elem_type")
    end

    # Emit ScanOp
    results = encode_ScanOp!(cb, [output_tile_type], [input_tv.v], axis, reverse, [identity], [scalar_tile_type]) do block_args
        acc, elem = block_args[1], block_args[2]
        res = encode_scan_body(cb, scalar_tile_type, acc, elem, Val(fn_type), elem_type)
        encode_YieldOp!(cb, [res])
    end

    CGVal(results[1], output_tile_type, Tile{elem_type, Tuple(output_shape)}, output_shape)
end

# Dispatch helpers for scan body operations - dispatch on Val{fn} and elem_type
encode_scan_body(cb, type, acc, elem, ::Val{:add}, ::Type{T}) where T <: AbstractFloat =
    encode_AddFOp!(cb, type, acc, elem)
encode_scan_body(cb, type, acc, elem, ::Val{:add}, ::Type{T}) where T <: Integer =
    encode_AddIOp!(cb, type, acc, elem)
encode_scan_body(cb, type, acc, elem, ::Val{:mul}, ::Type{T}) where T <: AbstractFloat =
    encode_MulFOp!(cb, type, acc, elem)
encode_scan_body(cb, type, acc, elem, ::Val{:mul}, ::Type{T}) where T <: Integer =
    encode_MulIOp!(cb, type, acc, elem)

# Custom scan with binary operator (e.g., WrappedAddMod for modulo arithmetic)
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.scan_with_op), args)
    emit_scan_with_op!(ctx, args)
end

# Custom operator scan - generates scan body from user-defined combine function
function emit_intrinsic!(ctx::CGCtx, ::typeof(Intrinsics.scan_with_custom_op), args)
    emit_scan_with_custom_op!(ctx, args)
end

function emit_scan_with_custom_op!(ctx::CGCtx, args)
    cb = ctx.cb
    tt = ctx.tt

    # Get input tile
    input_tv = emit_value!(ctx, args[1])
    input_tv === nothing && error("Cannot resolve input tile for scan")

    # Get scan axis
    axis = @something get_constant(ctx, args[2]) error("Scan axis must be a compile-time constant")

    # Get operator type (e.g., MyOp from scan(tile, axis(1), MyOp()))
    op_type_arg = args[3]
    op_type = unwrap_type(get_constant(ctx, op_type_arg))
    if !(op_type isa Type)
        error("Operator type must be a compile-time constant type")
    end
    OpT = op_type

    # Get reverse flag (optional, defaults to false)
    reverse = false
    if length(args) >= 4
        reverse_val = get_constant(ctx, args[4])
        reverse = reverse_val === true
    end

    # Get element type and shapes
    input_type = unwrap_type(input_tv.jltype)
    elem_type = input_type <: Tile ? input_type.parameters[1] : input_type
    input_shape = input_tv.shape

    # Output shape same as input
    output_shape = copy(input_shape)

    dtype = julia_to_tile_dtype!(tt, elem_type)
    output_tile_type = tile_type!(tt, dtype, output_shape)
    scalar_tile_type = tile_type!(tt, dtype, Int[])

    # Get identity value from operator's scan_identity method
    identity = get_scan_identity_for_operator(OpT, elem_type, dtype, tt)

    # Determine the operation from the user's scan_combine function
    op_kind = get_combine_operation(OpT, elem_type)

    # Emit ScanOp with custom body based on the operation
    results = encode_ScanOp!(cb, [output_tile_type], [input_tv.v], axis, reverse, [identity], [scalar_tile_type]) do block_args
        acc, elem = block_args[1], block_args[2]
        res = emit_combine_operation!(ctx, scalar_tile_type, acc, elem, op_kind, elem_type)
        encode_YieldOp!(cb, [res])
    end

    CGVal(results[1], output_tile_type, Tile{elem_type, Tuple(output_shape)}, output_shape)
end

"""
    get_scan_identity_for_operator(OpT, elem_type, dtype, tt) -> ScanIdentity

Get the identity value for a scan operator by calling its scan_identity method.
"""
function get_scan_identity_for_operator(::Type{OpT}, ::Type{T}, dtype::TypeId, tt::TypeTable) where {OpT, T}
    # Try to call scan_identity(OpT(), ::Type{T})
    identity_val = try
        Base.eval(Main, :(cuTile.scan_identity($OpT(), $(T))))
    catch
        # Default identity: 0 for most ops
        T(0)
    end

    if T <: AbstractFloat
        ScanFloatIdentity(Float64(identity_val), dtype, T)
    elseif T <: Integer
        is_signed = T <: Signed
        ScanIntegerIdentity(Int64(identity_val), dtype, T, is_signed)
    else
        error("Unsupported element type for custom scan: $T")
    end
end

"""
    CombineOpKind

Supported operation kinds for custom scan operators.
"""
@enum CombineOpKind begin
    COMBINE_ADD      # a + b
    COMBINE_MUL      # a * b
    COMBINE_MAX      # max(a, b)
    COMBINE_MIN      # min(a, b)
    COMBINE_SUB      # a - b
    COMBINE_UNKNOWN  # Unknown operation
end

"""
    get_combine_operation(OpT, elem_type) -> CombineOpKind

Analyze the user's scan_combine function to determine what operation it performs.
Uses known operator type detection as the primary method, with fallback to
analyzing the actual scan_combine function AST.
"""
function get_combine_operation(::Type{OpT}, ::Type{T}) where {OpT, T}
    # First, check if it's a known operator type with explicit mapping
    if OpT <: AddOp
        return COMBINE_ADD
    elseif OpT <: MulOp
        return COMBINE_MUL
    elseif OpT <: MaxOp
        return COMBINE_MAX
    elseif OpT <: MinOp
        return COMBINE_MIN
    elseif OpT <: WrappedAddMod
        return COMBINE_ADD
    end

    # For truly custom operators, analyze the scan_combine function AST
    return detect_operation_from_method(OpT, T)
end

"""
    detect_operation_from_method(OpT, T) -> CombineOpKind

Analyze the scan_combine method's AST to detect the binary operation.
Traces through SSA values to find the actual operation.
"""
function detect_operation_from_method(::Type{OpT}, ::Type{T}) where {OpT, T}
    try
        # Get the Method for scan_combine
        method = which(cuTile.scan_combine, (OpT, Tile{T, Tuple{}}, Tile{T, Tuple{}}))

        # Get the method's CodeInfo (uncompressed AST)
        code_info = Base.uncompressed_ast(method)

        # Find the return statement and trace SSA values
        for stmt in code_info.code
            if stmt isa Core.ReturnNode && isdefined(stmt, :val)
                result = trace_ssa_value(stmt.val, code_info.code)
                if result != COMBINE_UNKNOWN
                    return result
                end
            end
        end

        # Fallback: default to add
        return COMBINE_ADD
    catch
        return COMBINE_ADD
    end
end

"""
    trace_ssa_value(val, code) -> CombineOpKind

Trace an SSA value to find the expression it refers to.
"""
function trace_ssa_value(@nospecialize(val), code::Vector{Any})
    if val isa Core.SSAValue
        idx = val.id
        if idx > 0 && idx <= length(code)
            return analyze_statement(code[idx], code)
        end
    elseif val isa Core.ReturnNode
        return trace_ssa_value(val.val, code)
    elseif val isa Expr
        return analyze_expr(val)
    elseif val isa QuoteNode
        return trace_ssa_value(val.value, code)
    elseif val isa GlobalRef
        return detect_from_name(val.name)
    end
    return COMBINE_UNKNOWN
end

"""
    analyze_statement(stmt, code) -> CombineOpKind

Analyze a statement to detect the binary operation.
"""
function analyze_statement(@nospecialize(stmt), code::Vector{Any})
    if stmt isa Core.ReturnNode
        return trace_ssa_value(stmt.val, code)
    elseif stmt isa Expr
        return analyze_expr(stmt, code)
    elseif stmt isa GlobalRef
        return detect_from_name(stmt.name)
    elseif stmt isa Slot
        return COMBINE_UNKNOWN
    end
    return COMBINE_UNKNOWN
end

"""
    analyze_expr(expr, code) -> CombineOpKind

Analyze an expression to detect binary operations.
"""
function analyze_expr(@nospecialize(expr), code::Vector{Any})
    if expr isa Expr
        if expr.head === :call
            # Function call - check the function being called
            fn = expr.args[1]
            return detect_function_expr(fn, code)
        elseif expr.head === :body
            # Look for the last meaningful expression
            for i in length(expr.args):-1:1
                result = analyze_expr(expr.args[i], code)
                if result != COMBINE_UNKNOWN
                    return result
                end
            end
        elseif expr.head === :(=)
            # Assignment - analyze the RHS
            return analyze_expr(expr.args[end], code)
        elseif expr.head === :invoke
            # Direct method call
            fn = expr.args[1]
            if fn isa Core.MethodInstance
                return detect_function_expr(fn.def.name, code)
            end
        end
    elseif expr isa GlobalRef
        return detect_from_name(expr.name)
    elseif expr isa Symbol
        return detect_from_name(expr)
    elseif expr isa Core.SSAValue
        # Resolve the SSA value
        idx = expr.id
        if idx > 0 && idx <= length(code)
            return analyze_statement(code[idx], code)
        end
    elseif expr isa Core.Argument
        return COMBINE_UNKNOWN
    elseif expr isa Slot
        return COMBINE_UNKNOWN
    end
    return COMBINE_UNKNOWN
end

"""
    detect_function_expr(fn, code) -> CombineOpKind

Detect the operation from a function expression in a call.
"""
function detect_function_expr(@nospecialize(fn), code::Vector{Any})
    if fn isa Symbol
        return detect_from_name(fn)
    elseif fn isa GlobalRef
        return detect_from_name(fn.name)
    elseif fn isa Core.SSAValue
        # The function is stored in an SSA variable - resolve it
        idx = fn.id
        if idx > 0 && idx <= length(code)
            return analyze_statement(code[idx], code)
        end
    elseif fn isa Core.MethodInstance
        return detect_from_name(fn.def.name)
    elseif fn isa Core.Slot
        return COMBINE_UNKNOWN
    elseif fn isa Core.Argument
        return COMBINE_UNKNOWN
    end
    return COMBINE_UNKNOWN
end

"""
    detect_from_name(name::Symbol) -> CombineOpKind

Detect operation from a function name symbol.
"""
function detect_from_name(name::Symbol)
    if name === :mul || name === :*
        return COMBINE_MUL
    elseif name === :max
        return COMBINE_MAX
    elseif name === :min
        return COMBINE_MIN
    elseif name === :add || name === :+
        return COMBINE_ADD
    elseif name === :sub || name === :-
        return COMBINE_SUB
    end
    return COMBINE_UNKNOWN
end

"""
    emit_combine_operation!(ctx, type, acc, elem, op_kind, elem_type) -> Value

Emit the appropriate Tile operation based on the combine operation kind.
"""
function emit_combine_operation!(ctx::CGCtx, type::TypeId, acc::Value, elem::Value, op_kind::CombineOpKind, ::Type{T}) where T
    cb = ctx.cb

    if op_kind == COMBINE_ADD
        return T <: Integer ? encode_AddIOp!(cb, type, acc, elem) : encode_AddFOp!(cb, type, acc, elem)
    elseif op_kind == COMBINE_MUL
        return T <: Integer ? encode_MulIOp!(cb, type, acc, elem) : encode_MulFOp!(cb, type, acc, elem)
    elseif op_kind == COMBINE_MAX
        return T <: Integer ? encode_MaxIOp!(cb, type, acc, elem; signedness=SignednessSigned) : encode_MaxFOp!(cb, type, acc, elem)
    elseif op_kind == COMBINE_MIN
        return T <: Integer ? encode_MinIOp!(cb, type, acc, elem; signedness=SignednessSigned) : encode_MinFOp!(cb, type, acc, elem)
    elseif op_kind == COMBINE_SUB
        return T <: Integer ? encode_SubIOp!(cb, type, acc, elem) : encode_SubFOp!(cb, type, acc, elem)
    else
        return T <: Integer ? encode_AddIOp!(cb, type, acc, elem) : encode_AddFOp!(cb, type, acc, elem)
    end
end

function emit_scan_with_op!(ctx::CGCtx, args)
    cb = ctx.cb
    tt = ctx.tt

    # Get input tile
    input_tv = emit_value!(ctx, args[1])
    input_tv === nothing && error("Cannot resolve input tile for scan")

    # Get scan axis
    axis = @something get_constant(ctx, args[2]) error("Scan axis must be a compile-time constant")

    # Get modulus M from Type{Val{M}} type parameter
    mod_type_arg = args[3]
    mod_type = unwrap_type(get_constant(ctx, mod_type_arg))
    M = mod_type isa Type && mod_type <: Val ? mod_type.parameters[1] : mod_type

    # Get reverse flag (optional, defaults to false)
    reverse = false
    if length(args) >= 4
        reverse_val = get_constant(ctx, args[4])
        reverse = reverse_val === true
    end

    # Get element type and shapes
    input_type = unwrap_type(input_tv.jltype)
    elem_type = input_type <: Tile ? input_type.parameters[1] : input_type
    input_shape = input_tv.shape

    # For scan, output shape is same as input shape
    output_shape = copy(input_shape)

    dtype = julia_to_tile_dtype!(tt, elem_type)

    # Output tile type (same shape as input)
    output_tile_type = tile_type!(tt, dtype, output_shape)

    # Scalar type for scan body (0D tile)
    scalar_tile_type = tile_type!(tt, dtype, Int[])

    # Create identity value (0 for addmod)
    identity_val = 0
    if elem_type <: AbstractFloat
        identity = ScanFloatIdentity(Float64(identity_val), dtype, elem_type)
    elseif elem_type <: Integer
        is_signed = elem_type <: Signed
        identity = ScanIntegerIdentity(Int64(identity_val), dtype, elem_type, is_signed)
    else
        error("Unsupported element type for scan: $elem_type")
    end

    # Emit ScanOp with addmod body
    results = encode_ScanOp!(cb, [output_tile_type], [input_tv.v], axis, reverse, [identity], [scalar_tile_type]) do block_args
        acc, elem = block_args[1], block_args[2]
        res = encode_scan_addmod_body!(cb, scalar_tile_type, acc, elem, Val(M), elem_type)
        encode_YieldOp!(cb, [res])
    end

    CGVal(results[1], output_tile_type, Tile{elem_type, Tuple(output_shape)}, output_shape)
end

# Encode (acc + elem) % M for addmod scan
function encode_scan_addmod_body!(cb, type, acc, elem, ::Val{M}, ::Type{T}) where {M, T}
    # acc + elem
    sum_res = if T <: AbstractFloat
        encode_AddFOp!(cb, type, acc, elem)
    elseif T <: Integer
        encode_AddIOp!(cb, type, acc, elem)
    else
        error("Unsupported type for addmod scan: $T")
    end
    # Create constant M as 0D tile (same type as acc/elem)
    M_bytes = if T === Float32
        collect(reinterpret(UInt8, [Float32(M)]))
    elseif T === Float64
        collect(reinterpret(UInt8, [Float64(M)]))
    elseif T === Int32
        collect(reinterpret(UInt8, [Int32(M)]))
    elseif T === Int64
        collect(reinterpret(UInt8, [Int64(M)]))
    elseif T === UInt32
        collect(reinterpret(UInt8, [UInt32(M)]))
    elseif T === UInt64
        collect(reinterpret(UInt8, [UInt64(M)]))
    else
        error("Unsupported type for addmod constant: $T")
    end
    M_val = encode_ConstantOp!(cb, type, M_bytes)
    # (acc + elem) % M
    if T <: AbstractFloat
        encode_RemFOp!(cb, type, sum_res, M_val)
    else
        encode_RemIOp!(cb, type, sum_res, M_val; signedness=SignednessSigned)
    end
end



## cuda_tile.reshape

@eval Intrinsics begin
    """
        reshape(tile, shape_val)

    Reshape a tile to a new shape (same total elements).
    Compiled to cuda_tile.reshape.
    """
    @noinline function reshape(tile::Tile{T, S}, ::Val{Shape}) where {T, S, Shape}
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

    # Get element type
    source_type = unwrap_type(source.jltype)
    elem_type = source_type <: Tile ? source_type.parameters[1] : source_type

    # Create target tile type
    dtype = julia_to_tile_dtype!(tt, elem_type)
    result_type_id = tile_type!(tt, dtype, target_shape)

    # Emit ReshapeOp
    result_v = encode_ReshapeOp!(cb, result_type_id, source.v)

    CGVal(result_v, result_type_id, Tile{elem_type, Tuple(target_shape)}, target_shape)
end


## TODO: cuda_tile.scan


## cuda_tile.select

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


## TODO: cuda_tile.unpack
