# expression emission

"""
    emit_expr!(ctx, expr::Expr, result_type) -> Union{CGVal, Nothing}

Emit bytecode for an expression.
"""
function emit_expr!(ctx::CGCtx, expr::Expr, @nospecialize(result_type))
    if expr.head === :call
        return emit_call!(ctx, expr, result_type)
    elseif expr.head === :invoke
        return emit_invoke!(ctx, expr, result_type)
    elseif expr.head === :(=)
        return emit_assignment!(ctx, expr, result_type)
    elseif expr.head === :new
        return nothing  # Struct construction handled elsewhere
    elseif expr.head === :foreigncall
        error("Foreign calls not supported in Tile IR")
    elseif expr.head === :boundscheck
        return nothing
    else
        @warn "Unhandled expression head" expr.head expr
        return nothing
    end
end

function emit_assignment!(ctx::CGCtx, expr::Expr, @nospecialize(result_type))
    lhs = expr.args[1]
    rhs = expr.args[2]

    tv = emit_rhs!(ctx, rhs, result_type)

    if lhs isa SlotNumber && tv !== nothing
        ctx[lhs] = tv
    end

    return tv
end

function emit_rhs!(ctx::CGCtx, @nospecialize(rhs), @nospecialize(result_type))
    if rhs isa Expr
        return emit_expr!(ctx, rhs, result_type)
    elseif rhs isa SSAValue || rhs isa SlotNumber || rhs isa Argument
        return emit_value!(ctx, rhs)
    elseif rhs isa QuoteNode
        return emit_constant!(ctx, rhs.value, result_type)
    elseif rhs isa GlobalRef
        return nothing
    else
        return emit_constant!(ctx, rhs, result_type)
    end
end

#=============================================================================
 Call Emission
=============================================================================#

"""
    emit_call!(ctx, expr::Expr, result_type) -> Union{CGVal, Nothing}

Emit bytecode for a function call.
"""
function emit_call!(ctx::CGCtx, expr::Expr, @nospecialize(result_type))
    args = expr.args
    func = resolve_function(ctx, args[1])
    call_args = args[2:end]

    # Special handling for mapreduce with anonymous functions
    # Try to detect and inline lambda bodies
    if func === Intrinsics.mapreduce || func === Intrinsics.mapreduce_noinit || func === Intrinsics.mapreduce_withinit
        if length(call_args) >= 2
            # Try to extract lambda body from first argument
            lambda_body = try_extract_lambda(ctx, call_args[1])
            if lambda_body !== nothing
                # Replace lambda with body expression
                call_args = vcat([lambda_body], call_args[2:end])
                # Create new expression with replaced lambda
                if expr isa Expr
                    new_expr = Expr(:call, expr.args[1], call_args...)
                    return emit_call!(ctx, new_expr, result_type)
                end
            end
        end
    end

    # TODO: This is normally dynamic dispatch, which we should allow.
    #       However, we currently trigger this when emitting Julia intrinsics.
    #       We should switch to our own intrinsics entirely, which are only invoked.

    if func === Core.getfield
        tv = emit_getfield!(ctx, call_args, result_type)
        tv !== nothing && return tv
    elseif func === Base.getindex
        tv = emit_getindex!(ctx, call_args, result_type)
        tv !== nothing && return tv
    elseif func === Core.apply_type
        # Type construction is compile-time only - result_type tells us the constructed type
        return ghost_value(result_type, result_type)
    end

    result = emit_intrinsic!(ctx, func, call_args)
    result === missing && error("Unknown function call: $func")
    validate_result_type(result, result_type, func)
    return result
end

"""
    try_extract_lambda(ctx, arg) -> Union{Expr, Nothing}

Try to extract the body of a lambda from the first argument of mapreduce.
Returns the lambda body expression or nothing if extraction fails.
"""
function try_extract_lambda(ctx::CGCtx, @nospecialize(arg))
    # Case 1: Lambda is directly an expression (uncompiled form)
    if arg isa Expr && arg.head === :->
        # x -> body form
        return arg.args[2]
    end

    # Case 2: Lambda is a QuoteNode containing a Function
    if arg isa QuoteNode
        fn = arg.value
        if fn isa Function && !isgeneric(fn)
            return extract_lambda_body(fn)
        end
    end

    # Case 3: Lambda is stored in an SSA value
    if arg isa SSAValue
        tv = ctx[arg]
        if tv !== nothing && tv.constant !== nothing
            fn = something(tv.constant)
            if fn isa Function && !isgeneric(fn)
                return extract_lambda_body(fn)
            end
        end
    end

    # Case 4: Lambda type - try to extract from type information
    if arg isa Type && arg <: Function && !isabstracttype(arg)
        return extract_lambda_from_type(arg)
    end

    return nothing
end

"""
    extract_lambda_from_type(fn_type) -> Union{Expr, Nothing}

Extract lambda body from a function type using @generated mechanics.
"""
function extract_lambda_from_type(@nospecialize(fn_type))
    # Try to use the function's own @generated body to get info
    try
        # Check if we can get method instances
        ms = methods(fn_type)
        if isempty(ms)
            return nothing
        end

        m = first(ms)

        # Try to get the lambda's captured variables and body
        # The lambda type might contain closure data in its parameters
        params = fn_type.parameters
        if length(params) > 0
            # The first parameter is often the return type
            # Subsequent parameters are captured values
            for param in params
                if param isa Function && !isgeneric(param)
                    body = extract_lambda_body(param)
                    body !== nothing && return body
                end
            end
        end

        return nothing
    catch
        return nothing
    end
end

"""
    extract_lambda_body(fn) -> Union{Expr, Nothing}

Extract body from a non-generic function using reflection.
"""
function extract_lambda_body(@nospecialize(fn))
    try
        # Get method instances
        mis = Base.method_instances(fn)
        if isempty(mis)
            return nothing
        end

        mi = first(mis)

        # Get the method's source code info
        line = whereis(mi)[1]
        code = code_typed(mi; optimize=false)
        if isempty(code)
            return nothing
        end

        ci = first(code)
        src = ci.first
        ir = ci.second

        # For simple lambda, extract body from :lambda IR
        if src isa Core.CodeInfo
            for stmt in src.code
                if stmt isa Expr
                    if stmt.head === :return
                        if stmt.args[1] isa Expr && stmt.args[1].head === :lambda
                            # Found lambda definition
                            lambda_expr = stmt.args[1]
                            if length(lambda_expr.args) >= 2
                                body_expr = lambda_expr.args[2]
                                if body_expr isa Expr
                                    return body_expr
                                end
                            end
                        end
                    end
                end
            end
        end

        return nothing
    catch
        return nothing
    end
end

"""
    emit_invoke!(ctx, expr::Expr, result_type) -> Union{CGVal, Nothing}

Emit bytecode for a method invocation.
"""
function emit_invoke!(ctx::CGCtx, expr::Expr, @nospecialize(result_type))
    # invoke has: (MethodInstance, func, args...)
    func = resolve_function(ctx, expr.args[2])
    call_args = expr.args[3:end]

    result = emit_intrinsic!(ctx, func, call_args)
    result === missing && error("Unknown function call: $func")
    validate_result_type(result, result_type, func)
    return result
end

"""
    validate_result_type(result, expected_type, func)

Assert that the intrinsic returned a type compatible with what the IR expects.
"""
function validate_result_type(@nospecialize(result), @nospecialize(expected_type), @nospecialize(func))
    # For now, just validate that we got a result
    result === nothing && error("Intrinsic $func returned nothing")
    return result
end

"""
    resolve_function(ctx, ref) -> Any

Resolve a function reference to its actual value.
"""
function resolve_function(ctx::CGCtx, @nospecialize(ref))
    if ref isa GlobalRef
        return getfield(ref.mod, ref.name)
    elseif ref isa QuoteNode
        return ref.value
    elseif ref isa SSAValue
        tv = ctx[ref]
        tv !== nothing && tv.constant !== nothing && return something(tv.constant)
    end
    return ref
end
