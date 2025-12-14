@testset "restructuring" verbose=true begin

using cuTile: code_structured, StructuredCodeInfo, Block, IfOp, ForOp, LoopOp,
              YieldOp, ContinueOp, BreakOp

# Helper to check if a block contains a specific control flow op type
function has_nested_op(block::Block, ::Type{T}) where T
    any(op -> op isa T, block.nested)
end

# Helper to count nested ops of a type
function count_nested_ops(block::Block, ::Type{T}) where T
    count(op -> op isa T, block.nested)
end

# Recursive helper to find all ops of a type in a StructuredCodeInfo
function find_all_ops(sci::StructuredCodeInfo, ::Type{T}) where T
    ops = T[]
    find_ops_in_block!(ops, sci.entry, T)
    return ops
end

function find_ops_in_block!(ops::Vector{T}, block::Block, ::Type{T}) where T
    for op in block.nested
        if op isa T
            push!(ops, op)
        end
        find_ops_in_op!(ops, op, T)
    end
end

function find_ops_in_op!(ops::Vector{T}, op::IfOp, ::Type{T}) where T
    find_ops_in_block!(ops, op.then_block, T)
    find_ops_in_block!(ops, op.else_block, T)
end

function find_ops_in_op!(ops::Vector{T}, op::ForOp, ::Type{T}) where T
    find_ops_in_block!(ops, op.body, T)
end

function find_ops_in_op!(ops::Vector{T}, op::LoopOp, ::Type{T}) where T
    find_ops_in_block!(ops, op.body, T)
end

@testset "straight-line code" begin
    # Simple function with no control flow
    f(x) = x + 1

    sci = code_structured(f, Tuple{Int})
    @test sci isa StructuredCodeInfo
    @test !isempty(sci.entry.stmts)
    @test isempty(sci.entry.nested)  # No nested control flow
    @test sci.entry.terminator isa Core.ReturnNode

    # Multiple operations, still straight-line
    g(x, y) = (x + y) * (x - y)

    sci = code_structured(g, Tuple{Int, Int})
    @test sci isa StructuredCodeInfo
    @test isempty(sci.entry.nested)
    @test sci.entry.terminator isa Core.ReturnNode
end

# Note: Julia's optimized IR often has multiple returns instead of merging
# branches, which makes pattern matching difficult. These tests verify
# that code_structured handles control flow gracefully, even if not
# fully restructured into nested ops.
@testset "control flow handling" begin
    # Ternary operator - may fall back to flat representation
    # due to separate returns in each branch
    f(x) = x > 0 ? x + 1 : x - 1

    sci = code_structured(f, Tuple{Int})
    @test sci isa StructuredCodeInfo
    # Should have statements (either restructured or flat)
    @test !isempty(sci.entry.stmts) || !isempty(sci.entry.nested)

    # Multiple returns
    function multi_return(x)
        if x < 0
            return -1
        elseif x == 0
            return 0
        else
            return 1
        end
    end

    sci = code_structured(multi_return, Tuple{Int})
    @test sci isa StructuredCodeInfo

    # Boolean short-circuit
    f_and(x, y) = x > 0 && y > 0
    sci = code_structured(f_and, Tuple{Int, Int})
    @test sci isa StructuredCodeInfo

    f_or(x, y) = x > 0 || y > 0
    sci = code_structured(f_or, Tuple{Int, Int})
    @test sci isa StructuredCodeInfo
end

@testset "terminating if-then-else" begin
    # Simple ternary with Bool condition - both branches return
    f(x) = x ? 1 : 2
    sci = code_structured(f, Tuple{Bool})
    @test sci isa StructuredCodeInfo
    @test has_nested_op(sci.entry, IfOp)

    # Verify the IfOp structure
    if_ops = find_all_ops(sci, IfOp)
    @test length(if_ops) == 1
    if_op = if_ops[1]
    @test if_op.then_block.terminator isa Core.ReturnNode
    @test if_op.else_block.terminator isa Core.ReturnNode

    # Ternary with computed condition - stmt before if, both branches compute + return
    g(x) = x > 0 ? x + 1 : x - 1
    sci = code_structured(g, Tuple{Int})
    @test sci isa StructuredCodeInfo
    @test has_nested_op(sci.entry, IfOp)

    # Should have condition computation before the IfOp
    @test !isempty(sci.entry.stmts)

    # Verify IfOp with computations in branches
    if_ops = find_all_ops(sci, IfOp)
    @test length(if_ops) == 1
    if_op = if_ops[1]
    # Each branch should have a computation statement and a return
    @test !isempty(if_op.then_block.stmts)
    @test !isempty(if_op.else_block.stmts)
    @test if_op.then_block.terminator isa Core.ReturnNode
    @test if_op.else_block.terminator isa Core.ReturnNode

    # Display should show scf.if structure
    io = IOBuffer()
    show(io, MIME"text/plain"(), sci)
    output = String(take!(io))
    @test occursin("scf.if", output)
    @test occursin("return", output)
end

@testset "display output" begin
    # Test straight-line display
    f(x) = x + 1
    sci = code_structured(f, Tuple{Int})

    io = IOBuffer()
    show(io, MIME"text/plain"(), sci)
    output = String(take!(io))

    @test occursin("StructuredCodeInfo", output)
    @test occursin("%", output)  # Has SSA values
    @test occursin("return", output)  # Has terminator

    # Test compact display
    io = IOBuffer()
    show(io, sci)
    output = String(take!(io))

    @test occursin("StructuredCodeInfo", output)
    @test occursin("stmts", output)
end

@testset "display with control flow" begin
    # Terminating if-then-else should be properly restructured
    f(x) = x > 0 ? x + 1 : x - 1
    sci = code_structured(f, Tuple{Int})

    io = IOBuffer()
    show(io, MIME"text/plain"(), sci)
    output = String(take!(io))

    @test occursin("StructuredCodeInfo", output)
    @test occursin("scf.if", output)  # Should show structured if
    @test occursin("return", output)  # Both branches have returns
end

# Note: Loop restructuring requires complex analysis and may fall back to flat.
@testset "loop handling" begin
    # Simple for loop
    function sum_to_n(n)
        total = 0
        for i in 1:n
            total += i
        end
        return total
    end

    # Should not error, even if loops aren't fully restructured
    sci = code_structured(sum_to_n, Tuple{Int})
    @test sci isa StructuredCodeInfo

    # While loop
    function count_down(n)
        while n > 0
            n -= 1
        end
        return n
    end

    sci = code_structured(count_down, Tuple{Int})
    @test sci isa StructuredCodeInfo
end

@testset "type preservation" begin
    # Verify that types from original CodeInfo are preserved
    f(x::Float64) = x + 1.0

    sci = code_structured(f, Tuple{Float64})
    @test sci isa StructuredCodeInfo

    # The underlying CodeInfo should have proper types
    @test !isempty(sci.code.ssavaluetypes)
    # Float64 operations should appear
    @test any(t -> t isa Type && t <: AbstractFloat, sci.code.ssavaluetypes)
end

@testset "argument handling" begin
    # Single argument
    f(x) = x * 2
    sci = code_structured(f, Tuple{Int})
    @test sci isa StructuredCodeInfo

    # Multiple arguments
    g(x, y, z) = x + y + z
    sci = code_structured(g, Tuple{Int, Int, Int})
    @test sci isa StructuredCodeInfo

    # Different types
    h(x::Int, y::Float64) = x + y
    sci = code_structured(h, Tuple{Int, Float64})
    @test sci isa StructuredCodeInfo
end

end  # @testset "restructuring"
