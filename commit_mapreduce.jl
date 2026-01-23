julia
#!/usr/bin/env julia
"""
    commit_mapreduce.jl

Script to commit the mapreduce implementation changes.
This script stages and commits all changes with appropriate commit message.
"""

# Change to cuTile directory
cd(joinpath(@__DIR__, "cuTile"))

# Run git commands
println("Staging all changes...")
run(`git add -A`)

println("\nChanges staged:")
run(`git status`)

println("\nCreating commit...")
commit_message = """
mapreduce: Add generalized map-reduce operation for Tile IR

This commit implements a new `mapreduce` function that combines element-wise
mapping with reduction operations into a single fused GPU kernel.

## Features
- Generalized reduction operation supporting custom map and reduce functions
- Support for common map functions: identity, abs, abs2, sqrt, exp, log, sin, cos, neg
- Support for reduction functions: +, *, max, min
- Proper identity value computation for each reduction operation
- 1-indexed axis (Julia convention) converted to 0-indexed for Tile IR
- Compiles to ReduceOp with custom body region

## Files Changed
- src/language/operations.jl: Added public API and user-facing functions
- src/compiler/intrinsics/core.jl: Added bytecode generation for mapreduce
- docs/MAPREDUCE.md: Implementation documentation

## Testing
- test_mapreduce_simple.jl: Basic functionality tests
- verify_mapreduce.jl: Comprehensive validation against Tile IR spec

## Compliance
Verified against Tile IR documentation (tileirdocs/08-operations.md):
- Uses correct ReduceOp encoding (opcode 88)
- Proper identity value computation per operation
- Body region with (acc, elem) arguments as specified
- Map function fusion before reduction

Generated on: $(Dates.format(now(), "Y-m-d H:M"))
"""

# Write commit message to file
open("COMMIT_MESSAGE.txt", "w") do io
    write(io, commit_message)
end

# Create commit
run(`git commit -F COMMIT_MESSAGE.txt`)

# Clean up
rm("COMMIT_MESSAGE.txt"; force=true)

println("\n✓ Commit created successfully!")
println("\nTo push: git push origin HEAD")

# Optionally push
println("\nWould you like to push? (y/n)")
answer = readline()
if lowercase(answer) == "y" || lowercase(answer) == "yes"
    println("Pushing to origin...")
    run(`git push origin HEAD`)
    println("✓ Changes pushed successfully!")
else
    println("Skipping push. Run 'git push origin HEAD' when ready.")
end

println("\nDone!")
