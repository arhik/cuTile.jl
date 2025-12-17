using cuTile
using ParallelTestRunner

const init_code = quote
    import cuTile as ct
end

testsuite = find_tests(pwd())

# Add examples
function find_sources(path::String, sources=String[])
    if isdir(path)
        for entry in readdir(path)
            find_sources(joinpath(path, entry), sources)
        end
    elseif endswith(path, ".jl")
        push!(sources, path)
    end
    sources
end
examples_dir = joinpath(@__DIR__, "..", "examples")
examples = find_sources(examples_dir)
filter!(file -> readline(file) != "# EXCLUDE FROM TESTING", examples)
for example in examples
    name = splitext("examples/$(basename(example))")[1]
    testsuite[name] = quote
        cd($examples_dir) do
            mod = @eval module $(gensym()) end
            @eval mod begin
                redirect_stdout(devnull) do
                    include($example)
                end
            end
        end
    end
end

runtests(cuTile, ARGS; init_code, testsuite)
