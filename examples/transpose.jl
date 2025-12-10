# Matrix transpose example - Julia port of cuTile Python's Transpose.py sample

using CUDA
import cuTile as ct

# Transpose kernel with constant tile sizes
# Constant{Int} parameters are ghost types - filtered from parameters, values accessed via tm[], tn[]
function transpose_kernel(x::Ptr{T}, y::Ptr{T},
                          tm::ct.Constant{Int}, tn::ct.Constant{Int}) where {T}
    bidx = ct.bid(0)
    bidy = ct.bid(1)
    input_tile = ct.load(x, (bidx, bidy), (tm[], tn[]))
    transposed_tile = ct.transpose(input_tile)
    ct.store(y, (bidy, bidx), transposed_tile)
    return
end

function test_transpose(::Type{T}, m, n, tm, tn; name=nothing) where T
    name = something(name, "transpose ($m x $n, $T, tiles=$tm x $tn)")
    println("--- $name ---")
    x = CUDA.rand(T, m, n)

    # Compile with specific constant tile sizes
    argtypes = Tuple{Ptr{T}, Ptr{T}, ct.Constant{Int, tm}, ct.Constant{Int, tn}}
    cubin = ct.compile(transpose_kernel, argtypes; sm_arch="sm_120")
    cumod = CuModule(cubin)
    cufunc = CuFunction(cumod, "transpose_kernel")

    grid_x = cld(m, tm)
    grid_y = cld(n, tn)
    y = CUDA.zeros(T, n, m)

    # Note: Constant parameters are ghost types - NOT passed at launch time
    cudacall(cufunc, Tuple{CuPtr{T}, CuPtr{T}}, x, y; blocks=(grid_x, grid_y))

    @assert Array(y) ≈ transpose(Array(x))
    println("✓ passed")
end

function main()
    println("--- cuTile Matrix Transposition Examples ---\n")

    # Float32 tests (like Python's test case 2)
    test_transpose(Float32, 1024, 512, 32, 32)
    test_transpose(Float32, 1024, 512, 64, 64)

    # Float64 tests
    test_transpose(Float64, 1024, 512, 32, 32)
    test_transpose(Float64, 512, 1024, 64, 64)

    # Float16 tests (like Python's test case 1 with 128x128 tiles)
    test_transpose(Float16, 1024, 512, 128, 128)
    test_transpose(Float16, 1024, 1024, 64, 64)

    println("\n--- All transpose examples completed ---")
end

isinteractive() || main()
