using cuTile
using ParallelTestRunner

const init_code = quote
    import cuTile as ct
end

runtests(cuTile, ARGS; init_code)
