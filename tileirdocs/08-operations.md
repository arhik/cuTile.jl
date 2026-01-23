8. Operations — Tile IR
  

  

  
   Skip to main content

  

  

  


  

  

    Back to top

  



  

  

    Choose version  

  

  



Ctrl+K

 




Ctrl+K

  





 Table of Contents
 

 1. Introduction
 2. Programming Model
 3. Syntax
 4. Binary Format
 5. Type System
 6. Semantics
 7. Memory Model
 8. Operations
 9. Debug Info
 10. Stability
 11. Appendix
 12. Release Notes

 

 

 

 
 



 
 8. Operations#

This section describes a complete and categorized list of all Tile IR instructions names, signatures, and semantics.

## 8.1. Meta Types#

Operations have arguments which are Tile IR values with Tile IR types but many operations have immediate or static arguments which correspond
to attributes in the MLIR dialect. These meta types are not representable in the Tile IR type system but are used to construct Tile IR programs
and only present at compile time. Operations in the specification are described abstractly in both the Tile IR IR and bytecode independent of
the MLIR or bytecode encoding. For each of these types we provide a definition of them below and link to them from each operation.

Note
The convention is that the meta types are capitalized and Tile IR types are snake cased.

The convention is that the meta types are capitalized and the native Tile IR types are camel cased are snake cased.

## 8.1.1. Symbol#

Symbol a symbol in the program, begins with @ and uniquely identifies a symbol in the program.


## 8.1.2. Flag#

Flag a boolean value that can be used to control the behavior of an operation.


## 8.1.3. Token#

Token represents a memory ordering token that can be used to control the ordering of
memory operations.


## 8.1.4. Variadic#

Variadic represents an argument which can accept a statically sized, but variable, number of arguments.


## 8.1.5. Any#

Any represents a value of any valid Tile IR type.


## 8.1.6. Name#

Name represents a name in the program, begins with # and uniquely identifies a name in the program.


## 8.1.7. Type#

Type represents a Tile IR type and are attached as attributes to operations which define IR items.


## 8.1.8. Array#

Array represents a statically sized array of values that can be passed to attributes.


## 8.1.9. String#

String represents a string value that can be passed to attributes.


## 8.1.10. bool#

bool represents a boolean value that can be passed to attributes.


## 8.1.11. DenseConstant#

DenseConstant represents a dense constant value that can be passed to attributes.


## 8.1.12. view_type#

view_type represents a type which implements the view interface, currently this is only implemented by partition_view
but will have new implementers in future releases.



## 8.2. Operation Design Considerations#

The design of Tile IR has a set of design considerations that apply to all operations in the dialect
this section introduces some of the common design considerations that apply to all operations, or to
classes of operations generically.

## 8.2.1. Explicit Broadcast#

There are no implicit broadcast performed by operations in the Tile IR dialect all operations
that require operands of the same shape must be explicitly broadcasted. For example to use the
cuda_tile.offset operation to add an offset tile to a pointer, the pointer and offset
must be reshaped or broadcasted to have the same shape using the cuda_tile.reshape
or cuda_tile.broadcast operations.


## 8.2.2. Distinct Floating-Point and Integer Operations#

Numeric ooerations are split across integer and floating-point types due to differences in flags such as rounding modes, NaN handling,
and fast math.
For example, the cuda_tile.addf operation supports a rounding attribute, but the addi operation does not.


## 8.2.3. Explicit Overflow Annotations#

Some operations such as cuda_tile.addi support an explicit overflow annotation that expresses the expected overflow behavior
of the operation.
These attributes serve as assumptions that an implementation may use to reason about the operation. It is the responsibility of the code generator
to ensure that the operation respects these assumptions dynamically during execution.
We recommend that generators of Tile IR programs utilize these annotations to help the implementation reason about the overflow behavior of the
operation, enabling extra optimization opportunities.



## 8.3. Core#

## 8.3.1. cuda_tile.broadcast#

Broadcast tile to new shape
cuda_tile.broadcast %source



Parameters#

source (tile) - The tile to broadcast. 13.1



Results#

result (tile) - The broadcasted tile. 13.1



Description#

The broadcast operation expands each unary (1) dimension in the input tile
by duplicating the data along that dimension.
Expansion happens only for dimensions of size one that are stretched or "copied" to match
the size of the dimension implied by the result type of the operation. The operation
does not change the rank of the source tile.  Any change to the rank of the source tile
must be made using reshape-like operations before broadcasting.



Constraints#

The operation is conditionally speculatablebased on the specific operands and attributes.
The operation may be speculatively executed without side effects.
The operation is pure and does not perform any memory side effects.
source and result must have the same element type (tile).
source and result must have the same rank.




## 8.3.2. cuda_tile.cat#

Concatenate tiles along specified dimension
cuda_tile.cat %lhs %rhs %dim



Parameters#

lhs (tile) - The left hand side operand. 13.1
rhs (tile) - The right hand side operand. 13.1
dim (i64) - The dimension along which to concatenate. 13.1



Results#

result (tile) - The concatenated result tile. 13.1



Description#

The cat operation concatenates the two input tiles. The input tiles must have the same shape
in all but the concatenating dimension. Concatenation happens along the dimension specified by the
the attribute dim the resulting dimension is the sum of the the two input tiles concatenating
dimension.

\[\begin{split}\text{cat}(x, y, dim_{cat})[ \vec{i} ] =
  \begin{cases}
    x[..., i_{cat}, ..., i_n] & \text{if } i_{cat} < d_{cat} \\
    y[..., i_{cat} - d_{cat}, ..., i_n] & \text{if } i_{cat} \geq d_{cat}
  \end{cases}\end{split}\]



Constraints#

The operation is conditionally speculatablebased on the specific operands and attributes.
The operation may be speculatively executed without side effects.
The operation is pure and does not perform any memory side effects.
lhs, rhs and result must have the same rank.
lhs, rhs and result must have the same element type (tile).



Examples#
// A valid invocation of cat.
%0 = cat %arg0, %arg1 dim = 1
  : tile<2x4xf32>, tile<2x4xf32> -> tile<2x8xf32>

// >>> %arg0 = tile([[ A, B, C ],
//                   [ D, E, F ]])
// >>> %arg1 = tile([[ 1, 2, 3 ],
//                   [ 4, 5, 6 ]])
// >>> %0 = tile([[ A, B, C, 1, 2, 3 ],
//                [ D, E, F, 4, 5, 6 ]])

// A valid invocation of cat.
%1 = cat %arg0, %arg1 dim = 0
  : tile<2x4xf32>, tile<2x4xf32> -> tile<4x4xf32>

// >>> %arg0 = tile([[ A, B, C ],
//                   [ D, E, F ]])
//
// >>> %arg1 = tile([[ 1, 2, 3 ],
//                   [ 4, 5, 6 ]])
//
// >>> %1 = tile([[ A, B, C ],
//                [ D, E, F ],
//                [ 1, 2, 3 ],
//                [ 4, 5, 6 ]])


See cuda_tile.cat_0 for the full example listing.


## 8.3.3. cuda_tile.cmpf#

Element-wise floating-point comparison
cuda_tile.cmpf %comparison_predicate %comparison_ordering %lhs %rhs



Parameters#

comparison_predicate (ComparisonPredicate) - The comparison predicate. 13.1
comparison_ordering (ComparisonOrdering) - The comparison ordering. 13.1
lhs (tile<f16 | bf16 | f32 | f64>) - The left hand side operand. 13.1
rhs (tile<f16 | bf16 | f32 | f64>) - The right hand side operand. 13.1



Results#

result (tile<i1>) - The result of the comparison. 13.1



Description#

The cmpf operation is a generic comparison for float-like types. The
operands must have the same shape and type, and this type must be a float type.
The result is 1 if the comparison is true and 0 otherwise. The comparison is
performed element-wise and the element of the result indicates whether the
comparison is true for the operand elements with the same indices as those of
the result.

\[\begin{split}\text{cmpf}(x, y, \text{pred})_i = \begin{cases}
  1 & \text{if } x_i \text{ pred } y_i \\
  0 & \text{otherwise}
\end{cases}\end{split}\]
The comparison_predicate attribute specifies the kind of comparison to be performed.

equal - Equal comparison.
not_equal - Not equal comparison.
less_than - Less than comparison.
less_than_or_equal - Less than or equal comparison.
greater_than - Greater than comparison.
greater_than_or_equal - Greater than or equal comparison.

The comparison_ordering attribute specifies the kind of ordering to be performed in the comparison operation.

unordered - Unordered comparison.
ordered - Ordered comparison.



Constraints#

The operation is conditionally speculatablebased on the specific operands and attributes.
The operation may be speculatively executed without side effects.
The operation is pure and does not perform any memory side effects.
lhs and rhs must have the same shape and element type (tile<f16 | bf16 | f32 | f64>).
Result type has i1 element type and same shape as operands
The operation's result type may be inferred from its operands and attributes.



Examples#
%lhs0 = constant <f16: 0.0> : tile<f16>
%rhs0 = constant <f16: 0.0> : tile<f16>

// Custom form of scalar "ordered equal" comparison.
%x0 = cmpf equal ordered %lhs0, %rhs0 : tile<f16> -> tile<i1>

%lhs1 = constant <f16: 0.0> : tile<2x2xf16>
%rhs1 = constant <f16: 0.0> : tile<2x2xf16>

// Custom form of scalar "unordered less than" comparison.
%x2 = cmpf less_than unordered %lhs1, %rhs1 : tile<2x2xf16> -> tile<2x2xi1>

%lhs2 = constant <f64: 0.0> : tile<2x2xf64>
%rhs2 = constant <f64: 0.0> : tile<2x2xf64>


See cuda_tile.cmpf_0 for the full example listing.


## 8.3.4. cuda_tile.cmpi#

Element-wise integer comparison
cuda_tile.cmpi %comparison_predicate %lhs %rhs %signedness



Parameters#

comparison_predicate (ComparisonPredicate) - The comparison predicate. 13.1
lhs (tile<i1 | i8 | i16 | i32 | i64>) - The left hand side operand. 13.1
rhs (tile<i1 | i8 | i16 | i32 | i64>) - The right hand side operand. 13.1
signedness (Signedness) - Interpret integer(s) as signed or unsigned 13.1



Results#

result (tile<i1>) - The result of the comparison. 13.1



Description#

The cmpi operation is a generic comparison for integer-like types. The
operands must have the same shape and type, and this type must be an integer type.
The result type has i1 element type and the same shape as the operands.
The result is 1 if the comparison is true and 0 otherwise. The comparison is
performed element-wise and the element of the result indicates whether the
comparison is true for the operand elements with the same indices as those of
the result.

\[\begin{split}\text{cmpi}(x, y, \text{pred})_i = \begin{cases}
  1 & \text{if } x_i \text{ pred } y_i \\
  0 & \text{otherwise}
\end{cases}\end{split}\]
The comparison_predicate attribute specifies the kind of comparison to be performed.

equal - Equal comparison.
not_equal - Not equal comparison.
less_than - Less than comparison.
less_than_or_equal - Less than or equal comparison.
greater_than - Greater than comparison.
greater_than_or_equal - Greater than or equal comparison.

The signedness attribute specifies the signedness of operand(s).

unsigned - Treat the operands as unsigned integers.
signed - Treat the operands as signed integers.



Constraints#

The operation is conditionally speculatablebased on the specific operands and attributes.
The operation may be speculatively executed without side effects.
The operation is pure and does not perform any memory side effects.
lhs and rhs must have the same shape and element type (tile<i1 | i8 | i16 | i32 | i64>).
Result type has i1 element type and same shape as operands
The operation's result type may be inferred from its operands and attributes.



Examples#
%lhs0 = constant <i16: 0> : tile<i16>
%rhs0 = constant <i16: 0> : tile<i16>

// Scalar "signed less than" comparison.
%x0 = cmpi less_than %lhs0, %rhs0, signed : tile<i16> -> tile<i1>

%lhs1 = constant <i64: 0> : tile<2x2xi64>
%rhs1 = constant <i64: 0> : tile<2x2xi64>

// Tile equality comparison.
// There is no difference between "signed" and "unsigned" when performing equality and inequality comparison.
%x1 = cmpi equal %lhs1, %rhs1, signed : tile<2x2xi64> -> tile<2x2xi1>


See cuda_tile.cmpi_0 for the full example listing.


## 8.3.5. cuda_tile.constant#

Construct a constant tile
cuda_tile.constant %value



Parameters#

value (DenseConstant) - The constant value to create. 13.1



Results#

result (tile<i1 | i8 | i16 | i32 | i64 | f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The constant tile. 13.1



Description#

The constant operation creates a tile initialized by $value.
There are two main forms of using the operation:

One where the value is a single constant specified by <D: c>
and the tile is filled with identical values for all elements with element type D.
One where the value is a list of constants specified by dense<D: [c0, c1, c2, ...]>
and the constant value's shape must match the tile's shape with the element type D.

The annotated type of the tile constrains its rank, shape, and element type.



Constraints#

The operation has no operands and may be constant folded.
The operation is conditionally speculatablebased on the specific operands and attributes.
The operation may be speculatively executed without side effects.
The operation is pure and does not perform any memory side effects.
value and result must have the same shape and element type (DenseConstant).
The operation's result type may be inferred from its operands and attributes.



Examples#
%c0 = constant <i32: 0> : tile<i32>
%c1 = constant <i64: 1> : tile<i64>
%c2 = constant <i32: [0, 1, 2, 3]> : tile<4xi32>
%c3 = constant <f32: 0.0> : tile<2x4xf32>
%c4 = constant <f64: [0.0, 1.0, 2.0, 3.0]> : tile<4xf64>


See cuda_tile.constant_0 for the full example listing.


## 8.3.6. cuda_tile.entry#

Define a tile kernel
cuda_tile.entry %sym_name %function_type %arg_attrs %res_attrs %optimization_hints



Parameters#

sym_name (Symbol) - The name of the function. 13.1
function_type (Type) - The type of the function. 13.1
arg_attrs (Attributes) - The argument attributes of the function: none of these are supported by TileIR at the moment. 13.1
res_attrs (Attributes) - The result attributes of the function: none of these are supported by TileIR at the moment. 13.1
optimization_hints (OptimizationHints) - Compiler architecture-specific optimization hints 13.1



Results#

No results.



Description#

The entry operation defines a tile kernel; a kernel is a function that can
serve as the program entry point. It has a unique name per-module. A kernel can
not return any value. It must be launched from the host side using cuLaunchKernel
or similar CUDA runtime API functions.
Tile kernels require that the user specifies the 3-d grid dimensions at launch which
defines the number of tile blocks (or kernel instances) that will execute the kernel
in parallel.
For detailed semantics of tile kernels see Tile Kernel.
The optimization_hints attribute provides architecture-specific compiler hints in the form of nested dictionaries.
The hints are specified for each architecture (e.g., sm_100, sm_120) and for each architecture the user can specify
specific hints for each operation.

num_cta_in_cga - suggest the number of CTAs in a CGA (which must be the power of 2 less than or equal to 16) for cuda_tile.entry.
allow_tma - suggest whether to use TMA for cuda_tile.load_view_tko and cuda_tile.store_view_tko.
latency - latency hint for cuda_tile.load_view_tko and cuda_tile.store_view_tko.

For example they can be annotated as:
optimization_hints=<
  sm_100 = {num_cta_in_cga = 8},
  sm_120 = {num_cta_in_cga = 16}
>




Constraints#

The operation must be a symbol in the global symbol table.
The operation must implement callable target interface.
The operation must implement function-like behavior interface.
The region must not capture SSA values defined above the operation.
The operation must provide custom parsing and printing methods.
Each provided region must contain exactly one block.




## 8.3.7. cuda_tile.extract#

Extract a subtile from a tile
cuda_tile.extract %source %indices



Parameters#

source (tile) - The source tile to extract from. 13.1
indices (Variadic<tile<i32>>) - The indices of the slice to extract. 13.1



Results#

result (tile) - The extracted subtile. 13.1



Description#

The extract operation extracts a subtile from the given source tile.
The shape of the result tile must divide the shape of the source tile
evenly e.g., tile<4xf32> is a valid extraction from tile<8xf32>, but
tile<3xf32> is not.
The $indices indicate the number of the slice to extract, but importantly not the offsets
used to construct the subtile for extraction. The semantics of extract means that only
full size slices can be extracted.
Slices of a source tile with the same shape are non-overlapping by definition for
unique indices.
The indices operands are interpreted as unsigned integers.

Warning
If the indices specify a non-existent (i.e., out-of-bounds) slice, the
behavior of the operation is undefined.



Constraints#

The operation is conditionally speculatablebased on the specific operands and attributes.
The operation may be speculatively executed without side effects.
The operation is pure and does not perform any memory side effects.
source and result must have the same rank.



Examples#
// Extract a subtile from %t at dim_0 = [4;8) and dim_1 = [4;6).
%c1 = constant <i32: 1> : tile<i32>
%c2 = constant <i32: 2> : tile<i32>
%t = constant <f32: 0.0> : tile<32x8xf32>
// Valid indices are: [ {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3} ]
%0 = extract %t[%c1, %c2]
    : tile<32x8xf32> -> tile<4x2xf32>


See cuda_tile.extract_0 for the full example listing.


## 8.3.8. cuda_tile.get_global#

Get a pointer to a global variable
cuda_tile.get_global %name



Parameters#

name (Symbol) - The name of the global variable. 13.1



Results#

result (tile<ptr>) - The result of the get_global operation. 13.1



Description#

The get_global operation returns a pointer to the specified global
variable. A global variable is a form of static global memory allocation that can
be declared using the cuda_tile.global operation.
The element type of the returned pointer will be of the same type as the
element type of the declared global variable.
For detailed semantics of global variables see Global Variable.



Constraints#

The operation is conditionally speculatablebased on the specific operands and attributes.
The operation may be speculatively executed without side effects.
The operation is pure and does not perform any memory side effects.



Examples#
global @val <f32: [0.1, 0.2, 0.3, 0.4]> : tile<4xf32>

entry @example() {
  %ptr = get_global @val : tile<ptr<f32>>
  return
}


See cuda_tile.get_global_0 for the full example listing.


## 8.3.9. cuda_tile.get_num_tile_blocks#

Get total number of tile blocks
cuda_tile.get_num_tile_blocks



Parameters#

No parameters.



Results#

gridSize_x (tile<i32>) - The number of tile blocks in dimension x. 13.1
gridSize_y (tile<i32>) - The number of tile blocks in dimension y. 13.1
gridSize_z (tile<i32>) - The number of tile blocks in dimension z. 13.1



Description#

The get_num_tile_blocks operation queries the total number of tile blocks
in the form of a 3-tuple specifying the extent of each grid dimension.
A tile id is a coordinate in 3-space and therefore the must also be a 3-tuple containing
the extent of each dimension: x, y and z.
When launching 1- or 2-dimensional grids, the unspecified dimensions will have a cardinality of 1.
For example if the grid used to launch the kernel is (1024, 1024) then the
result of this operation will be (1024, 1024, 1).

Note
Grid Dimension Limitation: Grid dimensions are limited to 2^24-1 (16,777,215)
per axis. Larger dimensions may result in incorrect tile block ID calculations. Use multiple
kernel launches for larger workloads.



Constraints#

The operation is conditionally speculatablebased on the specific operands and attributes.
The operation may be speculatively executed without side effects.
The operation is pure and does not perform any memory side effects.
The operation's result type may be inferred from its operands and attributes.



Examples#
entry @example() {
  %x, %y, %z = get_num_tile_blocks : tile<i32>
  // print "x: %, y: %, z: %\n", %x, %y, %z : tile<i32>, tile<i32>, tile<i32>
}


See cuda_tile.get_num_tile_blocks_0 for the full example listing.


## 8.3.10. cuda_tile.get_tile_block_id#

Get the currently executing tile block coordinates
cuda_tile.get_tile_block_id



Parameters#

No parameters.



Results#

blockId_x (tile<i32>) - The tile block ID for dimension x. 13.1
blockId_y (tile<i32>) - The tile block ID for dimension y. 13.1
blockId_z (tile<i32>) - The tile block ID for dimension z. 13.1



Description#

get_tile_block_id returns a 3-d tile block coordinates (or ID) of the currently
executing tile block.
A tile ID has three dimensions: x, y, and z. This operation returns all
three of them simultaneously. The value of each dimension returned by this
operation is between 0 (including) and the value returned by get_num_tile_blocks
for the respective axis (excluding), represented by the inclusive interval
[0, get_num_tile_blocks(dim) - 1] . Grid dimensions unspecified at kernel
launch (i.e., a 1-d or 2-d grid) will always be 0 for all tile blocks.

Note
Grid Dimension Limitation: Grid dimensions are limited to 2^24-1 (16,777,215)
per axis. Larger dimensions may result in incorrect tile block ID calculations. Use multiple
kernel launches for larger workloads.



Constraints#

The operation is conditionally speculatablebased on the specific operands and attributes.
The operation may be speculatively executed without side effects.
The operation is pure and does not perform any memory side effects.
The operation's result type may be inferred from its operands and attributes.



## 8.3.11. cuda_tile.global#

Allocate static global memory
cuda_tile.global %sym_name %value %alignment



Parameters#

sym_name (Symbol) - The name of the global variable. 13.1
value (DenseConstant) - The value to initialize the allocation with. 13.1
alignment (i64) - The alignment of the buffer. 13.1



Results#

No results.



Description#

The global operation statically allocates a mutable 1-dimensional location in global
memory and initializes it using value. The initialization of the allocation is performed
at CUDA module
load time. The lifetime of the allocation is the same as the lifetime of the module.
The allocation may be read or written to by first using cuda_tile.get_global to obtain a pointer to the
the memory and then read using cuda_tile.load_ptr_tko or written to using cuda_tile.store_ptr_tko.
The initial values are stored in memory in linear order, so the pointer returned by cuda_tile.get_global
points to the first element, and offsetting the pointer by x would allow to load element at position x.
global operations must be directly nested within the Tile IR module. They cannot be defined inside functions.
As globals are defined at the module scope their names are globally unique symbols and must not collide with any other
symbol in the module.
For more detailed semantics of global variables see Global Variable.



Constraints#

The operation must be a symbol in the global symbol table.



Examples#
global @val alignment = 128 <f32: [0.1, 0.2, 0.3, 0.4]> : tile<4xf32>
entry @example() {}


See cuda_tile.global_0 for the full example listing.


## 8.3.12. cuda_tile.iota#

Generate a 1-d tile range from 0 to n-1
cuda_tile.iota



Parameters#

No parameters.



Results#

result (tile<i1 | i8 | i16 | i32 | i64>) - The result of the iota operation. 13.1



Description#

The iota operation generates a 1-d tile with a sequence of integer
values. The starting value is 0 and the stride is 1. If the shape of
the result tile is (n), then the generated values are [0, n - 1].

\[\text{iota}(n)_i = i \quad \text{for } i \in [0, n-1]\]
The result values should be interpreted as unsigned integers.

Note
The number of elements in the result tile must not exceed
the maximum value that the element type can express.



Constraints#

The operation is conditionally speculatablebased on the specific operands and attributes.
The operation may be speculatively executed without side effects.
The operation is pure and does not perform any memory side effects.



## 8.3.13. cuda_tile.mmaf#

Floating-point matrix-multiply-accumulate
cuda_tile.mmaf %lhs %rhs %acc



Parameters#

lhs (tile<f16 | bf16 | f32 | f64 | tile<tf32>>) - The left hand side matrix operand. 13.1
rhs (tile<f16 | bf16 | f32 | f64 | tile<tf32>>) - The right hand side matrix operand. 13.1
acc (tile<f16 | f32 | f64>) - The accumulator matrix operand. 13.1



Results#

result (tile<f16 | f32 | f64>) - The result matrix after multiplication and accumulation. 13.1



Description#

The mmaf operation implements an MMA (matrix-multiply-accumulate) operation for floating-point tiles.
It performs matrix multiplication on the floating-point tiles lhs and rhs, then adds the tile acc to the result.
lhs, rhs, and acc must be 2D tiles or 3D tiles. The latter case
indicates a batched matrix multiplication.

\[\text{mmaf}(A, B, C)_{ij} = \sum_{k=0}^{K-1} A_{ik} \times B_{kj} + C_{ij}\]
The types of all operands must be a supported combination (see mmaf Supported Data Types).
Shapes must be a valid matrix multiplication configuration. Unbatched (2D)
MMA expects the operands lhs, rhs, and acc to have shapes M x K,
K x N, and M x N (respectively). Batched (3D) MMA expects the operands
to have shapes B x M x K, B x K x N, and B x M x N (respectively).
The table below shows the supported output types for each possible mmaf input type. Input operands must be of the same element type.

mmaf Supported Data Types#

Input Type Supported Output Types


f8E4M3FN f16 or f32

f8E5M2 f16 or f32

f16 f16 or f32

bf16 f32

tf32 f32

f32 f32

f64 f64




Constraints#

The operation is conditionally speculatablebased on the specific operands and attributes.
The operation may be speculatively executed without side effects.
The operation is pure and does not perform any memory side effects.
acc and result must have the same shape and element type (tile<f16 | f32 | f64>).
lhs and rhs must have the same element type (tile<f16 | bf16 | f32 | f64 | tile<tf32>>).
lhs, rhs and acc must have the same rank.
The operation's result type may be inferred from its operands and attributes.



Examples#
%lhs0 = constant <f16: 0.0> : tile<4x8xf16>
%rhs0 = constant <f16: 0.0> : tile<8x2xf16>
%acc0 = constant <f32: 0.0> : tile<4x2xf32>

%0 = mmaf %lhs0, %rhs0, %acc0
    : tile<4x8xf16>, tile<8x2xf16>,
      tile<4x2xf32>

%lhs1 = constant <f16: 0.0> : tile<2x4x8xf16>
%rhs1 = constant <f16: 0.0> : tile<2x8x2xf16>
%acc1 = constant <f32: 0.0> : tile<2x4x2xf32>

%1 = mmaf %lhs1, %rhs1, %acc1
    : tile<2x4x8xf16>, tile<2x8x2xf16>,
      tile<2x4x2xf32>


See cuda_tile.mmaf_0 for the full example listing.


## 8.3.14. cuda_tile.mmai#

Integer matrix-multiply-accumulate
cuda_tile.mmai %lhs %rhs %acc %signedness_lhs %signedness_rhs



Parameters#

lhs (tile<i8>) - The left hand side matrix operand. 13.1
rhs (tile<i8>) - The right hand side matrix operand. 13.1
acc (tile<i32>) - The accumulator matrix operand. 13.1
signedness_lhs (Signedness) - The signedness of the lhs operand. 13.1
signedness_rhs (Signedness) - The signedness of the rhs operand. 13.1



Results#

result (tile<i32>) - The result matrix after multiplication and accumulation. 13.1



Description#

The mmai operation implements an MMA (matrix-multiply-accumulate) operation for integer tiles.
It performs matrix multiplication on the integer tiles lhs and rhs, then adds the tile acc to the result.
lhs, rhs, and acc must be 2D tiles or 3D tiles. The latter case indicates a batched matrix multiplication.

\[\text{mmai}(A, B, C)_{ij} = \sum_{k=0}^{K-1} A_{ik} \times B_{kj} + C_{ij}\]
Input tiles lhs and rhs must be of integer type i8. The signedness of
lhs and rhs are specified separately by the signedness_lhs and
signedness_rhs attributes, respectively. The accumulator tile acc must be
of type i32 and is always interpreted as signed. The output tile result
is of type i32 and is always interpreted as signed.
Shapes must be a valid matrix multiplication configuration. Unbatched (2D)
MMA expects the operands lhs, rhs, and acc to have shapes M x K,
K x N, and M x N (respectively). Batched (3D) MMA expects the operands
to have shapes B x M x K, B x K x N, and B x M x N (respectively).
The signedness attribute specifies the signedness of operand(s).

unsigned - Treat the operands as unsigned integers.
signed - Treat the operands as signed integers.



Constraints#

The operation is conditionally speculatablebased on the specific operands and attributes.
The operation may be speculatively executed without side effects.
The operation is pure and does not perform any memory side effects.
acc and result must have the same shape and element type (tile<i32>).
lhs and rhs must have the same element type (tile<i8>).
lhs, rhs and acc must have the same rank.
The operation's result type may be inferred from its operands and attributes.



Examples#
%lhs0 = cuda_tile.constant <i8: 0> : tile<4x8xi8>
%rhs0 = cuda_tile.constant <i8: 0> : tile<8x2xi8>
%acc0 = cuda_tile.constant <i32: 0> : tile<4x2xi32>

%0 = mmai %lhs0, %rhs0, %acc0 signed signed
    : tile<4x8xi8>, tile<8x2xi8>,
      tile<4x2xi32>

%lhs1 = cuda_tile.constant <i8: 0> : tile<2x4x8xi8>
%rhs1 = cuda_tile.constant <i8: 0> : tile<2x8x2xi8>
%acc1 = cuda_tile.constant <i32: 0> : tile<2x4x2xi32>

%1 = mmai %lhs1, %rhs1, %acc1 unsigned unsigned
    : tile<2x4x8xi8>, tile<2x8x2xi8>,
      tile<2x4x2xi32>


See cuda_tile.mmai_0 for the full example listing.


## 8.3.15. cuda_tile.module#

Top-level module containing a series of defined items.
cuda_tile.module %sym_name



Parameters#

sym_name (Symbol) - The name of the module. 13.1



Results#

No results.



Description#

A module operation represents a single compilation unit and contains
zero or more items (global variables, functions, or kernels).
For detailed description of the semantics of modules, and the full definition of each item type see
Modules.
The module operation is the top-level operation in a Tile IR module and must
contain only Tile IR operations and no other dialects.



Constraints#

The region must not capture SSA values defined above the operation.
The operation must provide custom parsing and printing methods.
All regions must have zero arguments.
Each provided region must contain exactly one block.
The operation must define a symbol scope.
The region must not require explicit terminator operations.
The operation must specify whether regions are SSACFG or Graph kind.
The operation must contain only dataflow graph regions.



## 8.3.16. cuda_tile.offset#

Offsets a tile of pointers
cuda_tile.offset %ptr %offset



Parameters#

ptr (ptr) - The base pointer tile to advance. 13.1
offset (tile<i1 | i8 | i16 | i32 | i64>) - The offset tile to add to the pointer. 13.1



Results#

result (ptr) - The resulting pointer tile after advancement. 13.1



Description#

offset advances a tile of pointers. It takes ptr as base
and offset as increment, and performs element-wise addition of
ptr by offset:

\[\text{offset}(\text{ptr}, \text{offset})_i = \text{ptr}_i + \text{offset}_i \times \text{bitwidth}\]
result[i,j] = ptr[i,j] + offset[i,j] * bitwidth


ptr is interpreted as an unsigned integer. offset is
interpreted as a signed integer. bitwidth is the storage bitwidth
of the pointee type. The multiplication must not overflow (wrap-around) in
a signed sense. The addition must not overflow (wrap-around) in an unsigned
sense. In case of an overflow, the result is undefined.



Constraints#

The operation is conditionally speculatablebased on the specific operands and attributes.
The operation may be speculatively executed without side effects.
The operation is pure and does not perform any memory side effects.
The operation must apply element-wise to its operands.
ptr, offset and result must have the same shape.
result and ptr must have the same shape and element type (ptr).
The operation's result type may be inferred from its operands and attributes.



## 8.3.17. cuda_tile.pack#

Pack a tile into a byte array
cuda_tile.pack %source



Parameters#

source (tile<i1 | i8 | i16 | i32 | i64 | f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The input tile. 13.2



Results#

result (tile<i8>) - The packed tile. 13.2



Description#

The pack operation takes a rank-1 numeric tile and produces a rank-1 tile<i8>.
Similar to bitcast, underlying bit-values are not changed. However, pack does not
operate elementwise, instead reinterpreting the entire tile as a byte array.
Input and output tiles must be rank-1 to eliminate packing ambiguity. The size of the output
tile must match the number of bytes in the input tile.



Constraints#

The operation is conditionally speculatablebased on the specific operands and attributes.
The operation may be speculatively executed without side effects.
The operation is pure and does not perform any memory side effects.
source and result must have the same rank.



Examples#
%arg0 = constant <f16: 0.0> : tile<64xf16>
%0 = pack %arg0 : tile<64xf16> -> tile<128xi8>


See cuda_tile.pack_0 for the full example listing.
%arg0 = constant <f4E2M1FN: 0.0> : tile<64xf4E2M1FN>
%0 = pack %arg0 : tile<64xf4E2M1FN> -> tile<32xi8>


See cuda_tile.pack_1 for the full example listing.


## 8.3.18. cuda_tile.permute#

Permute tile dimensions
cuda_tile.permute %source %permutation



Parameters#

source (tile) - The input tile. 13.1
permutation (Array<i32>) - The permutation of the dimensions. 13.1



Results#

result (tile) - The permuted tile. 13.1



Description#

Permute the dimensions of the input tile source according to the permutation array.
The permutation array is a list of integers that specify the new order of the dimensions.
For example, if the input tile has shape [2, 4, 8], and the permutation is [2, 0, 1],
the output tile will have shape [8, 2, 4].
This operation logically is a change in the indexing of the tile.



Constraints#

The operation is conditionally speculatablebased on the specific operands and attributes.
The operation may be speculatively executed without side effects.
The operation is pure and does not perform any memory side effects.
source and result must have the same element type (tile).
source and result must have the same rank.



Examples#
%arg0 = constant <f16: 0.0> : tile<2x4x8xf16>
%0 = permute %arg0 [2, 0, 1] : tile<2x4x8xf16> -> tile<8x2x4xf16>


See cuda_tile.permute_0 for the full example listing.


## 8.3.19. cuda_tile.reduce#

Variadic tile reduction across dimensions
cuda_tile.reduce %operands %dim %identities



Parameters#

operands (Variadic<tile>) - The set of tiles to reduce. 13.1
dim (i32) - The index of the dimension to perform reduction on. 13.1
identities (Array) - The reduction identities for each operand. 13.1



Results#

results (Variadic<tile>) - The set of reduced tiles. 13.1



Description#

The reduce operation applies a custom reduction function along a specified dimension of
one or more input tiles, producing the same number of output tiles.
The reduction function must be an associative operation defined within the reduce
operation's region. A single reduction operation can reduce over any number of input tiles in
parallel, producing a reduced output tile for each.
All input tiles must have the same shape. The output tiles will have a matching shape in every
dimension except the one being reduced, which is removed.
For each input tile, a constant identity value must be provided that matches the element type of
the input tile. Identity i of identities corresponds to input tile
i of operands. The correct identity value is a property of the reduction
function in the body. (For example, if the reduction function performs min,
the identity is +inf, while if the reduction function performs a sum,
the identity is 0.)
The reduction function must expect 2N arguments, where N is the number of input tiles.
Each pair of reduction arguments 2i and 2i+1 will correspond to the i-th input tile.
The first argument of each pair is an element of the input tile; the second is the accumulator from all
prior reductions along the specified dimension. This second value might be input element, the identity value,
or the result of a previous reduction iteration. The reduction function should yield the new accumulator value
for each input tile.

Note
There are no guarantees on the order of element reduction along the specified dimension.
However, the result is deterministic across different runs of the same kernel on the same device.



Constraints#

The operation must provide custom parsing and printing methods.
The operation only has an effect if and only if it the region's operation have an effect.
All operands must have identical shapes.
Each provided region must contain exactly one block.



Examples#
%input = constant <f32: 0.0> : tile<8xf32>
%0 = reduce %input dim=0 identities=[0.000000e+0 : f32] : tile<8xf32> -> tile<f32>
  (%input_arg: tile<2xf32>, %input_accum: tile<f32>) {
    %add_result = addf %input_arg, %input_accum : tile<f32>
    yield %add_result : tile<f32>
  }


See cuda_tile.reduce_0 for the full example listing.
%input = constant <f32: 0.0> : tile<8x64xf32>
%0 = reduce %input dim=0 identities=[0.000000e+0 : f32] : tile<8x64xf32> -> tile<8xf32>
  (%input_arg: tile<f32>, %input_accum: tile<f32>) {
    %add_result = addf %input_arg, %input_accum : tile<f32>
    yield %add_result : tile<f32>
  }


See cuda_tile.reduce_1 for the full example listing.


## 8.3.20. cuda_tile.reshape#

Reshape tile dimensions
cuda_tile.reshape %source



Parameters#

source (tile) - The source tile to reshape. 13.1



Results#

result (tile) - The reshaped tile. 13.1



Description#

The reshape operation changes the shape of the source operand. reshape is
only a change in the indexing of the tile. The number of elements and element type
must remain unchanged.
0-d tiles (i.e., scalars) contain precisely one element and thus are the one exception
where a 0-d tile can be reshaped to shape where the size(shape) == 1.
Conceptually reshaping a tile is equivalent to first creating a 1-d tile from the data of the source assuming
a row-major layout and then converting the 1-d tile into the new shape in a row-major layout.



Constraints#

The operation is conditionally speculatablebased on the specific operands and attributes.
The operation may be speculatively executed without side effects.
The operation is pure and does not perform any memory side effects.
source and result must have the same element type (tile).



Examples#
%cst = constant <i8: 0> : tile<i8>
%0 = reshape %cst
    : tile<i8> -> tile<1x1x1xi8>

%t = constant <f32: 0.0> : tile<8x2xf32>
%1 = reshape %t
    : tile<8x2xf32> -> tile<2x2x4x1xf32>


See cuda_tile.reshape_0 for the full example listing.
  %cst = constant <i32: [[0, 1, 2, 3], [4, 5, 6, 7]]>
      : tile<2x4xi32>
  %r0 = reshape %cst
: tile<2x4xi32> -> tile<2x2x2xi32>

// Step 1: Turn source into 1D tile. Use row-major by convention.
// %tmp: [0, 1, 2, 3, 4, 5, 6, 7]
%tmp = reshape %cst
    : tile<2x4xi32> -> tile<8xi32>

// Step 2: Turn 1D tile into result tile. Use row-major by convention.
// %r: [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
%r1 =  reshape %tmp
        : tile<8xi32> -> tile<2x2x2xi32>


See cuda_tile.reshape_1 for the full example listing.


## 8.3.21. cuda_tile.scan#

A parallel prefix sum operation
cuda_tile.scan %operands %dim %reverse %identities



Parameters#

operands (Variadic<tile>) - The a set of tiles to scan. 13.1
dim (i32) - The index of the dimension along which to scan. 13.1
reverse (bool) - Whether to scan in reverse order. 13.1
identities (Array) - The identities of the scan operation. 13.1



Results#

results (Variadic<tile>) - The resulting tiles from the scan operation. 13.1



Description#

The scan operation computes an inclusive parallel prefix along a given
dimension of the input tiles using a binary associative function and an identity.
The scan operation applies a scan function defined over a tile of elements
for a given type, utilizing an associative operation and an identity value. It
operates on operands and identities across the specified dim,
producing new results tile values. The exact evaluation order within each
prefix is implementation-defined but the result remains deterministic across different
runs of the same kernel on the same device.

\[\text{scan}(X, \text{dim}, \text{identity}, f)_{i_1,\ldots,i_d}[j] \;=\;
 \text{fold}\!\left(f, \text{identity},
   \left(X_{i_1,\ldots,i_{\text{dim}-1}, 0, i_{\text{dim}+1},\ldots,i_d}, \ldots,
         X_{i_1,\ldots,i_{\text{dim}-1}, j, i_{\text{dim}+1},\ldots,i_d}\right)\right)\]
The scan preserves all intermediate accumulator values:

\[\begin{split}\text{result}[0] \;=\; f(\text{identity}, X[\ldots, 0, \ldots]) \\
\text{result}[1] \;=\; f(\text{result}[0], X[\ldots, 1, \ldots]) \\
\vdots \\
\text{result}[j] \;=\; f(\text{result}[j-1], X[\ldots, j, \ldots])\end{split}\]
When reverse is true, the prefix is taken in decreasing index order.
Let \(N\) be the size of the scanned dimension; then:

\[\text{scan}_{\text{rev}}(X)[j] \;=\;\
 \text{fold}\!\left(f, \text{identity},
   \left(X[\ldots, N\!-\!1,\ldots], \ldots, X[\ldots, j,\ldots]\right)\right)\]
The identities attribute is a list of identity elements for each input
tile; the identity at position i binds with the operand tile at the same
position. The correct identity is a property of the scan function in the body
(e.g., sum uses 0, prod uses 1, min uses +inf, max uses -inf).
The body region represents the binary associative operation. The region must
contain Tile IR operations with 0-rank tile types. Region arguments are bound in
operand order as [op_0_current_iter, op_0_prev_iter, op_1_current_iter, op_1_prev_iter, ...],
where op_i_current_iter is the current element along dim and
op_i_prev_iter is the running accumulator for operand i. On the first
step, the accumulator is the corresponding identity element.

Note
Associativity of the binary operation permits the compiler to reorganize the
applications of the operation to achieve efficient parallel prefix scans on the GPU.



Warning
The scan operation is restricted to only support single tile input.



Constraints#

The operation must provide custom parsing and printing methods.
The operation only has an effect if and only if it the region's operation have an effect.
All operands must have identical shapes.
Each provided region must contain exactly one block.



Examples#
%input = constant <f32: 0.0> : tile<8x16xf32>
%result = scan %input dim=1 reverse=false identities=[1.0 : f32] : tile<8x16xf32> -> tile<8x16xf32>
(%acc: tile<f32>, %elem: tile<f32>) {
  %prod = mulf %acc, %elem rounding<nearest_even>: tile<f32>
  yield %prod : tile<f32>
}


See cuda_tile.scan_0 for the full example listing.


## 8.3.22. cuda_tile.select#

Select values based on condition
cuda_tile.select %cond %val_if_true %val_if_false



Parameters#

cond (tile<i1>) - The condition tile. 13.1
val_if_true (tile) - The value if true tile. 13.1
val_if_false (tile) - The value if false tile. 13.1



Results#

result (tile) - The tile of selected values. 13.1



Description#

The select op chooses values based on the binary conditions supplied as
the cond operand. The val_if_true operand contains the value(s) to use
if the condition is 1. The val_if_false operand contains the value(s) to
use if the condition is 0. The choice is made element-wise according to the
values in the condition tile.

\[\begin{split}\text{select}(\text{cond}, x, y)_i = \begin{cases}
  x_i & \text{if } \text{cond}_i = 1 \\
  y_i & \text{if } \text{cond}_i = 0
\end{cases}\end{split}\]
All tiles must have the same shape. The tiles val_if_true,
val_if_false, and the result must have the same element type. The cond
tile must be a tile of i1 values.



Constraints#

The operation is conditionally speculatablebased on the specific operands and attributes.
The operation may be speculatively executed without side effects.
The operation is pure and does not perform any memory side effects.
val_if_true, val_if_false and result must have the same shape and element type (tile).
The operation's result type may be inferred from its operands and attributes.



## 8.3.23. cuda_tile.unpack#

Unpack a byte array into a tile
cuda_tile.unpack %source



Parameters#

source (tile<i8>) - The input i8 tile. 13.2



Results#

result (tile<i1 | i8 | i16 | i32 | i64 | f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The output unpacked tile. 13.2



Description#

The unpack operation takes a rank-1 tile<i8> and produces a rank-1 numeric tile.
Similar to bitcast, underlying bit-values are not changed. However, unpack does not
operate elementwise, instead reinterpreting the entire tile as a numeric tile of different element type.
Input and output tiles must be rank-1 to eliminate unpacking ambiguity. The size of the input
tile must match the number of bytes in the output tile.



Constraints#

The operation is conditionally speculatablebased on the specific operands and attributes.
The operation may be speculatively executed without side effects.
The operation is pure and does not perform any memory side effects.
source and result must have the same rank.



Examples#
%arg0 = constant <i8: 0> : tile<64xi8>
%0 = unpack %arg0 : tile<64xi8> -> tile<32xf16>


See cuda_tile.unpack_0 for the full example listing.
%arg0 = constant <i8: 0> : tile<64xi8>
%0 = unpack %arg0 : tile<64xi8> -> tile<128xF4E2M1FN>


See cuda_tile.unpack_1 for the full example listing.


## 8.4. Conversions#

There are no implicit type conversions in Tile IR thus we expose a set of explicit conversion operations for interconverting between types which have compatible representations
or rules for conversion.
cuda_tile.bitcast preserves the contents of the input but allows for changing of element types, cuda_tile.exti and cuda_tile.trunci change the width of integer tiles,
cuda_tile.ftoi and cuda_tile.itof convert floating-point tiles to integer tiles and vice versa, and cuda_tile.ftof converts between different floating-point types.
For more details on conversions and their rules see the individual operation's documentation.

## 8.4.1. cuda_tile.bitcast#

Bitcast a tile from one element type to another
cuda_tile.bitcast %source



Parameters#

source (tile<i1 | i8 | i16 | i32 | i64 | f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The source tile to cast. 13.1



Results#

result (tile<i1 | i8 | i16 | i32 | i64 | f16 | bf16 | f32 | f64 | fp8e4m3fn | fp8e5m2 | tf32>) - The casted tile. 13.1



Description#

The bitcast operation casts the input tile from one element type to
another without modifying the underlying bits.
Only non-pointer types of the same bit width are allowed (e.g., i32 to f32).
Pointer types must use cuda_tile.ptr_to_int or cuda_tile.int_to_ptr instead.



Constraints#

The operation is conditionally speculatablebased on the specific operands and attributes.
The operation may be speculatively executed without side effects.
The operation is pure and does not perform any memory side effects.



## 8.4.2. cuda_tile.exti#

Extend the width of an integer tile
cuda_tile.exti %from %signedness



Parameters#

from (tile<i1 | i8 | i16 | i32 | i64>) - The input integer tile to extend. 13.1
signedness (Signedness) - Interpret integer(s) as signed or unsigned 13.1



Results#

to (tile<i1 | i8 | i16 | i32 | i64>) - The extended integer tile. 13.1



Description#

The exti operation converts a tile of integers of a given width to a
strictly larger width. Zero-extension is used
for unsigned integers and sign-extension is used for signed
integers.
The signedness attribute specifies the signedness of operand(s).

unsigned - Treat the operands as unsigned integers.
signed - Treat the operands as signed integers.



Constraints#

The operation is conditionally speculatablebased on the specific operands and attributes.
The operation may be speculatively executed without side effects.
The operation is pure and does not perform any memory side effects.
from and to must have the same rank.
The operation's result type may be inferred from its operands and attributes.



[Content continues with additional operations - see full documentation for complete list]