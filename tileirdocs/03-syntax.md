# 3. Syntax — Tile IR

[Skip to main content](#main-content)

Back to top Ctrl+K

[![Tile IR - Home](../_static/nvidia-logo-horiz-rgb-blk-for-screen.svg) ![Tile IR - Home](../_static/nvidia-logo-horiz-rgb-wht-for-screen.svg)

Tile IR](../index.html)

Choose version

Search Ctrl+K

Search Ctrl+K

[![Tile IR - Home](../_static/nvidia-logo-horiz-rgb-blk-for-screen.svg) ![Tile IR - Home](../_static/nvidia-logo-horiz-rgb-wht-for-screen.svg)

Tile IR](../index.html)

Choose version

Table of Contents

-   [1. Introduction](introduction.html)
-   [2. Programming Model](prog_model.html)
-   [3. Syntax](#)
-   [4. Binary Format](bytecode.html)
-   [5. Type System](types.html)
-   [6. Semantics](semantics.html)
-   [7. Memory Model](memory_model.html)
-   [8. Operations](operations.html)
-   [9. Debug Info](debug_info.html)
-   [10. Stability](stability.html)
-   [11. Appendix](appendix.html)
-   [12. Release Notes](release_notes.html)

-   [](../index.html)
-   3. Syntax

# 3. Syntax[#](#syntax "Link to this heading")

**Tile IR** is intended to be constructed using the **Tile IR** MLIR dialect and stored as bytecode.

To enable humans to comprehend **Tile IR** bytecode programs, we provide an **unstable** textual representation based on the MLIR dialect. This textual representation has no stability guarantees and it is not intended to be used for writing **Tile IR** programs.

## 3.1. Module[#](#module "Link to this heading")

A **Tile IR** program consists of a **Tile IR** module which contains a series of items.

symbol_name := `@` identifier

cuda_tile.module @symbol_name {
    <items>*
}

## 3.2. Items[#](#items "Link to this heading")

An item is a kernel definition or a global variable definition.

<items> ::= <kernel_definition> | <global_variable_definition>

## 3.3. Globals[#](#globals "Link to this heading")

A global variable definition is a variable that is defined outside of a kernel.

global_variable_definition ::= `global` <symbol_name> `:` <type> `=` <value>

## 3.4. Kernels[#](#kernels "Link to this heading")

A kernel definition is a function that is defined inside a **Tile IR** module.

ssa_name := `%` identifier

function_signature ::= <function_parameter>*

function_parameter ::= <ssa_name> `:` <type>

<kernel_definition> ::= `func` @kernel_name `(` <function_signature> `)` `->` <function_type> {
    <kernel_body>
}

A kernel definition's body is a sequence of operations.

kernel_body ::= <operation>*

operation ::= (ssa_name `,`?)* `=` <operation_name> <ssa_name>* attribute=attribute_value : type ...

## 3.5. Types[#](#types "Link to this heading")

A type in **Tile IR** is a fixed pre-defined set of types.

element_type ::= `f32` | `f64` | `i8` | `i16` | `i32` | `i64` | `b8` | `b16` | `b32` | `b64`

type ::= `tile` `<` shape `x` element_type `>`

shape ::= `[` integer_literal (`x` integer_literal)* `]`

[

previous

2. Programming Model



](prog_model.html "previous page")[

next

4. Binary Format

](bytecode.html "next page")

On this page

-   [3.1. Module](#module)
-   [3.2. Items](#items)
-   [3.3. Globals](#globals)
-   [3.4. Kernels](#kernels)
-   [3.5. Types](#types)

[![NVIDIA](../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg) ![NVIDIA](../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

[Privacy Policy](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/) | [Your Privacy Choices](https://www.nvidia.com/en-us/about-nvidia/privacy-center/) | [Terms of Service](https://www.nvidia.com/en-us/about-nvidia/terms-of-service/) | [Accessibility](https://www.nvidia.com/en-us/about-nvidia/accessibility/) | [Corporate Policies](https://www.nvidia.com/en-us/about-nvidia/company-policies/) | [Product Security](https://www.nvidia.com/en-us/about-nvidia/product-security/) | [Contact](https://www.nvidia.com/en-us/contact/)

Copyright © 2025, NVIDIA.