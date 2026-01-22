# 12. Release Notes — Tile IR

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
-   [3. Syntax](syntax.html)
-   [4. Binary Format](bytecode.html)
-   [5. Type System](types.html)
-   [6. Semantics](semantics.html)
-   [7. Memory Model](memory_model.html)
-   [8. Operations](operations.html)
-   [9. Debug Info](debug_info.html)
-   [10. Stability](stability.html)
-   [11. Appendix](appendix.html)
-   [12. Release Notes](#)

-   [](../index.html)
-   12. Release Notes

# 12. Release Notes[#](#release-notes "Link to this heading")

## 12.1. Known Issues[#](#known-issues "Link to this heading")

-   The programming model is missing a section on a cross-tile block kernel such as split-k.
    
-   The bytecode section does not provide exact encoding of each operation, expect this to be introduced in a future release.
    
-   The semi-formal memory model section is written but does not provide detailed examples of how to to utilize it.
    
-   Atomics are currently limited in **Tile IR** and will be expanded in a future release.
    

## 12.2. Changelog[#](#changelog "Link to this heading")

### 12.2.1. Spec 13.1 (2025-12-08)[#](#spec-13-1-2025-12-08 "Link to this heading")

#### Bugfixes[#](#bugfixes "Link to this heading")

-   Fixed typo in the type section, where the tile size was incorrectly specified as 128x64 instead of 128x4.
    

#### Improved Documentation[#](#improved-documentation "Link to this heading")

-   Fixed numerous small typos
    
-   Strengthen the wording around the stability of the textual format of **Tile IR**.
    

[

previous

11. Appendix



](appendix.html "previous page")

On this page

-   [12.1. Known Issues](#known-issues)
-   [12.2. Changelog](#changelog)
    -   [12.2.1. Spec 13.1 (2025-12-08)](#spec-13-1-2025-12-08)
        -   [Bugfixes](#bugfixes)
        -   [Improved Documentation](#improved-documentation)

[![NVIDIA](../_static/nvidia-logo-horiz-rgb-1c-blk-for-screen.svg) ![NVIDIA](../_static/nvidia-logo-horiz-rgb-1c-wht-for-screen.svg)](https://www.nvidia.com)

[Privacy Policy](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/) | [Your Privacy Choices](https://www.nvidia.com/en-us/about-nvidia/privacy-center/) | [Terms of Service](https://www.nvidia.com/en-us/about-nvidia/terms-of-service/) | [Accessibility](https://www.nvidia.com/en-us/about-nvidia/accessibility/) | [Corporate Policies](https://www.nvidia.com/en-us/about-nvidia/company-policies/) | [Product Security](https://www.nvidia.com/en-us/about-nvidia/product-security/) | [Contact](https://www.nvidia.com/en-us/contact/)

Copyright © 2025, NVIDIA.