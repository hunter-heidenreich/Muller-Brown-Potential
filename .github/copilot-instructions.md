---
applyTo: "**"
---
# Project general coding standards

This repo is a modern, modular PyTorch implementation of the Müller-Brown potential energy surface with Langevin dynamics simulation.

This project uses `uv`. All commands should be run with it.

## General Principles
- Seek clarity, concision, and low cyclomatic complexity.
- If something can be simplified or streamlined, it should be. 
- Never sacrifice technical correctness. 
- Scientific correctness is paramount. Simulations must be reproducible and accurate.
- Comments should only be written where necessary. They should either clarify things that are non-obvious or provide broader needed context. They should be removed where they do not achieve these goals.
- When writing comments, documentation, or READMEs, aim for clarity and concision. Be humble. Never brag nor boast. Approachability and credibility are key. Language should be re-written where it violates these principles.

## Naming Conventions
- Use PascalCase for component names, interfaces, and type aliases
- Use camelCase for variables, functions, and methods
- Prefix private class members with underscore (_)
- Use ALL_CAPS for constants

## Type Annotations

- Avoid deprecated `typing` type annotations:
- `List` should never be used; use `list` instead.
- `Dict` should never be used; use `dict` instead.
- `Tuple` should never be used; use `tuple` instead.
- `Set` should never be used; use `set` instead.
- `Union` should never be used; use `|` instead.
- `Optional` should never be used; use `| None` instead.
