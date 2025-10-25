# Repository Guidelines

## Coding Style & Naming Conventions

Write Python with 4-space indentation, type hints, and descriptive snake_case function names. Keep adapters encapsulated in classes or pure functions; prefer small, composable helpers over monolithic coroutines. Reuse the `CLIChatCompletionClient` pattern for new backends and keep constructor keyword names consistent (`make_argv`, `parse_response`, `extra_flags`). Include module-level docstrings and inline comments only where the flow is non-obvious.

## Commit & Pull Request Guidelines

commit after getting allowance from user.
