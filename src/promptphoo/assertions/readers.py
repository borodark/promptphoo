"""Re-export so config loading can import the reader without the whole dispatcher."""

from promptphoo.assertions.index import read_assertions

__all__ = ["read_assertions"]
