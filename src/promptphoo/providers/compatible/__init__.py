"""OpenAI-compatible provider families (see D31).

`table.py` holds the data table for the families whose upstream factory only presets a base
URL and key env var on the OpenAI chat/completion/embedding providers. Families with their own
behaviour (cost tables, request rewriting, URL building) keep a small module each, named after
the upstream file. Every module exposes `create_provider(path, options)` for the registry.
"""
