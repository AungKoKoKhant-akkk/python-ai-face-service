"""Compatibility shim so the package path `app.main:app` works.

This module imports the FastAPI app defined in the project's top-level
`main.py` and re-exports it as `app`. That lets users run

    uvicorn app.main:app --reload --port 8000

without moving or duplicating the main application code.
"""
from importlib import import_module

# Import the top-level main module and re-export its `app` object
_main = import_module("main")
app = getattr(_main, "app")
