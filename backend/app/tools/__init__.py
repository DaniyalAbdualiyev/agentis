"""
Import all tools so their registry.register() side-effects run at startup.
"""
from app.tools import web_search, code_execution  # noqa: F401
from app.tools.registry import registry  # noqa: F401
