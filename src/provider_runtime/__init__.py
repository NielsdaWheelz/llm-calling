"""Public provider runtime API.

INTERIM STATE during the v2 hard cutover: the old facade re-exported the dead
planner/catalog/codec surface and can no longer import. The v2 facade (with
its final ``__all__``, <= 40 names) is rewritten in the runtime/facade stage.
Until then, import from the submodules directly: ``provider_runtime.types``,
``provider_runtime.errors``, ``provider_runtime.registry``,
``provider_runtime.engines``.
"""
