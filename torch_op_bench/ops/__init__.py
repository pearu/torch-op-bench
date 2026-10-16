"""In-package ops, by short name: ``name -> "module:OBJECT"``.

Modules are imported only when their op is resolved.
"""

REGISTRY: dict[str, str] = {}
