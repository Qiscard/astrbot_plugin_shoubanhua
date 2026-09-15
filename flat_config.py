"""Live flat view over the nested (object-grouped) plugin config.

The WebUI saves config values nested under object groups (e.g.
``network_settings.timeout``), while ~100 existing read sites across the
plugin still use flat top-level keys (``config.get("timeout")``). This view
resolves flat keys through a schema-derived mapping so existing code keeps
working unchanged; keys outside the schema (runtime-managed keys such as
``generic_active_source``) fall through to the top level.
"""

from typing import Any, Iterator, MutableMapping

from astrbot import logger


class FlatConfigView(MutableMapping):
    """A mutable mapping that presents the nested config as flat keys.

    Args:
        config: The real (nested) AstrBotConfig instance. Reads and writes are
            resolved live, so WebUI edits are always visible.
        schema: The plugin config schema. Object groups' items are remapped to
            flat keys; everything else stays top-level.
    """

    def __init__(self, config: MutableMapping, schema: Any = None):
        self._config = config
        self._flat_paths: dict = {}
        self._rebuild_paths(schema)

    def _rebuild_paths(self, schema: Any) -> None:
        """Collect item-key -> (group_key, item_key) mappings from object groups."""
        self._flat_paths = {}
        if not isinstance(schema, dict):
            return
        try:
            self._walk_schema(schema, prefix=())
        except Exception as e:
            logger.warning(f"FigurinePro: failed to build flat config paths, "
                           f"falling back to top-level reads only: {e}")
            self._flat_paths = {}

    def _walk_schema(self, schema: dict, prefix: tuple) -> None:
        for key, meta in schema.items():
            if not isinstance(meta, dict):
                continue
            if meta.get("type") == "object" and isinstance(meta.get("items"), dict):
                self._walk_schema(meta["items"], prefix + (key,))
            elif prefix:
                self._flat_paths[key] = prefix + (key,)

    def _resolve_container(self, key: str):
        """Return the dict that holds ``key``, or None when absent."""
        path = self._flat_paths.get(key)
        if path is None:
            return self._config
        node = self._config
        for part in path[:-1]:
            node = node.get(part) if isinstance(node, dict) else None
            if not isinstance(node, dict):
                return None
        return node

    def __getitem__(self, key: str) -> Any:
        container = self._resolve_container(key)
        if container is None:
            raise KeyError(key)
        return container[key]

    def __setitem__(self, key: str, value: Any) -> None:
        path = self._flat_paths.get(key)
        if path is None:
            self._config[key] = value
            return
        node = self._config
        for part in path[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[path[-1]] = value

    def __delitem__(self, key: str) -> None:
        container = self._resolve_container(key)
        if container is None or key not in container:
            raise KeyError(key)
        del container[key]

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        container = self._resolve_container(key)
        return container is not None and key in container

    def __iter__(self) -> Iterator[str]:
        yield from self._flat_paths
        for key in self._config:
            if key not in self._flat_paths:
                yield key

    def __len__(self) -> int:
        return len(list(iter(self)))
