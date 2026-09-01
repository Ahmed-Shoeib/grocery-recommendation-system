"""Persistent external-identity resolution: slug / GUID -> stable internal int.

The backend addresses products and categories by SLUG and users by GUID,
and exposes no numeric ids. The recommender core, the canonical schemas,
and every trained model artifact operate on integer ids. This resolver is
the single boundary that bridges the two, and it is the ONLY component
that holds backend identifiers - nothing downstream of the adapter layer
ever sees a slug or a GUID.

Guarantees:

- **Deterministic & persistent.** The mapping is written to a JSON
  registry file (`config.paths.backend_identity_registry`, atomic
  write). The same external key always resolves to the same internal id,
  across process restarts and data refreshes.
- **Append-only.** A key is never re-numbered or removed. A product that
  disappears from the catalog keeps its id reserved, so if it comes back
  (or an activity still references it) the id is unchanged.
- **Namespace-isolated.** `product`, `category`, and `user` each have an
  independent 1-based id sequence and an independent key->id map. Internal
  `user_id` and `product_id` values are used in structurally separate
  lookups throughout this codebase (as they already are for the SQLite
  dataset, whose synthetic users 1..1000 and products 1..1200 overlap
  numerically without issue) and are never compared or merged, so
  independent sequences are safe. Set `namespace_offsets` to make the
  numeric ranges disjoint too if a future consumer needs that.
- **No `hash()`, no list position.** Ids come from a persisted
  monotonic counter per namespace, not from `hash(key)` (salted, unstable
  across interpreters) or enumeration order (unstable across catalog
  edits).
- **Collision-checked.** Loading a registry whose stored map assigns one
  id to two keys, or whose counter has fallen behind its keys, raises /
  self-repairs loudly rather than silently minting a duplicate id.

Slug mutability: if the backend changes a product's slug, this resolver
treats the new slug as a new product (new id); the old id is orphaned but
harmless. The robust backend contract is *immutable id + mutable slug*;
until that exists, this compromise is isolated here. See
docs/data-mapping.md section 19.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path

from recommendation.utils.logging import get_logger

logger = get_logger(__name__)

_REGISTRY_VERSION = 1
_NAMESPACES = ("product", "category", "user")


class IdentityRegistryError(RuntimeError):
    """The on-disk identity registry is structurally invalid (unparseable,
    wrong version, or an internally-inconsistent map) and cannot be used
    without risking wrong id assignment.
    """


class _Namespace:
    __slots__ = ("by_key", "next_id")

    def __init__(self, by_key: dict[str, int] | None = None, next_id: int = 1) -> None:
        self.by_key: dict[str, int] = dict(by_key or {})
        self.next_id = next_id

    def _validate_and_repair(self, name: str) -> None:
        ids = list(self.by_key.values())
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise IdentityRegistryError(
                f"identity registry namespace {name!r} maps multiple keys to id(s) {dupes} - refusing to load"
            )
        highest = max(ids, default=0)
        if self.next_id <= highest:
            logger.warning(
                "identity registry namespace %r counter (%d) is behind its keys (max id %d); advancing it",
                name, self.next_id, highest,
            )
            self.next_id = highest + 1

    def resolve(self, key: str) -> tuple[int, bool]:
        existing = self.by_key.get(key)
        if existing is not None:
            return existing, False
        assigned = self.next_id
        self.by_key[key] = assigned
        self.next_id += 1
        return assigned, True

    def peek(self, key: str) -> int | None:
        return self.by_key.get(key)


class ExternalIdentityResolver:
    """Load once, resolve many, persist on `save()`. Not tied to any HTTP
    client - `recommendation.data.backend.loader` drives it.
    """

    def __init__(self, registry_path: str | Path, *, namespace_offsets: dict[str, int] | None = None) -> None:
        self._path = Path(registry_path)
        self._lock = threading.Lock()
        self._dirty = False
        offsets = namespace_offsets or {}
        self._namespaces: dict[str, _Namespace] = {
            n: _Namespace(next_id=1 + offsets.get(n, 0)) for n in _NAMESPACES
        }
        self._offsets = {n: offsets.get(n, 0) for n in _NAMESPACES}
        self._load()

    # --- persistence -----------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            logger.info("identity registry %s does not exist yet; starting empty", self._path)
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IdentityRegistryError(f"cannot read identity registry {self._path}: {exc}") from exc

        if not isinstance(raw, dict) or raw.get("version") != _REGISTRY_VERSION:
            raise IdentityRegistryError(
                f"identity registry {self._path} has unsupported version {raw.get('version')!r} "
                f"(expected {_REGISTRY_VERSION})"
            )
        stored = raw.get("namespaces", {})
        for name in _NAMESPACES:
            entry = stored.get(name, {})
            by_key = {str(k): int(v) for k, v in entry.get("by_key", {}).items()}
            ns = _Namespace(by_key=by_key, next_id=int(entry.get("next_id", 1 + self._offsets[name])))
            ns._validate_and_repair(name)
            self._namespaces[name] = ns
        logger.info(
            "loaded identity registry %s (products=%d, categories=%d, users=%d)",
            self._path, *(len(self._namespaces[n].by_key) for n in _NAMESPACES),
        )

    def save(self) -> bool:
        """Atomically persist the current mapping if anything changed.
        Merges in any keys another process added since load (last-writer
        keeps its own assignments; a key present in both keeps the on-disk
        value and logs if they disagree). Returns True if a write happened.
        """
        with self._lock:
            if not self._dirty:
                return False
            if self._path.exists():
                try:
                    disk = json.loads(self._path.read_text(encoding="utf-8"))
                    self._merge_from_disk(disk.get("namespaces", {}))
                except (OSError, json.JSONDecodeError):
                    logger.warning("could not re-read identity registry %s before save; writing our view", self._path)

            payload = {
                "version": _REGISTRY_VERSION,
                "namespaces": {
                    name: {"next_id": ns.next_id, "by_key": ns.by_key}
                    for name, ns in self._namespaces.items()
                },
            }
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=self._path.name, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, sort_keys=True)
                os.replace(tmp, self._path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
            self._dirty = False
            logger.info("persisted identity registry %s", self._path)
            return True

    def _merge_from_disk(self, stored: dict) -> None:
        for name in _NAMESPACES:
            ns = self._namespaces[name]
            for key, value in stored.get(name, {}).get("by_key", {}).items():
                value = int(value)
                current = ns.by_key.get(key)
                if current is None:
                    ns.by_key[key] = value
                elif current != value:
                    logger.warning(
                        "identity registry conflict for %s %r: memory=%d disk=%d; keeping disk value",
                        name, key, current, value,
                    )
                    ns.by_key[key] = value
            ns.next_id = max(ns.next_id, max(ns.by_key.values(), default=0) + 1)

    # --- resolution ----------------------------------------------------

    def _resolve(self, namespace: str, key: str) -> int:
        if not key:
            raise ValueError(f"empty {namespace} identity key")
        with self._lock:
            value, created = self._namespaces[namespace].resolve(key)
            if created:
                self._dirty = True
        return value

    def resolve_product(self, slug: str) -> int:
        return self._resolve("product", slug)

    def resolve_category(self, slug: str) -> int:
        return self._resolve("category", slug)

    def resolve_user(self, guid: str) -> int:
        return self._resolve("user", guid)

    def peek_product(self, slug: str | None) -> int | None:
        return self._namespaces["product"].peek(slug) if slug else None

    def peek_user(self, guid: str | None) -> int | None:
        return self._namespaces["user"].peek(guid) if guid else None

    # --- introspection (diagnostics / tests) -------------------------

    def counts(self) -> dict[str, int]:
        return {n: len(self._namespaces[n].by_key) for n in _NAMESPACES}

    @property
    def registry_path(self) -> Path:
        return self._path
