"""ExternalIdentityResolver: deterministic, persistent, namespace-isolated
slug/GUID -> int resolution (docs/data-mapping.md section 19).
"""

import json

import pytest

from recommendation.data.backend.identity import ExternalIdentityResolver, IdentityRegistryError


def test_same_key_always_resolves_to_same_id_within_a_session(tmp_path):
    r = ExternalIdentityResolver(tmp_path / "reg.json")
    first = r.resolve_product("orange-juice")
    assert r.resolve_product("orange-juice") == first
    assert r.resolve_product("milk") != first


def test_namespaces_are_isolated(tmp_path):
    r = ExternalIdentityResolver(tmp_path / "reg.json")
    p = r.resolve_product("same-key")
    u = r.resolve_user("same-key")
    c = r.resolve_category("same-key")
    # Independent 1-based sequences: each namespace's first key is id 1,
    # and they are stored under separate maps (never merged/compared).
    assert p == u == c == 1
    assert r.counts() == {"product": 1, "category": 1, "user": 1}


def test_mapping_persists_and_is_identical_after_reload(tmp_path):
    path = tmp_path / "reg.json"
    r1 = ExternalIdentityResolver(path)
    ids = {slug: r1.resolve_product(slug) for slug in ("a", "b", "c")}
    users = {g: r1.resolve_user(g) for g in ("guid-x", "guid-y")}
    assert r1.save() is True

    # Simulate a process restart.
    r2 = ExternalIdentityResolver(path)
    for slug, want in ids.items():
        assert r2.resolve_product(slug) == want
    for g, want in users.items():
        assert r2.resolve_user(g) == want
    # No new assignments happened, so nothing to persist.
    assert r2.save() is False


def test_new_keys_after_reload_get_fresh_ids_never_reused(tmp_path):
    path = tmp_path / "reg.json"
    r1 = ExternalIdentityResolver(path)
    a = r1.resolve_product("a")
    b = r1.resolve_product("b")
    r1.save()

    r2 = ExternalIdentityResolver(path)
    c = r2.resolve_product("c")
    assert c not in (a, b)
    assert c == max(a, b) + 1


def test_removing_a_key_from_catalog_does_not_free_its_id(tmp_path):
    path = tmp_path / "reg.json"
    r1 = ExternalIdentityResolver(path)
    gone = r1.resolve_product("discontinued")
    kept = r1.resolve_product("kept")
    r1.save()

    # "discontinued" is never resolved again this run; a NEW product must
    # not reuse its id.
    r2 = ExternalIdentityResolver(path)
    fresh = r2.resolve_product("brand-new")
    assert fresh not in (gone, kept)
    # And if the old slug comes back, it keeps its original id.
    assert r2.resolve_product("discontinued") == gone


def test_peek_never_assigns(tmp_path):
    r = ExternalIdentityResolver(tmp_path / "reg.json")
    assert r.peek_product("unknown") is None
    assert r.peek_user("unknown") is None
    assert r.counts() == {"product": 0, "category": 0, "user": 0}
    r.resolve_product("known")
    assert r.peek_product("known") == 1


def test_empty_key_is_rejected(tmp_path):
    r = ExternalIdentityResolver(tmp_path / "reg.json")
    with pytest.raises(ValueError):
        r.resolve_product("")


def test_corrupt_registry_file_raises(tmp_path):
    path = tmp_path / "reg.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(IdentityRegistryError):
        ExternalIdentityResolver(path)


def test_unsupported_version_raises(tmp_path):
    path = tmp_path / "reg.json"
    path.write_text(json.dumps({"version": 999, "namespaces": {}}), encoding="utf-8")
    with pytest.raises(IdentityRegistryError):
        ExternalIdentityResolver(path)


def test_duplicate_id_in_stored_map_is_rejected(tmp_path):
    path = tmp_path / "reg.json"
    path.write_text(
        json.dumps({
            "version": 1,
            "namespaces": {"product": {"next_id": 3, "by_key": {"a": 1, "b": 1}}},
        }),
        encoding="utf-8",
    )
    with pytest.raises(IdentityRegistryError):
        ExternalIdentityResolver(path)


def test_counter_behind_keys_is_repaired_not_duplicated(tmp_path):
    path = tmp_path / "reg.json"
    path.write_text(
        json.dumps({
            "version": 1,
            "namespaces": {"product": {"next_id": 1, "by_key": {"a": 1, "b": 2, "c": 5}}},
        }),
        encoding="utf-8",
    )
    r = ExternalIdentityResolver(path)
    new_id = r.resolve_product("d")
    assert new_id == 6  # max(existing)=5 -> next is 6, never a collision


def test_namespace_offsets_make_ranges_disjoint(tmp_path):
    r = ExternalIdentityResolver(tmp_path / "reg.json", namespace_offsets={"user": 1_000_000})
    assert r.resolve_product("p") == 1
    assert r.resolve_user("u") == 1_000_001


def test_save_is_atomic_and_wellformed(tmp_path):
    path = tmp_path / "reg.json"
    r = ExternalIdentityResolver(path)
    r.resolve_product("p")
    r.resolve_user("u")
    r.save()
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["version"] == 1
    assert doc["namespaces"]["product"]["by_key"] == {"p": 1}
    assert doc["namespaces"]["user"]["by_key"] == {"u": 1}
