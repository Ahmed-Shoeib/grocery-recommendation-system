import numpy as np
import pytest

from recommendation.data.schemas.product import Product
from recommendation.embeddings.encoder import SentenceTransformerEncoder
from recommendation.embeddings.product_embeddings import (
    compute_product_embeddings,
    get_or_compute_product_embeddings,
    load_cache,
    save_cache,
)
from recommendation.embeddings.text_builder import build_product_text
from recommendation.utils.config import get_config

MODEL_NAME = get_config().embedding.sentence_transformer_model


@pytest.fixture(scope="module")
def encoder() -> SentenceTransformerEncoder:
    return SentenceTransformerEncoder(MODEL_NAME, device="cpu")


def _sample_products() -> list[Product]:
    return [
        Product(
            id=1, category_id=1, slug="greek-yogurt", name="Greek Yogurt Plain 500g", price=4.29,
            brand="GreenValley", category_name="Dairy & Eggs", description="High-protein low-sugar yogurt.",
            ingredients="Milk, live cultures.", tags=["healthy", "high-protein", "low-sugar"],
        ),
        Product(
            id=2, category_id=4, slug="potato-chips", name="Sea Salt Potato Chips 200g", price=2.79,
            brand="SnackWorks", category_name="Snacks", description="Crunchy kettle-cooked chips.",
            tags=["indulgent", "budget-friendly"],
        ),
    ]


# --- text builder -----------------------------------------------------------

def test_build_product_text_includes_all_specified_fields():
    product = Product(
        id=1, category_id=6, slug="coffee", name="Ground Coffee", price=7.99, brand="BrewHouse",
        category_name="Coffee & Tea", parent_category_name="Beverages", description="Smooth medium roast.",
        ingredients="100% Arabica.", tags=["premium", "on-the-go"],
    )
    text = build_product_text(product)
    for expected in ["Ground Coffee", "BrewHouse", "Coffee & Tea", "Beverages", "premium", "on-the-go",
                      "Smooth medium roast", "100% Arabica"]:
        assert expected in text


def test_build_product_text_handles_missing_optional_fields():
    product = Product(id=1, category_id=1, slug="x", name="Mystery Item", price=1.0)
    text = build_product_text(product)
    assert text == "Mystery Item"


# --- encoder ------------------------------------------------------------------

def test_encoder_produces_configured_dimension(encoder):
    assert encoder.embedding_dim == 384


def test_encoder_encode_shape_and_dtype(encoder):
    vectors = encoder.encode(["oat milk", "chicken breast"])
    assert vectors.shape == (2, 384)
    assert vectors.dtype == np.float32


def test_encoder_encode_empty_list_returns_empty_array(encoder):
    vectors = encoder.encode([])
    assert vectors.shape == (0, 384)


def test_encoder_similar_texts_are_closer_than_dissimilar_ones(encoder):
    a, b, c = encoder.encode(["Greek yogurt, high protein healthy breakfast", "Plain yogurt, protein-rich",
                               "Cola soda, sugary indulgent drink"])

    def cos(x, y):
        return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))

    assert cos(a, b) > cos(a, c)


# --- product embedding cache ---------------------------------------------------

def test_compute_product_embeddings_shape_matches_catalog(encoder):
    products = _sample_products()
    cache = compute_product_embeddings(products, encoder)
    assert cache.embeddings.shape == (2, 384)
    assert cache.product_ids == [1, 2]
    assert cache.model_name == MODEL_NAME
    assert set(cache.content_hashes.keys()) == {1, 2}


def test_cache_get_returns_correct_row(encoder):
    products = _sample_products()
    cache = compute_product_embeddings(products, encoder)
    vec = cache.get(1)
    assert vec is not None
    assert vec.shape == (384,)
    assert cache.get(999) is None


def test_save_and_load_cache_round_trips(tmp_path, encoder):
    products = _sample_products()
    cache = compute_product_embeddings(products, encoder)
    path = tmp_path / "embeddings.npz"
    save_cache(cache, path)

    loaded = load_cache(path)
    assert loaded is not None
    assert loaded.product_ids == cache.product_ids
    assert loaded.model_name == cache.model_name
    assert np.allclose(loaded.embeddings, cache.embeddings)
    assert loaded.content_hashes == cache.content_hashes


def test_load_cache_missing_file_returns_none(tmp_path):
    assert load_cache(tmp_path / "does_not_exist.npz") is None


def test_get_or_compute_recomputes_when_no_cache_exists(tmp_path, encoder):
    products = _sample_products()
    cache, recomputed = get_or_compute_product_embeddings(products, encoder, tmp_path / "cache.npz")
    assert recomputed is True
    assert cache.embeddings.shape == (2, 384)


def test_get_or_compute_reuses_cache_when_content_unchanged(tmp_path, encoder):
    products = _sample_products()
    path = tmp_path / "cache.npz"
    cache_a, recomputed_a = get_or_compute_product_embeddings(products, encoder, path)
    cache_b, recomputed_b = get_or_compute_product_embeddings(products, encoder, path)
    assert recomputed_a is True
    assert recomputed_b is False
    assert np.array_equal(cache_a.embeddings, cache_b.embeddings)


def test_get_or_compute_invalidates_cache_when_product_text_changes(tmp_path, encoder):
    path = tmp_path / "cache.npz"
    products = _sample_products()
    get_or_compute_product_embeddings(products, encoder, path)

    changed = _sample_products()
    changed[0] = changed[0].model_copy(update={"description": "Completely different description now."})
    _, recomputed = get_or_compute_product_embeddings(changed, encoder, path)
    assert recomputed is True


def test_get_or_compute_invalidates_cache_when_catalog_size_changes(tmp_path, encoder):
    path = tmp_path / "cache.npz"
    products = _sample_products()
    get_or_compute_product_embeddings(products, encoder, path)

    _, recomputed = get_or_compute_product_embeddings(products[:1], encoder, path)
    assert recomputed is True


def test_get_or_compute_force_recompute_ignores_valid_cache(tmp_path, encoder):
    path = tmp_path / "cache.npz"
    products = _sample_products()
    get_or_compute_product_embeddings(products, encoder, path)
    _, recomputed = get_or_compute_product_embeddings(products, encoder, path, force_recompute=True)
    assert recomputed is True
