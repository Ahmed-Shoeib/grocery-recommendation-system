"""Real-backend REST integration - the third `AdapterBundle` producer.

There is NO direct database access to the production backend. The
recommendation service obtains catalog, user, and engagement data through
the backend's HTTP API (Swagger: ``/swagger/index.html``). This package is
the ONLY place in the codebase that knows HTTP / the backend's JSON wire
shapes:

    backend REST API
        -> recommendation.data.backend.client   (HTTP, pagination, retries, TLS)
        -> recommendation.data.backend.dtos      (external response models)
        -> recommendation.data.backend.loader    (DTO -> Raw* / UserInteraction,
                                                   via ExternalIdentityResolver)
        -> recommendation.data.adapters.backend_factory.build_backend_api_adapters
        -> AdapterBundle   (identical interface to the synthetic / SQLite paths)
        -> EngagementProfile -> feature engineering -> Two-Tower -> ranker -> serving

Everything downstream of `build_backend_api_adapters` is unchanged and
never sees a slug, a GUID, an HTTP status code, or a backend field name -
the canonical schemas remain the stability boundary (docs/data-mapping.md
section 19).

Identity: the backend exposes products/categories by SLUG and users by
GUID, with no numeric ids. `recommendation.data.backend.identity
.ExternalIdentityResolver` maps each external key to a stable, persistent
internal `int` so the recommender core and the trained model artifacts
keep operating on the canonical integer-id contract they were built
against.
"""
