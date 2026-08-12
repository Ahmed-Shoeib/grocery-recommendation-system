"""ReviewAdapter backed by the ERD Review entity (auxiliary ranking signal)."""

from __future__ import annotations

from collections import defaultdict

from recommendation.data.adapters.base import ReviewAdapter
from recommendation.data.schemas.engagement import ReviewRecord
from recommendation.data.synthetic.raw_schemas import RawReview


class InMemoryReviewAdapter(ReviewAdapter):
    def __init__(self, reviews: list[RawReview]) -> None:
        self._reviews_by_user: dict[int, list[ReviewRecord]] = defaultdict(list)
        for review in reviews:
            self._reviews_by_user[review.user_id].append(
                ReviewRecord(
                    user_id=review.user_id,
                    product_id=review.product_id,
                    rating=review.rating,
                    comment=review.comment,
                    review_created_at=review.creation_date,
                )
            )

    def get_reviews(self, user_id: int) -> list[ReviewRecord]:
        return list(self._reviews_by_user.get(user_id, []))

    def list_all_reviews(self) -> list[ReviewRecord]:
        return [record for records in self._reviews_by_user.values() for record in records]
