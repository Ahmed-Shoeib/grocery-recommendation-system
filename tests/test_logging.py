import logging

from recommendation.utils.logging import get_logger, setup_logging


def test_get_logger_returns_configured_logger():
    logger = get_logger("recommendation.test")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "recommendation.test"


def test_setup_logging_is_idempotent():
    setup_logging("DEBUG")
    handlers_after_first_call = len(logging.getLogger().handlers)
    setup_logging("DEBUG")
    handlers_after_second_call = len(logging.getLogger().handlers)
    assert handlers_after_first_call == handlers_after_second_call
