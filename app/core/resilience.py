"""Shared retry policy for outbound calls.

Both the Cohere and Wikipedia clients retry transient failures with exponential
backoff. The retry mechanics are identical; only the predicate for "is this
error transient" differs per provider. This helper centralizes the mechanics so
each client supplies just its predicate.
"""

import logging
from collections.abc import Callable

from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)


def make_retrying(
    *,
    max_attempts: int,
    wait_min_s: float,
    wait_max_s: float,
    predicate: Callable[[BaseException], bool],
    logger: logging.Logger,
) -> AsyncRetrying:
    """Build an `AsyncRetrying` controller with exponential backoff."""
    return AsyncRetrying(
        retry=retry_if_exception(predicate),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=wait_min_s, max=wait_max_s),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
