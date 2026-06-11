"""Domain exceptions and their HTTP mapping.

These insulate the rest of the app from provider-specific SDK errors. The
router and the wrapper raise these, and a single exception handler (registered
in `app.main`) renders them as a consistent error envelope. Each exception
carries the HTTP status and a stable machine-readable `error_code` so clients
can branch on failures without parsing prose.
"""


class AppError(Exception):
    """Base class for errors that map to a structured HTTP response."""

    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str) -> None:
        """Initialize the error."""
        super().__init__(message)
        self.message = message


class CohereError(AppError):
    """A call to the Cohere API failed in a way we surface as a bad gateway."""

    status_code = 502
    error_code = "cohere_error"


class CohereTimeoutError(CohereError):
    """The Cohere API did not respond within the configured timeout."""

    status_code = 504
    error_code = "cohere_timeout"


class CohereUnavailableError(CohereError):
    """The Cohere API was unavailable after exhausting retries."""

    status_code = 502
    error_code = "cohere_unavailable"


class CohereAuthError(CohereError):
    """Cohere rejected our credentials.

    This is a server-side configuration problem, so we report a 500 and do not
    leak provider detail to the caller.
    """

    status_code = 500
    error_code = "cohere_auth_error"


class WikipediaError(AppError):
    """A Wikipedia search failed.

    The chat orchestrator catches this and lets the model continue without the
    tool result, so it rarely reaches the HTTP layer. The mapping exists for
    the case where a future caller surfaces it directly.
    """

    status_code = 502
    error_code = "wikipedia_unavailable"


class ConversationNotFoundError(AppError):
    """A request referenced a conversation id that does not exist.

    Also raised when a conversation exists but belongs to another owner, so the
    response does not reveal whether the id exists.
    """

    status_code = 404
    error_code = "conversation_not_found"


class AuthenticationError(AppError):
    """A request was missing or presented an invalid API key."""

    status_code = 401
    error_code = "unauthorized"


class RateLimitExceededError(AppError):
    """A client exceeded its allowed request rate."""

    status_code = 429
    error_code = "rate_limited"
