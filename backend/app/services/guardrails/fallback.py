from app.models.schemas import QueryResponse

CONTACT_LINE = "please contact the MSE program office at mse-applications@andrew.cmu.edu."


def off_limits_response(redirect_message: str) -> QueryResponse:
    return QueryResponse(answer=redirect_message, citations=[])


def low_confidence_response() -> QueryResponse:
    return QueryResponse(
        answer=(
            "I don't have reliable information to answer that from the knowledge base — "
            f"{CONTACT_LINE}"
        ),
        citations=[],
    )


def verification_failed_response() -> QueryResponse:
    return QueryResponse(
        answer=(
            "I found information that may be relevant, but couldn't verify the specific "
            f"figures well enough to share them confidently — {CONTACT_LINE}"
        ),
        citations=[],
    )
