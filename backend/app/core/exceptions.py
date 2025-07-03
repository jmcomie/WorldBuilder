"""Custom exceptions for the Worldbuilder application."""
from fastapi import HTTPException
from typing import Optional


class WorldbuilderException(Exception):
    """Base exception for Worldbuilder application."""
    pass


class RateLimitException(HTTPException):
    """Rate limit exceeded exception."""
    def __init__(self, detail: str = "Rate limit exceeded. Please try again later."):
        super().__init__(status_code=429, detail=detail)


class AuthenticationException(HTTPException):
    """Authentication failed exception."""
    def __init__(self, detail: str = "Authentication failed. Please check your API key."):
        super().__init__(status_code=401, detail=detail)


class GraphitiException(WorldbuilderException):
    """Graphiti-specific exception."""
    pass


def handle_graphiti_error(error: Exception) -> HTTPException:
    """Convert Graphiti errors to appropriate HTTP exceptions."""
    error_message = str(error)
    
    if "rate_limit_exceeded" in error_message.lower() or "rate limit" in error_message.lower():
        return RateLimitException()
    elif "api_key" in error_message.lower() or "authentication" in error_message.lower():
        return AuthenticationException("OpenAI API authentication failed. Please check your API key.")
    else:
        return HTTPException(status_code=500, detail=error_message)