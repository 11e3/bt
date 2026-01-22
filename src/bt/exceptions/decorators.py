"""Error handling decorators."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from bt.exceptions.handler import ErrorHandler

if TYPE_CHECKING:
    from collections.abc import Callable
    from logging import Logger

P = ParamSpec("P")
T = TypeVar("T")


def handle_errors(
    logger: Logger, reraise: bool = False
) -> Callable[[Callable[P, T]], Callable[P, T | None]]:
    """Decorator for automatic error handling on functions.

    Args:
        logger: Logger instance
        reraise: Whether to re-raise exceptions

    Returns:
        Decorated function
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T | None]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T | None:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                ErrorHandler.handle_error(e, logger, reraise, context={"function": func.__name__})
                if not reraise:
                    return None
                raise  # This line is unreachable but helps type checker

        return wrapper

    return decorator


def validate_parameters(
    **validators: Callable[[Any], None],
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for parameter validation.

    Args:
        **validators: Dict of parameter_name -> validation_function

    Returns:
        Decorated function
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Validate each parameter
            for param_name, validator in validators.items():
                if param_name in kwargs:
                    try:
                        validator(kwargs[param_name])
                    except Exception as e:
                        raise ErrorHandler.create_validation_error(
                            message=str(e), field=param_name, value=kwargs[param_name]
                        ) from e
            return func(*args, **kwargs)

        return wrapper

    return decorator
