"""Test logging module."""

import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from bt.utils.logging import (
    JSONFormatter,
    TextFormatter,
    get_logger,
    setup_logging,
)


class TestJSONFormatter:
    """Test JSONFormatter class."""

    def test_basic_format(self) -> None:
        """Test basic JSON formatting."""
        formatter = JSONFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test"
        assert parsed["message"] == "Test message"
        assert "timestamp" in parsed

    def test_format_with_args(self) -> None:
        """Test JSON formatting with message args."""
        formatter = JSONFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Value: %s",
            args=("test_value",),
            exc_info=None,
        )

        result = formatter.format(record)
        parsed = json.loads(result)

        assert parsed["message"] == "Value: test_value"

    def test_format_with_exception(self) -> None:
        """Test JSON formatting with exception info."""
        formatter = JSONFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        result = formatter.format(record)
        parsed = json.loads(result)

        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]


class TestTextFormatter:
    """Test TextFormatter class."""

    def test_basic_format(self) -> None:
        """Test basic text formatting."""
        formatter = TextFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)

        assert "INFO" in result
        assert "test" in result
        assert "Test message" in result

    def test_format_with_colors(self) -> None:
        """Test that different levels have colors."""
        formatter = TextFormatter()

        levels = [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ]

        for level in levels:
            record = logging.LogRecord(
                name="test",
                level=level,
                pathname="test.py",
                lineno=1,
                msg="Test",
                args=(),
                exc_info=None,
            )

            result = formatter.format(record)
            # ANSI escape codes present
            assert "\033[" in result

    def test_format_with_exception(self) -> None:
        """Test text formatting with exception."""
        formatter = TextFormatter()

        try:
            raise RuntimeError("Test error")
        except RuntimeError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error",
            args=(),
            exc_info=exc_info,
        )

        result = formatter.format(record)

        assert "RuntimeError" in result
        assert "Test error" in result


class TestSetupLogging:
    """Test setup_logging function."""

    def test_setup_default(self) -> None:
        """Test default logging setup."""
        setup_logging()

        root = logging.getLogger()
        assert len(root.handlers) > 0

    def test_setup_json_format(self) -> None:
        """Test JSON format setup."""
        setup_logging(log_format="json")

        root = logging.getLogger()
        handler = root.handlers[0]

        assert isinstance(handler.formatter, JSONFormatter)

    def test_setup_text_format(self) -> None:
        """Test text format setup."""
        setup_logging(log_format="text")

        root = logging.getLogger()
        handler = root.handlers[0]

        assert isinstance(handler.formatter, TextFormatter)

    def test_setup_with_level(self) -> None:
        """Test logging level setup."""
        setup_logging(level="DEBUG")

        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_setup_with_file(self) -> None:
        """Test file handler setup."""
        with TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"

            setup_logging(log_file=log_file)

            logger = get_logger("test_file")
            logger.info("Test message")

            # File should exist and contain log
            assert log_file.exists()
            content = log_file.read_text()
            assert "Test message" in content

            # Clean up file handlers to avoid file lock
            root = logging.getLogger()
            for handler in root.handlers[:]:
                handler.close()
                root.removeHandler(handler)

    def test_setup_clears_existing_handlers(self) -> None:
        """Test that setup clears existing handlers."""
        root = logging.getLogger()

        # Add dummy handler
        dummy = logging.StreamHandler()
        root.addHandler(dummy)

        len(root.handlers)

        setup_logging()

        # Old handlers should be cleared
        assert dummy not in root.handlers


class TestGetLogger:
    """Test get_logger function."""

    def test_get_named_logger(self) -> None:
        """Test getting named logger."""
        logger = get_logger("test_module")

        assert logger.name == "test_module"

    def test_logger_has_correct_level(self) -> None:
        """Test logger respects root level."""
        setup_logging(level="DEBUG")

        logger = get_logger("test")

        # Logger should be able to log debug
        assert logger.isEnabledFor(logging.DEBUG)

    def test_multiple_calls_same_logger(self) -> None:
        """Test multiple calls return same logger."""
        logger1 = get_logger("same_name")
        logger2 = get_logger("same_name")

        assert logger1 is logger2
