"""Tests for the web_fetch plain-text wrapper.

The wrapper converts WebFetchResult (structured JSON) to plain text with preserved
newlines, making spilled content pageable via read_tool_result's line-based pagination.
BinaryContent (images, PDFs, audio) passes through unchanged.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic_ai.common_tools.web_fetch import BinaryContent

from agent.tools import web_fetch, _underlying_web_fetch


@pytest.fixture
def mock_ctx():
    """Minimal mock RunContext for tool calls."""
    ctx = MagicMock()
    ctx.deps = MagicMock()
    return ctx


def _mock_result(url="https://example.com", title="", content="x"):
    """Build a WebFetchResult-shaped dict."""
    return {"url": url, "title": title, "content": content}


async def _call_with_mock_result(mock_ctx, mock_result, url="https://example.com"):
    """Call web_fetch with a mocked underlying tool return."""
    with patch.object(_underlying_web_fetch, "function", new_callable=AsyncMock) as mock_fn:
        mock_fn.return_value = mock_result
        result = await web_fetch(mock_ctx, url)
    return result, mock_fn


class TestWebFetchWrapper:
    """Tests for the web_fetch plain-text wrapper."""

    async def test_text_result_format(self, mock_ctx):
        """Wrapper returns plain text with header, preserving content newlines."""
        content = "First paragraph.\n\nSecond paragraph."
        result, _ = await _call_with_mock_result(
            mock_ctx,
            _mock_result("https://example.com/page", "Example Page", content),
        )

        # Build expected output piece by piece
        expected = "URL: https://example.com/page\n"  # URL line
        expected += "Title: Example Page\n"  # Title line (present when non-empty)
        expected += "---\n"  # Separator
        expected += content  # Content with preserved newlines
        assert result == expected

    @pytest.mark.parametrize("title,expected_output", [
        ("", "URL: https://example.com\n---\nx"),  # No title line when empty
        ("My Page Title", "URL: https://example.com\nTitle: My Page Title\n---\nx"),  # Title line when present
    ])
    async def test_title_inclusion(self, mock_ctx, title, expected_output):
        """Title line included only when title is non-empty."""
        result, _ = await _call_with_mock_result(mock_ctx, _mock_result(title=title))
        assert result == expected_output

    async def test_line_count_matches_content(self, mock_ctx):
        """Output line count reflects actual content lines for pagination."""
        content_lines = [f"Line {i}" for i in range(10)]
        content = "\n".join(content_lines)
        result, _ = await _call_with_mock_result(mock_ctx, _mock_result(title="Test", content=content))

        # Header: URL, Title, ---, then 10 content lines = 13 total
        expected = "URL: https://example.com\nTitle: Test\n---\n" + content
        assert result == expected
        assert len(result.split("\n")) == 13  # Verify line count assumption

    async def test_passes_url_to_underlying_tool(self, mock_ctx):
        """URL is passed through to the underlying web_fetch_tool."""
        _, mock_fn = await _call_with_mock_result(mock_ctx, _mock_result("https://test.com"), "https://test.com")

        mock_fn.assert_called_once_with("https://test.com")

    async def test_binary_content_passes_through_unchanged(self, mock_ctx):
        """BinaryContent (images, PDFs, audio) passes through unchanged."""
        mock_binary = BinaryContent(data=b"fake image data", media_type="image/png")
        result, _ = await _call_with_mock_result(mock_ctx, mock_binary, "https://example.com/image.png")

        # Identity check — same object returned unchanged
        assert result is mock_binary
