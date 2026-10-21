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


class TestWebFetchWrapper:
    """Tests for the web_fetch plain-text wrapper."""

    async def test_returns_plain_text_with_header(self, mock_ctx):
        """Wrapper returns plain text with URL/title header, not structured JSON."""
        mock_result = {
            "url": "https://example.com/page",
            "title": "Example Page",
            "content": "Line 1\nLine 2\nLine 3",
        }

        with patch.object(_underlying_web_fetch, "function", new_callable=AsyncMock) as mock_fn:
            mock_fn.return_value = mock_result
            result = await web_fetch(mock_ctx, "https://example.com/page")

        # Should be plain text, not JSON
        assert not result.startswith("{")
        assert "URL: https://example.com/page" in result
        assert "Title: Example Page" in result
        assert "---" in result

    async def test_preserves_content_newlines(self, mock_ctx):
        """Content newlines are preserved as actual newlines, not escaped."""
        content_with_newlines = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        mock_result = {
            "url": "https://example.com",
            "title": "",
            "content": content_with_newlines,
        }

        with patch.object(_underlying_web_fetch, "function", new_callable=AsyncMock) as mock_fn:
            mock_fn.return_value = mock_result
            result = await web_fetch(mock_ctx, "https://example.com")

        # Content should appear with real newlines
        assert "First paragraph.\n\nSecond paragraph." in result
        # Should NOT have escaped newlines like JSON would
        assert "\\n" not in result

    async def test_omits_empty_title(self, mock_ctx):
        """Empty title is not included in header."""
        mock_result = {
            "url": "https://example.com",
            "title": "",
            "content": "Some content",
        }

        with patch.object(_underlying_web_fetch, "function", new_callable=AsyncMock) as mock_fn:
            mock_fn.return_value = mock_result
            result = await web_fetch(mock_ctx, "https://example.com")

        assert "Title:" not in result
        assert "URL: https://example.com" in result

    async def test_includes_nonempty_title(self, mock_ctx):
        """Non-empty title is included in header."""
        mock_result = {
            "url": "https://example.com",
            "title": "My Page Title",
            "content": "Some content",
        }

        with patch.object(_underlying_web_fetch, "function", new_callable=AsyncMock) as mock_fn:
            mock_fn.return_value = mock_result
            result = await web_fetch(mock_ctx, "https://example.com")

        assert "Title: My Page Title" in result

    async def test_line_count_matches_content(self, mock_ctx):
        """Output line count reflects actual content lines for pagination."""
        # 10 lines of content
        content_lines = [f"Line {i}" for i in range(10)]
        content = "\n".join(content_lines)

        mock_result = {
            "url": "https://example.com",
            "title": "Test",
            "content": content,
        }

        with patch.object(_underlying_web_fetch, "function", new_callable=AsyncMock) as mock_fn:
            mock_fn.return_value = mock_result
            result = await web_fetch(mock_ctx, "https://example.com")

        result_lines = result.split("\n")
        # Header: URL, Title, ---, then 10 content lines = 13 total
        assert len(result_lines) == 13
        # Content lines should be present
        assert "Line 0" in result
        assert "Line 9" in result

    async def test_passes_url_to_underlying_tool(self, mock_ctx):
        """URL is passed through to the underlying web_fetch_tool."""
        mock_result = {"url": "https://test.com", "title": "", "content": "x"}

        with patch.object(_underlying_web_fetch, "function", new_callable=AsyncMock) as mock_fn:
            mock_fn.return_value = mock_result
            await web_fetch(mock_ctx, "https://test.com")

        # Underlying tool only takes url, not ctx
        mock_fn.assert_called_once_with("https://test.com")

    async def test_binary_content_passes_through_unchanged(self, mock_ctx):
        """BinaryContent (images, PDFs, audio) passes through unchanged."""
        mock_binary = BinaryContent(data=b"fake image data", media_type="image/png")

        with patch.object(_underlying_web_fetch, "function", new_callable=AsyncMock) as mock_fn:
            mock_fn.return_value = mock_binary
            result = await web_fetch(mock_ctx, "https://example.com/image.png")

        # Should return the exact same BinaryContent object
        assert result is mock_binary
        assert isinstance(result, BinaryContent)
        assert result.data == b"fake image data"
        assert result.media_type == "image/png"
