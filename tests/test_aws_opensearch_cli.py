import pytest
from pydantic import SecretStr

from vectordb_bench.backend.clients.aws_opensearch.cli import optional_secret_str


class TestOptionalSecretStr:
    """Test cases for the optional_secret_str helper function."""

    def test_none_input_returns_none(self):
        """Test that None input returns None."""
        result = optional_secret_str(None)
        assert result is None

    def test_string_input_returns_secret_str(self):
        """Test that string input returns SecretStr."""
        test_password = "my_secret_password"
        result = optional_secret_str(test_password)

        assert isinstance(result, SecretStr)
        assert result.get_secret_value() == test_password

    def test_empty_string_returns_secret_str(self):
        """Test that empty string returns SecretStr with empty value."""
        result = optional_secret_str("")

        assert isinstance(result, SecretStr)
        assert result.get_secret_value() == ""

    @pytest.mark.parametrize("test_input,expected_value", [
        ("password123", "password123"),
        ("", ""),
        ("special!@#$%^&*()chars", "special!@#$%^&*()chars"),
        ("   spaces   ", "   spaces   "),
        ("unicode_ñáéíóú", "unicode_ñáéíóú"),
    ])
    def test_various_string_inputs(self, test_input, expected_value):
        """Test various string inputs return SecretStr with correct values."""
        result = optional_secret_str(test_input)

        assert isinstance(result, SecretStr)
        assert result.get_secret_value() == expected_value

    def test_none_vs_empty_string_difference(self):
        """Test that None and empty string are handled differently."""
        none_result = optional_secret_str(None)
        empty_result = optional_secret_str("")

        assert none_result is None
        assert isinstance(empty_result, SecretStr)
        assert empty_result.get_secret_value() == ""

    def test_return_type_annotations(self):
        """Test that return types match the function signature."""
        # Test None case
        none_result = optional_secret_str(None)
        assert none_result is None

        # Test string case
        string_result = optional_secret_str("test")
        assert isinstance(string_result, SecretStr)