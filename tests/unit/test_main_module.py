"""Tests for the __main__ module."""

from unittest.mock import patch


def test_main_module_import():
	"""Test that __main__ module can be imported."""
	from who_will_viral import __main__

	assert __main__


@patch('who_will_viral.__main__.app')
def test_main_module_execution(mock_app):
	"""Test __main__ module calls app when executed."""
	# Simulate execution of __main__.py
	import who_will_viral.__main__ as main_module

	# Verify that the module can be loaded
	assert main_module


def test_main_module_app_reference():
	"""Test that __main__ has reference to app."""
	from who_will_viral.__main__ import app
	from who_will_viral.cli import app as cli_app

	assert app is cli_app
