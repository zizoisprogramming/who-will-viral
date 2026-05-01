"""Tests for CLI and utils modules."""

from who_will_viral import utils
from who_will_viral.cli import main


def test_cli_main_function(mocker):
    """Test the main CLI function executes successfully."""
    # Mock the console and utils to avoid actual execution
    mock_console = mocker.patch('who_will_viral.cli.console')
    mock_utils = mocker.patch('who_will_viral.cli.utils.do_something_useful')

    main()

    # Verify console.print was called
    assert mock_console.print.call_count == 2
    # Verify utils function was called
    mock_utils.assert_called_once()


def test_cli_main_calls_utils_function(mocker):
    """Test that CLI main calls utils.do_something_useful."""
    mock_utils = mocker.patch('who_will_viral.cli.utils.do_something_useful')
    mocker.patch('who_will_viral.cli.console')

    main()
    mock_utils.assert_called_once()


def test_utils_do_something_useful(capsys):
    """Test that utils.do_something_useful prints to stdout."""
    utils.do_something_useful()
    captured = capsys.readouterr()
    assert 'Replace this with a utility function' in captured.out
