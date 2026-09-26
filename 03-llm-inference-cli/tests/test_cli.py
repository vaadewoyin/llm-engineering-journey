"""Tests for the LLM CLI using Typer's CliRunner."""
from typer.testing import CliRunner
from llm_cli.cli import app

runner = CliRunner()

def test_run_command():
    """Test the 'run' command of the CLI."""
    result = runner.invoke(app, ["run", "--model-id", "HuggingFaceTB/SmolLM2-135M-Instruct", 
                                 "--prompt", "What is the capital of France?", 
                                 "--no-streamer", 
                                 "--no-do-sample", 
                                 "--temperature", "0.7", 
                                 "--top-p", "1.0", 
                                 "--max-new-tokens", "10", 
                                 "--repetition-penalty", "1.0"])
    assert result.exit_code == 0
    assert "Outputs:" in result.output
    assert "Paris" in result.output

def test_run_bad_model_id():
    """Bad model ID should exit with non-zero and show error message."""
    result = runner.invoke(app, ["run", "--model-id", "not-a-real-model"])
    assert result.exit_code != 0

def test_cli_help():
    """Test that the CLI help command works."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "compare" in result.output