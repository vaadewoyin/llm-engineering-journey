"""CLI for LLM inference and comparison."""

import typer
from rich.console import Console
from rich.table import Table
from llm_cli.config import GenConfig, load_gen_config
from llm_cli.generate import GenerationEngine, ModelNotInstructionTunedError
from llm_cli.compare import compare_models

app = typer.Typer()
console = Console()

# CLI command to generate text from a model
@app.command()
def generate(config_file: str = typer.Option(None, help= "config file to use"), 
             model_id: str = typer.Option(None, help="ID of the model to use"),
             prompt: str = typer.Option("Explain fine‑tuning in machine learning in one sentence.", help = "prompt for text generation"),
             streamer: bool = typer.Option(True, help="Whether to use streaming output"),
             do_sample: bool = typer.Option(True, help="Whether to use sampling for generation"),
             temperature: float = typer.Option(0.7, help="Temperature for sampling"),
             top_p: float = typer.Option(0.8, help="Top-p value for sampling"),
             max_new_tokens: int = typer.Option(100, help="Maximum number of new tokens to generate"),
             repetition_penalty: float = typer.Option(1.25, help="Repetition penalty for generation")
             ):
    
    # Why: If config_file is provided, load config from file and ignore other CLI args (with a warning).
    # If config file is not provided, use CLI args if provided, otherwise use defaults. 
    if config_file is not None:
        try:
            cfg = load_gen_config(config_file)
            console.print(f"\nUsing config from: {config_file}\n{cfg}")
        except FileNotFoundError as e:
            console.print(f"\n[red]Error:[/red] {e}") 
            raise typer.Exit(code=1)

        #gen_engine = GenerationEngine(cfg)

    else:
        defaults = GenConfig()  
        
        # Why: If any CLI arg is provided, override the corresponding field in the config.
        if (model_id is not None or
            streamer != defaults.streamer or
            do_sample != defaults.do_sample or
            temperature != defaults.temperature or
            top_p != defaults.top_p or
            max_new_tokens != defaults.max_new_tokens or
            repetition_penalty != defaults.repetition_penalty): 

            if model_id is None:
                model_id = defaults.model_id

            cfg = GenConfig(model_id=model_id, streamer=streamer, do_sample=do_sample, temperature=temperature, 
                            top_p=top_p, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty)
            console.print(f"\nUsing config from command line arguments:\n{cfg}")

        else:
            cfg = defaults
            console.print("\nUsing default configuration:")
            console.print()
    try:
        gen_engine = GenerationEngine(cfg)
        result = gen_engine.generate(prompt)
    except (RuntimeError, ModelNotInstructionTunedError) as e:
        console.print(f"\n[red]Error:[/red] {e}") 
        raise typer.Exit(code=1)

    # Generate text based on the prompt and configuration
    #result = gen_engine.generate(prompt)

    if streamer is False:
        console.print(f"\nOutputs: {result['tokenizer'].decode(result['outputs'], skip_special_tokens=True)}")


    
# CLI command to compare two models
@app.command()
def compare(
    model_id1: str = typer.Option(..., help="ID of the first model to compare"), 
    model_id2: str = typer.Option(..., help="ID of the second model to compare"), 
    prompt: str = typer.Option("Explain fine‑tuning in machine learning in one sentence.", help="Prompt for generation")
    ):
    try:
        comparison_result = compare_models(model_id1, model_id2, prompt)
        console.print(f"\nComparison of models '{model_id1}' and '{model_id2}' for prompt: '{prompt}'\n")
    except (RuntimeError, ModelNotInstructionTunedError) as e:
        console.print(f"\n[red]Error:[/red] {e}")
        raise typer.Exit(code=1)
   
    # Get outputs and performance metrics for both models
    model_id1_output = comparison_result["model_id1_output"]
    model_id2_output = comparison_result["model_id2_output"]
    model_id1_decoded_output = model_id1_output["tokenizer"].decode(model_id1_output["outputs"], skip_special_tokens=True)
    model_id2_decoded_output = model_id2_output["tokenizer"].decode(model_id2_output["outputs"], skip_special_tokens=True)
    model_id1_tokens_per_sec = model_id1_output["tokens_per_sec"]
    model_id2_tokens_per_sec = model_id2_output["tokens_per_sec"]
    model_id1_generation_time = model_id1_output["token_gen_time_sec"]
    model_id2_generation_time = model_id2_output["token_gen_time_sec"]
    model_id1_token_count = model_id1_output["outputs"].shape[-1]
    model_id2_token_count = model_id2_output["outputs"].shape[-1]

    # Print model decoded outputs
    console.print(f"[bold]Model[/bold] '[bold]{model_id1}[/bold]' \n[bold]output[/bold]: {model_id1_decoded_output}")
    console.print(f"\n[bold]Model[/bold] '[bold]{model_id2}[/bold]' \n[bold]output[/bold]: {model_id2_decoded_output}")

    model_results = [
        {"Model": model_id1, "Tokens Generated": model_id1_token_count, "Generation Time (s)": model_id1_generation_time, 
         "Tokens/sec": model_id1_tokens_per_sec},
        {"Model": model_id2, "Tokens Generated": model_id2_token_count, "Generation Time (s)": model_id2_generation_time, 
         "Tokens/sec": model_id2_tokens_per_sec}
    ]
    
    console.print('\n')

    # Rich table for comparison
    table = Table(title="Model Comparison")
    table.add_column("Model", justify="left", style="cyan", no_wrap=True)
    table.add_column("Tokens Generated", justify="right", style="magenta")
    table.add_column("Generation Time (s)", justify="right", style="green")
    table.add_column("Tokens/sec", justify="right", style="yellow")

    for result in model_results:    
        table.add_row(result["Model"], str(result["Tokens Generated"]), f"{result['Generation Time (s)']:.2f}", 
                      f"{result['Tokens/sec']:.2f}")
    
    # Display table with rich
    console.print(table)