# Imports
from llm_cli.config import GenConfig, load_gen_config
from llm_cli.generate import GenerationEngine
import typer
from rich.console import Console

app = typer.Typer()
console = Console()

# Cli command to generate text from a model
@app.command()
def generate(config_file: str = typer.Option(None, help= "config file to use"), 
             model_id: str = typer.Option(None, help="ID of the model to use"),
             prompt: str = typer.Option("What is the capital of France?", help = "prompt for text generation"),
             streamer: bool = typer.Option(True, help="Whether to use streaming output"),
             do_sample: bool = typer.Option(True, help="Whether to use sampling for generation"),
             temperature: float = typer.Option(0.7, help="Temperature for sampling"),
             top_p: float = typer.Option(0.8, help="Top-p value for sampling"),
             max_new_tokens: int = typer.Option(100, help="Maximum number of new tokens to generate"),
             repetition_penalty: float = typer.Option(1.25, help="Repetition penalty for generation")
             ):
    
    if config_file is not None:
        cfg = load_gen_config(config_file)
        console.print(f"\nUsing config from: {config_file}\n{cfg}")
        gen_engine = GenerationEngine(cfg)
    else:
        gen_cfg = GenConfig() # default config
        if model_id is not None or streamer is not True or do_sample is not True or temperature != 0.7 or top_p != 0.8 or max_new_tokens != 100 or repetition_penalty != 1.25:
            if model_id is None:
                model_id = gen_cfg.model_id
            cfg = GenConfig(model_id=model_id, streamer=streamer, do_sample=do_sample, temperature=temperature, 
                            top_p=top_p, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty)
            console.print(f"\nUsing config from command line arguments:\n{cfg}")
        else:
            cfg = gen_cfg
            console.print("\nUsing default configuration:")
            console.print()

        gen_engine = GenerationEngine(cfg) 

    gen_engine.generate(prompt)

    

