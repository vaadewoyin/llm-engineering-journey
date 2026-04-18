# Imports 
import json
from dataclasses import dataclass,  asdict
from pathlib import Path

# Config dataclass
@dataclass
class GenConfig:
    model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct" #why: SmolLM2 for quick experimentation
    system_prompt: str =  "You are a helpful assistant."
    streamer: bool = True
    do_sample: bool = True
    temperature: float= 0.7
    top_p: float= 0.8
    max_new_tokens: int = 256 
    repetition_penalty: float = 1.25

class SaveAndLoadConfig():
    """A class to save and load GenConfig objects to and from disk."""
    def __init__(self, config: GenConfig):
        """ Initializes the SaveAndLoadConfig instance."""
        self.config = config
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        self.CONFIG_DIR = PROJECT_ROOT / "configs"
        self.CONFIG_DIR.mkdir(exist_ok=True)

    def save_json(self, file_name: str) -> None:
        with open(f"{self.CONFIG_DIR}/{file_name}.json", "w") as  f:
                json.dump(asdict(self.config), f, indent =2)
    
    def load_json(self, file_name: str) -> GenConfig:
        with open(f"{self.CONFIG_DIR}/{file_name}.json", "r") as  f:
            config = json.load(f)
        return GenConfig(**config) #why: unpacking the json to GenConfig fields for consistency
         
# Check that config can be saved and loaded properly
if __name__ == "__main__":
    save_and_load = SaveAndLoadConfig(GenConfig())
    save_and_load.save_json("default_config")
    config = save_and_load.load_json("default_config")
    print(config)