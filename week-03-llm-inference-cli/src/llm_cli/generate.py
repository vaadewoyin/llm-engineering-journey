# Imports
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from llm_cli.config import GenConfig
from typing import Optional, Dict, Any


# Custom error
class ModelNotInstructionTunedError(Exception):
    """Error raised when the model is not instruction-tuned."""
    pass


# Generation Engine
class GenerationEngine():
    """ This class generates texts and token/secs using hf model"""
    def __init__(self, gen_config: GenConfig) -> None:
        """
        Initialises the class
        Args:
            gen_config (dataclass): Configs for GenerationEngine
        Raises:
            ModelNotInstructionTunedError: if model_id is not an instruction-tuned model.
            OSError: if model_id is not correct.
        """
        self.config = gen_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == 'cuda' else torch.float32
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
            if self.tokenizer.chat_template is not None:
               self.model = AutoModelForCausalLM.from_pretrained(self.config.model_id, device_map = "auto",
                                                                 dtype = self.dtype)
            else:
              self.model = None
              raise ModelNotInstructionTunedError('Load a huggingface instruction-tuned model')

        except ModelNotInstructionTunedError as e:
            print(e)
            raise

        except OSError:
            print(f'Error: Model {self.config.model_id} not found. Check the model ID at huggingface.co/models.')
            raise
        
    def generate(self, prompt) -> Optional[Dict[str, Any]]:
        """Generates text based on the prompt and system message in the config using 
        the model specified in the config.

        Returns: A dictionary containing the model name, generated outputs, and 
        tokens per second if generation is successful.
        """
        messages = [
          {"role": "system", "content": self.config.system_prompt},
          {"role": "user", "content": prompt}
          ]
        
        if self.model is not None:
            inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, 
                                              return_tensors="pt", add_generation_prompt=True)
            input_ids=inputs.to(self.device)

            if self.config.streamer:
                streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            else:
                streamer = None

            token_gen_start_time = time.perf_counter() #why: time.perf_counter is better for timing performance
            with torch.no_grad():
                outputs = self.model.generate(input_ids, streamer=streamer, 
                                              do_sample=self.config.do_sample,
                                              temperature=self.config.temperature, 
                                              top_p=self.config.top_p, 
                                              max_new_tokens=self.config.max_new_tokens, 
                                              repetition_penalty=self.config.repetition_penalty
                                              )
              
            token_gen_end_time = time.perf_counter()
            input_token_len = inputs["input_ids"].shape[1]
            outputs = outputs[0][input_token_len:]
            num_output_tokens = outputs.shape[-1]
            tokens_gen_time = token_gen_end_time - token_gen_start_time
            tokens_per_sec = num_output_tokens / tokens_gen_time 
            return {"model":self.config.model_id, "outputs":outputs, "tokens_per_sec":tokens_per_sec}

# Check that config was properly imported
if __name__ == "__main__":
    config = GenConfig()
    print(type(config))