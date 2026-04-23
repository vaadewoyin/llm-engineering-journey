"""Generation engine for LLM inference CLI."""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from llm_cli.config import GenConfig
from typing import Optional, Dict, Any


# Why: Custom exception for non-instruction-tuned models input, 
# to enforce instruction-tuned model requirement.
class ModelNotInstructionTunedError(Exception):
    """Error raised when a non-instruction-tuned model is used for generation."""
    pass


class GenerationEngine():
    """Generation engine that loads a model and tokenizer based on the provided GenConfig,
    and generates text based on a prompt."""
    def __init__(self, gen_config: GenConfig) -> None:
        """
        Initialises the generation engine

        Args: gen_config (dataclass): Configs for GenerationEngine

        Raises: ModelNotInstructionTunedError: if model_id is not an instruction-tuned model.
                OSError: if model_id is not correct.
        """

        self.config = gen_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == 'cuda' else torch.float32

        try:
            # Load tokenizer (may fail if wrong model ID)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)

            # Check that model is instruction-tuned
            if self.tokenizer.chat_template is None:
                raise ModelNotInstructionTunedError(
                    f"Model '{self.config.model_id}' is not instruction‑tuned. "
                    "Only instruction‑tuned models (with a chat template) are allowed.")

            # Load model 
            self.model = AutoModelForCausalLM.from_pretrained(self.config.model_id, 
                                                              device_map=self.device, dtype=self.dtype)

        except ModelNotInstructionTunedError:
            raise
    
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{self.config.model_id}'. Reason: {e}") from None
        
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
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            if self.config.streamer:
                streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            else:
                streamer = None

            # Why: time.perf_counter for timing the token generation process
            token_gen_start_time = time.perf_counter() 

            if self.config.do_sample is False:
                # Why: Greedy decoding if do_sample is False, otherwise use sampling-based decoding.
                with torch.no_grad():
                    outputs = self.model.generate(inputs["input_ids"], streamer=streamer, 
                                                  do_sample=False,
                                                  max_new_tokens=self.config.max_new_tokens, 
                                                  repetition_penalty=self.config.repetition_penalty,
                                                  attention_mask=inputs["attention_mask"]
                                                  )
            else:
                with torch.no_grad():
                    outputs = self.model.generate(inputs["input_ids"], streamer=streamer, 
                                                do_sample=self.config.do_sample,
                                                temperature=self.config.temperature, 
                                                top_p=self.config.top_p, 
                                                max_new_tokens=self.config.max_new_tokens, 
                                                repetition_penalty=self.config.repetition_penalty,
                                                attention_mask=inputs["attention_mask"]
                                                )
                
            token_gen_end_time = time.perf_counter()

            input_token_len = inputs["input_ids"].shape[1]
            outputs = outputs[0][input_token_len:]
            num_output_tokens = outputs.shape[-1]
            tokens_gen_time = token_gen_end_time - token_gen_start_time
            tokens_per_sec = num_output_tokens / tokens_gen_time 

            if streamer is None:
                print(f"\nOutputs: {self.tokenizer.decode(outputs, skip_special_tokens=True)}")

            return {
                "model":self.config.model_id, 
                "outputs":outputs, 
                "tokens_per_sec":tokens_per_sec, 
                "tokenizer": self.tokenizer,
                "token_gen_time_sec": tokens_gen_time
                }



# Check that generation engine works properly
if __name__ == "__main__":
    gen_engine = GenerationEngine(GenConfig())
    result = gen_engine.generate("Explain fine‑tuning in machine learning in one sentence.")