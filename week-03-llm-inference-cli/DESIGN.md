# DESIGN.md — LLM Inference CLI

## 1. Problem
*[What am I solving and for whom? One or two sentences maximum.]*

I am building an LLM inference CLI for devs to quickly run and compare Hugging Face instruction-tuned models from the terminal without writing boilerplate loading code.

## 2. Components
*[List each component and its single responsibility.]*

- `cli.py` — creates a Typer based CLI app for running a single model and for comparing two models.
- `compare.py` — contains the logic for comparing two text generation models. It returns comparison data but does not render it.
- `config.py` — configuration used for both generation and streaming.
- `generate.py` — logic for generating text from models stays here.

tests: `test_cli.py / test_config.py` — for running unit tests.

## 3. Component Communication
*[How do the components talk to each other? What calls what? What data flows where?]*

`generate.py` imports `config.py`, uses the configs to generate text from model and returns generated text, and other parameters like tokens/sec, number of generated tokens, time taken for generation etc. These parameters are used for comparison. `compare.py` imports `generate.py` and uses it in generating text for both models and also returns performance metrics for both models for purpose of comparison. `cli.py` imports `config.py`, `generate.py`, and `compare.py` for creating the Typer based CLI app. There are two functions, one for generate and another for compare depending on the function called on the command line.

## 4. Failure Modes
*[The three most likely ways this breaks in real use. Not edge cases — the most probable failures.]*

1. The user inputs a non-existing model or wrong model ID.
2. OOM error if user inputs a large model.
3. Slow networks prevent Hugging Face model download.

## 5. Definition of Done
*[Specific and measurable. Not "it works." What exact behavior proves this is complete?]*

1. When I invoke `llm-cli run` with a model ID and prompt, the CLI renders the generated text, and other performance metrics like tokens/second using Rich.
2. When I invoke `llm-cli compare` with two model IDs and a prompt, the CLI renders the comparison table with response text and performance metrics for both models.
3. All necessary comparison metrics (tokens/sec, total tokens, generation time) are included in `compare.py` outputs.

## 5a. Tests I Will Write
*[How I will verify the Definition of Done programmatically.]*

- `test_config_save_and_load()`: Save a GenConfig to JSON, load it back, verify all values match.
- `test_config_defaults()`: Creating a GenConfig with no arguments uses the default values.
- `test_cli_help()`: Running `llm-cli --help` doesn't crash and shows commands to use.
- `test_cli_run_parses_args()`: To check the `run` command correctly parses arguments.

## 6. Production Boundaries

### What is deterministic
*[List every part that must never involve the LLM or randomness.]*

1. Argument parsing
2. CLI output structure
3. Error handling
4. Config loading, message building

### Human inspection point
*[Where can a human look to see exactly what the system did?]*
The CLI app outputs which include tokens/second printed after every generation, Rich comparison table showing responses side by side, other parameters like total time taken for generation etc.
Config JSON saved to disk to see used configuration.

### State representation
*[How does the system store and represent state? Must be simple and inspectable by a human opening a file.]*

The configs are stored in a JSON file that can be inspected. Also, model info and details are cached in a dictionary just for fast loading to avoid recurrent downloads of the same model.

### Serial vs parallel
*[Default is serial. If anything is parallel, justify it explicitly here.]*

Serial processing: model 1 runs completely before model 2 runs, then comparison is done. Running both in parallel will cause OOM error.

## 7. Pre-Build Questions
*[Answer every project-specific question here before writing code.]*

**Q:** [Why does config.py need to be a separate file? What breaks if GenerationConfig lives in cli.py?]

**A:** [config.py will be imported by generate.py and compare.py, both will also be imported by cli.py. But if GenerationConfig lives in cli.py, it means both generate.py and compare.py will import cli.py, which pulls Typer and all logic in cli just to call the config dataclass. Also this will lead to circular imports when cli in turn calls generate and compare.py.]

**Q:** [A user types a model ID that doesn't exist on HuggingFace. What happens right now? What should happen?]

**A:** [This results in OSError. This error should be caught so end user does not see the traceback, but rather sees a clean message: "Model 'xyz' not found on HuggingFace Hub. Check the model ID at huggingface.co/models."]

**Q:** [The compare command loads two models. On limited GPU memory, what is the risk and what is your mitigation?]

**A:** [The risk is getting an OOM error. This is prevented by loading one model at a time if we are comparing two models. Model 1 is loaded, text is generated, and all necessary details are stored. Then we clear memory: del model, torch.cuda.empty_cache(). Then model 2 is loaded, and memory is cleared too after saving its details. The two saved details are then used in comparison and in the display.]

**Q:** [Your generate.py caches models in a dict. If someone runs 10 different models in sequence, what happens to memory? Is this a problem for your use case?]

**A:** [If someone runs 10 different models without restarting the CLI session, the cache dictionary will hold all 10 models and cause OOM error. However, this is not a problem for this use case because the CLI is designed to run one model interactively or compare exactly two models at once. The cache helps speed up repeat prompts to the same model. This limitation will be documented in the README.]

**Q:** [Why does message order in the messages list matter? What breaks if system and user are swapped?]

**A:** [Message order matters because the system role tells the model how to behave in order to fulfill the user's request or instruction. So the system message must come before the user role. This order is maintained to avoid errors, and also to prevent the model from generating rubbish (which is worse than a crash). The end user has no business with message order since they are only entering a prompt via the CLI, the logic about message order is hidden from end user.]

## 8. Known Limitations
*[What does this system not handle? What would break it that you are aware of right now?]*

**What does this system not handle?**

The system was not designed to compare more than two models at a time.

**What would break it that you are aware of right now?**

1. OOM memory error if a very large model was used as input. However, the error will be caught and a clean message will be given instead of the traceback message.
2. OSError when trying to access a gated repo without proper authentication.