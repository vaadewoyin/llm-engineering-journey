# DESIGN.md — ARXIV-QA-DATASET

## 1. Problem
To create a high-quality Q&A dataset from Arxiv ML paper abstract for fine-tuning language model to answer technical question


## 2. Components
*[List each component and its single responsibility.]*

- `ArXiv API Scraper` — Fetches paper metadata (title, abstract, ID, category)
- `Cleaning Pipeline` — Deduplication, language check, length filter (min 100 tokens, max 2048)
- `Q&A Generator` — Uses local LLM to generates Q&A pairs from abstracts in ChatML format
- `Quality filter` — Validates ChatML format, checks question/answer token lengths (min 20/25), and verifies groundedness via keyword overlap (min 75%).
- `Logger` — uses CometML for logging at each step, Every Q&A Generator LLM call is traced in Opik with latency, and quality


## 3. Component Communication
*[How do the components talk to each other? What calls what? What data flows where?]*

Scraper saves raw jsonl. Cleaner reads raw jsonl, outputs cleaned jsonl. Generator reads cleaned jsonl, outputs pairs jsonl. Filter reads pairs jsonl, outputs final dataset jsonl. All stage counts logged to Comet ML.

## 4. Failure Modes
*[The three most likely ways this breaks in real use. Not edge cases — the most probable failures.]*

1. ArXiv API fails during scrapping process (mitigated by saving progress after each request and resuming).
2. Local LLM generates low-quality or ungrounded pairs (mitigated by the quality filter and overlap check respectively).
3. Quality filter rejects too many pairs, resulting in insufficient data (mitigated by manual inspection of rejected samples and threshold adjustment).



## 5. Definition of Done
*[Specific and measurable. Not "it works." What exact behavior proves this is complete?]*

1. A dataset of Q&A pairs in ChatML format, pushed to HuggingFace Hub with a dataset card.
2. Comet ML shows statistics at each pipeline stage. 
3. The quality filter passes manual inspection on some random samples.
4. At least 2000 Q&A pairs in final dataset after filtering.

## 5a. Tests I Will Write
*[How I will verify the Definition of Done programmatically.]*

- `test_quality_filter`: verifies random samples are in ChatMl format & properly grounded
- `test_dataset_exist`: verifies if dataset exists on hugging face

## 6. Production Boundaries

### What is deterministic
*[List every part that must never involve the LLM or randomness.]*

1. The ArXiv API Scraper
2. Cleaning pipeline
3. Quality filter
4. Logging

### Human inspection point
*[Where can a human look to see exactly what the system did?]*

At each stage, the user can check the respective output jsonl for that stage (e.g., outputs/final_qa_pairs.jsonl to check final output). Also, Comet Ml interface can be used to check logged statistics for each pipeline stage. The uploaded dataset with its datacard can also be checked online on Hf hub.


### State representation
*[How does the system store and represent state? Must be simple and inspectable by a human opening a file.]*

The dataset is stored in jsonl with each Q&A pair represented per line.

### Serial vs parallel
*[Default is serial. If anything is parallel, justify it explicitly here.]*

since each stage depends on previous data & steps, pipeline processing are treated serially



## 7. Pre-Build Questions
*[Answer every project-specific question here before writing code.]*


**What makes a Q&A pair 'high quality' for fine-tuning? What specific properties must it have, and what makes it unusable?**

A high-quality Q&A pair must have: minimum token for question will be 20, justification is question should be less than entire abstract length but more than a few sentences for reasonableness. With at least 20 tokens, if we assume 20 words, that's more than enough question length. Minimum token count for answer will be say 25, typically answers are responses and are expected to be longer than questions. Python check to verify Q&A format: to verify that a pair is correct, a Python function checks if role exists and if first content is question and next one is answer and if second role is assistant, and returns true, else false. For groundedness check, we check overlap with abstract, a deterministic one, so answer should have at least 75% overlap with abstract. 75% is a good benchmark since answers are supposed to be directly gotten from abstract, then we expect a significant overlap more than 50% to be sure abstract is serving as base for answer source, and we leave remaining 25% to LLM randomness.

**Your scraper hits the ArXiv API 2000 times. What happens if it fails at request 1500? How do you resume without restarting from zero?**

After each scrape, we save the scraped data to a file to prevent us from scraping afresh. We log the total number of papers scraped. When resuming, we skip paper ids that are already in the log, and we scrape the remaining number of papers needed.

**A paper has 50 tokens. Another has 3000 tokens. Why does your length filter exist, and what is the right threshold?**

Length filter exists to ensure high quality dataset. For a paper with less than 50 tokens, generating high quality Q&A may be difficult because the words may not be sufficient to generate a reasonable question and corresponding answer. For a paper with 3000 tokens, word count may be unnecessarily too long to generate the right question and answer, since long word count may confuse the model and cause it to generate a question from an unimportant part of the abstract. Right threshold will be between min 100 tokens, max 2048.

**You generate Q&A pairs with an LLM. How do you know the generated answer is actually grounded in the abstract?**

To know the generated answer is grounded in the abstract, I will check for overlap of keywords in the answer with the abstract, and accept 75% overlap. Overlap check uses content words only, stopwords like "the", "is", "and" are excluded

**Your quality filter rejects 40% of pairs. How do you know the filter is working correctly and not being too aggressive?**

If quality filter rejects 40% of the pairs, we know the filter is working correctly. First, I will manually check a random sample of rejected pairs to see if their overlap proportion is less than 75%. If that is the case, the filter is okay. For comparison, we can still compare the overlap proportion, that is the metric to compare. To check if a filter is too aggressive: if a large proportion of randomly sampled filtered pairs make sense visually and if their overlap proportion is close to 75%, then the filter is aggressive. Also, if we have less than 50% of cleaned data remaining, the filter is aggressive too.


## 8. Known Limitations
*[What does this system not handle? What would break it that you are aware of right now?]*

**What does this system not handle?**

1. Papers with non-English abstracts that pass language detection
2. Abstracts that are technically long enough but contain no answerable content

**What would break it that you are aware of right now?**

1. ArXiv API rate limiting during scraping
