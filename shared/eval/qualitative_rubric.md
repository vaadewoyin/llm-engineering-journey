# Qualitative Evaluation Rubric (Locked)

After training, each answer from the fine-tuned model is scored on five dimensions. Each dimension gets a score of 1 to 5.

## Five Dimensions

### 1. Factual Correctness
Is the answer factually accurate given the source papers?
- **5** – Completely accurate, no false claims
- **4** – Accurate with a minor imprecision
- **3** – Mostly accurate, one notable error
- **2** – Partially accurate, multiple errors
- **1** – Factually wrong or hallucinated

### 2. Relevance
Does the answer actually address what the question asked?
- **5** – Directly answers the question, nothing extra
- **4** – Mostly on topic, minor drift
- **3** – Partially relevant, misses part of the question
- **2** – Barely on topic, misses the core
- **1** – Does not address the question at all

### 3. Fluency
Is the answer coherent and well-formed as text?
- **5** – Clear, well-structured, natural language
- **4** – Readable, slightly awkward in places
- **3** – Understandable but noticeably rough
- **2** – Difficult to read, frequent issues
- **1** – Incoherent or broken output

### 4. Groundedness
Is every claim traceable to the retrieved/source content, not invented?
- **5** – All claims grounded in the source
- **4** – Mostly grounded, one unsupported claim
- **3** – Some claims unsupported but not fabricated
- **2** – Several claims appear invented
- **1** – Largely ungrounded, fabricated content

### 5. Standards / Technical Accuracy
Are the technical details and any referenced standards correct (e.g. ASTM, BS EN, mix-design numbers, material properties)?
- **5** – All technical details and standards correct
- **4** – Minor technical imprecision, no wrong standard cited
- **3** – One wrong value or misattributed standard
- **2** – Multiple technical errors
- **1** – Technically wrong throughout