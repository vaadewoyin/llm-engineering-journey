# DESIGN.md — [Project Name]

> **Before you write:** every sentence in this file should name something
> specific to *this* project, *this* phase, or *this* number. If a sentence
> could be pasted into a different project's DESIGN.md unchanged, it is filler.
> Delete it.

---

## 1. Problem

*What question does this project answer, and what decision does the answer feed?*

[One paragraph. Name the who and the why. "Analysing X" is a topic, not a
problem. "I need to know whether A or B works on C hardware so I can choose D
for the serving line" is a problem.]

**Decision this project feeds:** [the downstream choice this makes possible]

---

## 2. Components

*List each component, its single responsibility, and where its output goes.*

- `[filename].py` — [what it does]. Output: `[path]`.
- `[filename].py` — [what it does]. Output: `[path]`.
- `[config_file]` — [what's in it AND what it's explicitly NOT for].

*Where do parameters/config live? Say it explicitly. Inline? External file?
Per-script dict? Ambiguity here causes bugs later.*

---

## 3. Component Communication

*How do components talk? Which reads which file? Which writes which file?
Which calls which?*

**Draw it first (three boxes, arrows between them), then describe the arrows:**

[Paragraph. Every claim names a file. "Script A loads config X, runs training,
writes checkpoints to Y. Script B reads Y, evaluates, writes metrics to Z."]

---

## 4. Failure Modes

*Ranked by likelihood. Not edge cases — the failures you will actually hit.*

*For each: **cause → symptom → mitigation**. Every mitigation is a specific
action, not a hope.*

1. **[Failure name.]** Cause: [X]. Symptom: [Y]. Mitigation: [Z].
2. **[Failure name.]** Cause: [X]. Symptom: [Y]. Mitigation: [Z].
3. **[Failure name.]** Cause: [X]. Symptom: [Y]. Mitigation: [Z].
4. **[Failure name.]** Cause: [X]. Symptom: [Y]. Mitigation: [Z].

---

## 5. Definition of Done

*A stranger must be able to verify every item. Every item is a **file path**,
a **number**, or a **passing test** — never "it works."*

- [ ] [File/path exists.]
- [ ] [Dashboard/metric shows X with real numbers.]
- [ ] [Specific number appears in a specific file.]
- [ ] [Test passes.]
- [ ] [Named artifact produced.]

---

## 5a. Tests I Will Write

*How the Definition of Done is verified programmatically. Tests that would
still pass if the code were wrong are not tests — they are decoration.*

- `test_[name]()`: asserts [specific behavior]. Fails if [specific wrong state].
- `test_[name]()`: asserts [specific behavior]. Fails if [specific wrong state].
- `test_[name]()`: asserts [specific behavior]. Fails if [specific wrong state].
- `test_[name]()`: asserts [specific behavior]. Fails if [specific wrong state].

---

## 6. Production Boundaries

### Is the LLM called at runtime?
*Yes or no. If no, say so — then name what this phase locks down for later
phases that DO call the LLM.*

[Answer.]

### What is deterministic (LLM never decides)
*Every part that must be plain code — no LLM, no randomness, no silent defaults.*

1. [Item.]
2. [Item.]
3. [Item.]

### Human inspection point
*Where can a human look to see exactly what the system did and why? Name a
dashboard, a file, or a log — not "the logs."*

[Answer.]

### State representation
*How is state stored? Must be plain, inspectable files. If a human opens it,
they must understand it without reading code.*

[Answer.]

### Serial vs parallel
*Default is serial. If anything is parallel, justify it: what state is shared?
can a race condition corrupt results? is the speed gain worth the debug cost?*

[Answer.]

---

## 7. Pre-Build Questions

*Every project-specific question goes here — and gets a real answer, before
any code. If you don't know the answer, write "prediction, to verify" and
mark your confidence. A marked guess is honest; a confident wrong answer is
not.*

**Q:** [Question.]

**A:** [Answer. Numbers, not adjectives. If it's a prediction, say so and
state confidence + what would falsify it.]

**Q:** [Question.]

**A:** [Answer.]

**Q:** [Question.]

**A:** [Answer.]

**Q:** [Question.]

**A:** [Answer.]

**Q:** [Question.]

**A:** [Answer.]

---

## 8. Known Limitations

*What does this system NOT do, and what would break it that you already know
about? Be honest — this section is the one a reviewer trusts most.*

**What this system does not handle:**

1. [Limitation. Where does the boundary sit?]
2. [Limitation.]

**What would break it, that I already know about:**

1. [Known fragility + why it wasn't fixed.]
2. [Known fragility + why it wasn't fixed.]

---

## Self-check before shipping

Run this against your own draft. Fix every unchecked box.

- [ ] Section 1 names a decision this project feeds, not just a topic
- [ ] Section 2 lists files that actually exist in the repo
- [ ] Section 3 names real files on both ends of every arrow
- [ ] Section 4 items are **ranked**, and every mitigation is an action
- [ ] Section 5 items are checkable by a stranger (path/number/test)
- [ ] Section 5a tests would actually fail if the code were wrong
- [ ] Section 6 answers "is the LLM called at runtime?" honestly
- [ ] Section 7 has real answers — no "no idea" left unaddressed
- [ ] Section 8 admits something real
- [ ] No typos
- [ ] Every sentence is false if pasted into a different project