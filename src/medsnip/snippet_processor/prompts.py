"""Prompts for the snippet processor.

Two modes:

  - **atom-to-snippet** (default): two LLM calls.
      1. extract: query + numbered sentences  → atoms + shared_context
      2. cluster: query + atoms + shared_context → snippets
      Mirrors the human annotation workflow (annotators saw extracted
      atoms and grouped them).

  - **snippet-direct**: one LLM call.
      query + numbered sentences → snippets + shared_context
      Cheaper; slightly weaker on consumer-style dense responses.

Both modes follow the same merge guidelines (Patterns A–F) and use the same
subset-specific few-shot examples (e1 consumer, e61 vignette).
"""

import json

# ---------------------------------------------------------------------------
# §1 + §4 — shared merge guidelines (Patterns A–F)
# ---------------------------------------------------------------------------

SHARED_GUIDELINES = """\
You are mirroring the work of a medical-text annotator on the MedFactCheck \
project. Your job: take an LLM-generated medical response and group its content \
into **snippets** — coherent, self-contained, verifiable units of medical \
information.

# Why merge at all
Atomic claim extraction is often too fine-grained: a single atom like \
"The factors include the individual's age" is unverifiable in isolation — it \
needs the surrounding reasoning. Medical responses chain claims with \
because/since/therefore. Splitting destroys the verifiable unit.

**Guiding question for every claim:** *If this stood alone, could a reader \
verify it medically?* If no → merge with neighbors that carry the missing \
context.

# Three patterns that trigger MERGE

**Pattern A — Enumeration of properties of one subject.** The response lists \
multiple features, factors, or properties of the *same* thing. Merge the list, \
even if each item is independently verifiable.

**Pattern B — Causal or conditional chain.** Items connected by \
because/since/as/due to/requires/therefore. Merge so the logical link survives.

**Pattern C — Conclusion + supporting premises.** A recommendation/diagnosis \
that is only meaningful with its reasoning.

# Three patterns that justify KEEPING ATOMIC

**Pattern D — Standalone fact, definition, or lab interpretation.** Reads \
self-contained; no surrounding text needed.

**Pattern E — Topic genuinely shifts.** Response moves to an unrelated \
subject. *Never merge across subjects.*

**Pattern F — Same topic, different checkable facts.** Two claims about the \
same drug can stay separate.

# When torn → default to atomic. If you cannot articulate which of Patterns \
A/B/C is firing in one sentence, do not merge.

# Coverage requirement
**Every unit (every sentence in `ft` mode, every atom in `atom` mode) must \
appear in exactly one snippet — no drops, no duplicates.**
"""


# ---------------------------------------------------------------------------
# §3 — shared_context schema (subset-specific)
# ---------------------------------------------------------------------------

CONSUMER_CONTEXT_SCHEMA = """\
# Shared context schema (consumer query)
Extract a small key-value dict capturing the topic. Typical keys:
  topic, drug_a, drug_b, condition, symptom
Omit keys that don't apply. Short concrete values.
"""

VIGNETTE_CONTEXT_SCHEMA = """\
# Shared context schema (clinical vignette)
Extract a key-value dict capturing the patient's clinical context. Standard \
keys:
  age, sex, chief_complaint, medical_history, medications, vitals,
  exam_findings, treatment_history, correct_answer
Omit keys not present in the case. Short concrete values.
"""


# ---------------------------------------------------------------------------
# §5 — snippet text rules
# ---------------------------------------------------------------------------

CONSUMER_TEXT_RULES = """\
# Snippet text rules (consumer)
Each snippet's `output` must be a self-contained verifiable statement. A \
reader should judge correctness without the query. Do NOT introduce facts \
absent from the source.
"""

VIGNETTE_TEXT_RULES = """\
# Snippet text rules (vignette)
Each snippet's `output` must be self-contained. **Inline relevant patient \
demographics from shared_context** so a reviewer can judge it standalone \
(e.g., "this patient" → "this 55-year-old male with right arm weakness"). \
Do NOT introduce facts absent from the source.
"""


# ---------------------------------------------------------------------------
# Step 1 (ft pipeline) — sentence-grounded claim extraction
# ---------------------------------------------------------------------------

EXTRACT_COMMON = """\
You are an atomic-claim extractor for medical fact-checking. Given a user \
query and an LLM medical response that has been pre-segmented into numbered \
sentences, produce:
  1. A `shared_context` dict summarizing the topic/condition (schema below).
  2. A list of **atomic claims** that follow the response's sentence \
structure.

# Definition of an atomic claim
Each atom states ONE medically-meaningful proposition, light-normalized from \
the sentence(s) it comes from. Resolve pronouns where needed (e.g., "it" → \
"Benadryl"). Exclude citation markers and meta-text. \
**Preserve the response's stance — do NOT correct claims that look wrong.**

# Sentence ↔ atom mapping
Each atom must point back to the sentence(s) it was derived from via \
`source_sentences`. Usually:
  - One sentence → one atom (default).
  - One sentence → multiple atoms only if the sentence is a clear \
comma-separated enumeration of distinct outcomes/items (consumer-style); each \
item becomes its own atom citing the same sentence index.
  - Two consecutive sentences → one atom is rare; allowed only if they carry \
one indivisible claim that cannot stand alone.

# Coverage requirement
**Every sentence index must appear in at least one atom's `source_sentences`.** \
No content sentence may be dropped.
"""

CONSUMER_EXTRACT_TASK = """

# Granularity — CONSUMER (fine)
Consumer responses are short and dense. Use fine-grained atoms:
  - "Taking X and Y together can increase the risk of drowsiness, dizziness, \
and confusion" → THREE atoms, one per side effect, all citing the same \
sentence.
  - "Drug X is an antihistamine that contains compound Z" → ONE atom.
  - Single-property sentences → ONE atom.
"""

VIGNETTE_EXTRACT_TASK = """

# Granularity — VIGNETTE (coarse — one sentence ≈ one atom)
Vignette responses chain clinical reasoning. Default: ONE atom per sentence.
  - Causal chains and multi-property sentences stay as one atom.
  - The final-answer sentence is its own atom.
  - Each differential-option sentence becomes one atom citing one sentence.
"""

def _extract_output(context_schema: str) -> str:
    return f"""

# shared_context schema
{context_schema}

# Output
Output ONLY this JSON object (no prose, no code fences):

{{
  "shared_context": {{... per the schema above ...}},
  "atoms": [
    {{"index": 1, "text": "<atomic claim>", "source_sentences": [<int>, ...]}},
    {{"index": 2, "text": "<atomic claim>", "source_sentences": [<int>, ...]}}
  ]
}}
"""


EXTRACT_SYSTEM_PROMPTS = {
    "consumer": EXTRACT_COMMON + CONSUMER_EXTRACT_TASK + _extract_output(CONSUMER_CONTEXT_SCHEMA),
    "vignette": EXTRACT_COMMON + VIGNETTE_EXTRACT_TASK + _extract_output(VIGNETTE_CONTEXT_SCHEMA),
}


def _system_snippet_direct(context_schema: str, text_rules: str) -> str:
    return f"""{SHARED_GUIDELINES}
{context_schema}
{text_rules}
# Task — single-pass sentence-grounded snippet processor
You will be given:
  - `query`     : the user's question.
  - `sentences` : a deterministic numbered list of sentences extracted from \
the LLM response.

Do everything in one pass:
  1. Extract a `shared_context` key-value dict (schema above).
  2. Identify atomic verifiable claims in each sentence and group them into \
snippets per Patterns A–F. Every sentence index must appear in at least one \
snippet's `source_sentences`.
  3. Write each snippet's `output` per the text rules above (decontextualize, \
no pronouns).
  4. Tag each snippet with the dominant pattern (A–F) and a short `notes` \
rationale.

# IMPORTANT — sentence boundary is the default
**The default is one sentence → one snippet.** Merge sentences *only* when one \
of these structural cues is present across them:
  - Pattern A: multiple sentences enumerate properties of the same subject.
  - Pattern B: an explicit causal/conditional link connects them \
(because/since/therefore/requires/leads to).
  - Pattern C: a conclusion sentence is unverifiable without its premise.

If none of A/B/C clearly fires, **split**. Topical similarity alone is NOT \
enough to merge. A common failure mode is collapsing the whole response into \
one snippet — do not do that. If the response has N sentences, the output \
usually has between 0.6N and N snippets.

Output ONLY this JSON object (no prose, no code fences):

{{
  "shared_context": {{... per the schema above ...}},
  "snippets": [
    {{
      "source_sentences": [<int>, ...],
      "output": "...",
      "pattern": "A|B|C|D|E|F",
      "notes": "..."
    }}
  ]
}}
"""


SNIPPET_DIRECT_SYSTEM_PROMPTS = {
    "consumer": _system_snippet_direct(CONSUMER_CONTEXT_SCHEMA, CONSUMER_TEXT_RULES),
    "vignette": _system_snippet_direct(VIGNETTE_CONTEXT_SCHEMA, VIGNETTE_TEXT_RULES),
}


# ---------------------------------------------------------------------------
# Cluster step (atom-to-snippet mode, step 2)
# ---------------------------------------------------------------------------

def _system_atom(context_schema: str, text_rules: str) -> str:
    return f"""{SHARED_GUIDELINES}
{context_schema}
{text_rules}
# Task — atom-grounded snippet clustering
You will be given:
  - `query`           : the user's question.
  - `shared_context`  : a dict already extracted from the response.
  - `atomic_claims`   : a numbered list of pre-extracted atomic claims.

Your job is to GROUP the atoms (the `shared_context` was already produced in
the extraction step — use it; don't re-emit it):

  1. Group atoms into snippets per Patterns A–F. Every atom index appears in \
EXACTLY ONE snippet.
  2. Write each snippet's `output` per the text rules above.
  3. Tag each snippet with the dominant pattern (A–F) and a short `notes` \
rationale.

Output ONLY this JSON object (no prose, no code fences):

{{
  "snippets": [
    {{
      "source_claims": [<int>, ...],
      "output": "...",
      "pattern": "A|B|C|D|E|F",
      "notes": "..."
    }}
  ]
}}
"""


CLUSTER_SYSTEM_PROMPTS = {
    "consumer": _system_atom(CONSUMER_CONTEXT_SCHEMA, CONSUMER_TEXT_RULES),
    "vignette": _system_atom(VIGNETTE_CONTEXT_SCHEMA, VIGNETTE_TEXT_RULES),
}

# `atom` mode (research upper bound) reuses the cluster prompt directly.
# `ft` mode is now a two-step pipeline (extract -> cluster); its cluster step
# also uses these same prompts.
SYSTEM_PROMPTS = {
    ("consumer", "atom"): CLUSTER_SYSTEM_PROMPTS["consumer"],
    ("vignette", "atom"): CLUSTER_SYSTEM_PROMPTS["vignette"],
    ("consumer", "ft"):   CLUSTER_SYSTEM_PROMPTS["consumer"],
    ("vignette", "ft"):   CLUSTER_SYSTEM_PROMPTS["vignette"],
}


# ---------------------------------------------------------------------------
# Few-shot examples — Hasan's batches (e1 consumer, e61 vignette)
# ---------------------------------------------------------------------------

# Consumer e1: 4 sentences in full_text, grouped into 5 snippets via gold atoms.
# In sentence-grounded view, the 4 sentences map to snippets like this:
#   S0 [sent 0]: recommendation + premise
#   S1 [sent 1]: drug A composition + drug B composition (in one sentence!)
#   S2 [sent 2]: side effects enumeration
#   S3 [sent 3]: overdose causal chain
# So 4 sentences ↦ 4 snippets in ft mode.

CONSUMER_FEW_SHOT_FT_USER = """### Query
What is the most safe amount of Advil to take at one time?

### Full response
The safe amount of Advil (ibuprofen) to take at one time depends on several factors, including the individual's age, weight, and overall health. For adults and children over 12 years old, the typical over-the-counter dose is 200-400 mg every 4-6 hours as needed for pain or fever [1]. The maximum daily dose should not exceed 1200 mg unless directed by a healthcare provider [2]. For children under 12 years old, dosing is based on weight, and it is essential to follow the dosing instructions provided by a healthcare provider or the product label [3].

### Sentences (4)
0. The safe amount of Advil (ibuprofen) to take at one time depends on several factors, including the individual's age, weight, and overall health
1. For adults and children over 12 years old, the typical over-the-counter dose is 200-400 mg every 4-6 hours as needed for pain or fever
2. The maximum daily dose should not exceed 1200 mg unless directed by a healthcare provider
3. For children under 12 years old, dosing is based on weight, and it is essential to follow the dosing instructions provided by a healthcare provider or the product label

Output the JSON described in the system prompt."""

CONSUMER_FEW_SHOT_FT_ASSISTANT = """{
  "shared_context": {
    "topic": "safe amount of Advil to take at one time",
    "drug_a": "Advil (ibuprofen)",
    "condition": "pain or fever"
  },
  "snippets": [
    {
      "source_sentences": [0],
      "output": "The safe amount of Advil (ibuprofen) to take at one time depends on several factors, including the individual's age, weight, and overall health.",
      "pattern": "A",
      "notes": "Enumeration of factors that determine safe dose; one subject (the safe amount)."
    },
    {
      "source_sentences": [1],
      "output": "For adults and children over 12 years old, the typical over-the-counter dose of Advil (ibuprofen) is 200-400 mg every 4-6 hours as needed for pain or fever.",
      "pattern": "D",
      "notes": "Standalone dosing fact for adults and 12+."
    },
    {
      "source_sentences": [2],
      "output": "The maximum daily dose of Advil (ibuprofen) for adults and children over 12 should not exceed 1200 mg unless directed by a healthcare provider.",
      "pattern": "D",
      "notes": "Standalone maximum-daily-dose fact."
    },
    {
      "source_sentences": [3],
      "output": "For children under 12 years old, Advil (ibuprofen) dosing is based on weight, and it is essential to follow the dosing instructions provided by a healthcare provider or the product label.",
      "pattern": "B",
      "notes": "Conditional rule + instruction; merged because each piece needs the other for verifiability."
    }
  ]
}"""


# Vignette e61: long differential analysis. The full_text has ~34 sentences;
# Hasan grouped the 17 atoms into 8 snippets. In sentence-grounded view we
# group sentences by differential diagnosis and final conclusion.

VIGNETTE_FEW_SHOT_FT_USER = """### Query
A 55-year-old male bodybuilder presents to the ED with weakness of his right arm. He has diabetes, drinks 2-7 alcoholic drinks/day, smokes 2 packs/day since age 25, uses anabolic steroids, and has lost 17 lb in the last month. BP 177/108. Exam: decreased sensation right arm, 2/5 strength right arm, 5/5 left arm; dull aching/burning right arm pain; symptoms unchanged with head/neck position. Which is the most likely diagnosis? (A) Apical lung tumor (B) Brachial plexopathy (C) Cerebral infarction (D) Scalenus anticus syndrome (E) Subclavian steal syndrome

### Full response
[The differential discusses each option in turn and concludes with cerebral infarction; full text abbreviated for example.]

### Sentences (10, condensed)
0. The patient's history of smoking and hypertension raises the risk for vascular events.
1. The profound weakness and sensory changes suggest potential central nervous system involvement.
2. Apical lung tumor (Pancoast tumor) may present with shoulder pain and muscle wasting.
3. Pancoast tumor often affects the ulnar nerve distribution.
4. The patient's symptoms are not consistent with Horner syndrome, which often accompanies Pancoast tumors.
5. Brachial plexopathy can present with weakness and sensory deficits in the arm.
6. Brachial plexopathy often results from trauma, compression, or radiation.
7. The absence of positional changes somewhat argues against brachial plexopathy.
8. Cerebral infarction can cause sudden onset of unilateral weakness and sensory loss.
9. The most likely diagnosis is cerebral infarction (stroke).

Output the JSON described in the system prompt."""

VIGNETTE_FEW_SHOT_FT_ASSISTANT = """{
  "shared_context": {
    "age": "55",
    "sex": "male",
    "chief_complaint": "right arm weakness over a few weeks; dropped a cup of tea today",
    "medical_history": "diabetes; bodybuilder using anabolic steroids; 2 packs/day smoking since age 25; 2-7 alcoholic drinks/day; 17 lb weight loss over the last month",
    "vitals": "BP 177/108, pulse 90, RR 17, O2 98% RA",
    "exam_findings": "decreased sensation in right arm, 2/5 strength right arm, 5/5 left arm; symptoms unchanged with head/neck position",
    "correct_answer": "Cerebral infarction (C)"
  },
  "snippets": [
    {
      "source_sentences": [0],
      "output": "The 55-year-old male's history of heavy smoking (2 packs/day since age 25) and hypertension (BP 177/108) raises the risk for vascular events.",
      "pattern": "D",
      "notes": "Standalone risk-factor statement; demographics inlined."
    },
    {
      "source_sentences": [1],
      "output": "In a 55-year-old male with right arm weakness over a few weeks, decreased sensation, and 2/5 strength in the right arm, the profound weakness and sensory changes suggest potential central nervous system involvement.",
      "pattern": "D",
      "notes": "Standalone clinical impression; patient details inlined."
    },
    {
      "source_sentences": [2, 3, 4],
      "output": "Apical lung tumor (Pancoast tumor) may present with shoulder pain, muscle wasting, and often affects the ulnar nerve distribution, but this 55-year-old male's symptoms are not consistent with Horner syndrome, which often accompanies Pancoast tumors.",
      "pattern": "A",
      "notes": "All properties of one subject (Pancoast tumor)."
    },
    {
      "source_sentences": [5, 6, 7],
      "output": "Brachial plexopathy can present with weakness and sensory deficits in the arm and often results from trauma, compression, or radiation; in this 55-year-old male, the absence of positional changes somewhat argues against brachial plexopathy.",
      "pattern": "A",
      "notes": "All properties of one subject (brachial plexopathy)."
    },
    {
      "source_sentences": [8],
      "output": "Cerebral infarction can cause sudden onset of unilateral weakness and sensory loss.",
      "pattern": "D",
      "notes": "Standalone differential characterization."
    },
    {
      "source_sentences": [9],
      "output": "In this 55-year-old male, the most likely diagnosis is cerebral infarction (stroke).",
      "pattern": "C",
      "notes": "Final answer with patient context inlined."
    }
  ]
}"""


# atom-mode few shots (from earlier work — kept for the research upper bound)

CONSUMER_FEW_SHOT_ATOM_USER = """## Example task (consumer, atom mode)

### Query
Can I take nyquil and benedryl at the same time?

### Shared context (from extract step)
{
  "topic": "Taking NyQuil and Benadryl at the same time",
  "drug_a": "NyQuil",
  "drug_b": "Benadryl"
}

### Atomic claims (11)
1. It is not recommended to take NyQuil and Benadryl at the same time.
2. Both NyQuil and Benadryl contain antihistamines.
3. NyQuil usually contains doxylamine.
4. Benadryl contains diphenhydramine.
5. Taking NyQuil and Benadryl together can increase the risk of side effects.
6. Taking NyQuil and Benadryl together can increase the risk of severe drowsiness.
7. Taking NyQuil and Benadryl together can increase the risk of dizziness.
8. Taking NyQuil and Benadryl together can increase the risk of confusion.
9. Taking NyQuil and Benadryl together can increase the risk of impaired coordination.
10. Combining NyQuil and Benadryl can potentially lead to an overdose of antihistamines.
11. An overdose of antihistamines can be dangerous.

Output the JSON described in the system prompt."""

CONSUMER_FEW_SHOT_ATOM_ASSISTANT = """{
  "snippets": [
    {"source_claims": [1, 2], "output": "It is not recommended to take NyQuil and Benadryl at the same time because both contain antihistamines.", "pattern": "C", "notes": "Recommendation + premise."},
    {"source_claims": [3], "output": "NyQuil usually contains doxylamine.", "pattern": "D", "notes": "Standalone composition."},
    {"source_claims": [4], "output": "Benadryl contains diphenhydramine.", "pattern": "D", "notes": "Different drug (Pattern E topic shift); kept atomic."},
    {"source_claims": [5, 6, 7, 8, 9], "output": "Taking NyQuil and Benadryl together can increase the risk of side effects including severe drowsiness, dizziness, confusion, and impaired coordination.", "pattern": "A", "notes": "Enumeration of side effects."},
    {"source_claims": [10, 11], "output": "Combining NyQuil and Benadryl can lead to an antihistamine overdose, and an antihistamine overdose can be dangerous.", "pattern": "B", "notes": "Causal chain."}
  ]
}"""

VIGNETTE_FEW_SHOT_ATOM_USER = """## Example task (vignette, atom mode)

### Query
A 55-year-old male bodybuilder presents to the ED with right arm weakness for several weeks. He has diabetes, heavy smoking, alcohol use, anabolic steroid use, recent weight loss, and hypertension. Exam: decreased sensation right arm, 2/5 strength right arm. Symptoms unchanged with head/neck position. Which is the most likely diagnosis? (A) Apical lung tumor (B) Brachial plexopathy (C) Cerebral infarction (D) Scalenus anticus syndrome (E) Subclavian steal syndrome

### Shared context (from extract step)
{
  "patient_age": 55,
  "patient_sex": "male",
  "presenting_complaint": "right arm weakness over several weeks",
  "key_history": "diabetes, heavy smoking, alcohol use, anabolic steroid use, weight loss, hypertension",
  "exam": "decreased sensation right arm, 2/5 strength right arm, symptoms unchanged with head/neck position",
  "question_type": "most likely diagnosis"
}

### Atomic claims (17)
1. The patient's history of smoking and hypertension raises the risk for vascular events.
2. The profound weakness and sensory changes suggest potential central nervous system involvement.
3. Apical lung tumor (Pancoast tumor) may present with shoulder pain and muscle wasting.
4. Pancoast tumor often affects the ulnar nerve distribution.
5. The patient's symptoms are not consistent with Horner syndrome, which often accompanies Pancoast tumors.
6. Brachial plexopathy can present with weakness and sensory deficits in the arm.
7. Brachial plexopathy often results from trauma, compression, or radiation.
8. The absence of positional changes somewhat argues against brachial plexopathy.
9. Cerebral infarction (stroke) can cause sudden onset of unilateral weakness and sensory loss.
10. The acute exacerbation of symptoms aligns well with cerebral infarction.
11. Scalenus Anticus Syndrome is a type of thoracic outlet syndrome.
12. Scalenus Anticus Syndrome typically presents with positional changes in symptoms.
13. Scalenus Anticus Syndrome is less likely given the patient's presentation.
14. Subclavian Steal Syndrome can cause arm weakness.
15. Subclavian Steal Syndrome usually presents with arm claudication, dizziness, or syncope due to reversed blood flow in the vertebral artery.
16. Subclavian Steal Syndrome is less likely without these additional symptoms.
17. The most likely diagnosis is cerebral infarction (stroke).

Output the JSON described in the system prompt."""

VIGNETTE_FEW_SHOT_ATOM_ASSISTANT = """{
  "snippets": [
    {"source_claims": [1], "output": "The 55-year-old male's history of heavy smoking and hypertension raises the risk for vascular events.", "pattern": "D", "notes": "Standalone risk-factor."},
    {"source_claims": [2], "output": "In this 55-year-old male with right arm weakness, decreased sensation, and 2/5 strength in the right arm, the profound weakness and sensory changes suggest potential CNS involvement.", "pattern": "D", "notes": "Clinical impression."},
    {"source_claims": [3, 4, 5], "output": "Apical lung tumor (Pancoast tumor) may present with shoulder pain, muscle wasting, and often affects the ulnar nerve distribution, but this 55-year-old male's symptoms are not consistent with Horner syndrome which often accompanies Pancoast tumors.", "pattern": "A", "notes": "One-subject enumeration."},
    {"source_claims": [6, 7, 8], "output": "Brachial plexopathy can present with weakness and sensory deficits in the arm and often results from trauma, compression, or radiation; in this 55-year-old male, the absence of positional changes argues against brachial plexopathy.", "pattern": "A", "notes": "One-subject enumeration."},
    {"source_claims": [9, 10], "output": "Cerebral infarction (stroke) can cause sudden onset unilateral weakness and sensory loss, and the acute exacerbation of right arm weakness in this 55-year-old male aligns well with cerebral infarction.", "pattern": "A", "notes": "Differential discussion + case alignment."},
    {"source_claims": [11, 12, 13], "output": "Scalenus anticus syndrome is a type of thoracic outlet syndrome that typically presents with positional changes; it is less likely in this 55-year-old male whose right arm symptoms are unchanged with head and neck position.", "pattern": "A", "notes": "One-subject enumeration."},
    {"source_claims": [14, 15, 16], "output": "Subclavian steal syndrome can cause arm weakness but usually presents with arm claudication, dizziness, or syncope due to reversed vertebral artery flow; in this 55-year-old male without those symptoms it is less likely.", "pattern": "A", "notes": "One-subject enumeration."},
    {"source_claims": [17], "output": "In this 55-year-old male with progressive right arm weakness, decreased sensation, and symptoms unchanged by head and neck position, the most likely diagnosis is cerebral infarction (stroke).", "pattern": "C", "notes": "Final answer with reasoning inlined."}
  ]
}"""


# ---------------------------------------------------------------------------
# Few-shot examples — claim extraction step (ft pipeline, step 1)
#
# Drawn from Hasan's batches: e17 (Advil consumer) and e61 (vignette).
# ---------------------------------------------------------------------------

CONSUMER_FEW_SHOT_EXTRACT_USER = """### Query
What is the most safe amount of Advil to take at one time?

### Sentences (4)
0. The safe amount of Advil (ibuprofen) to take at one time depends on several factors, including the individual's age, weight, and overall health
1. For adults and children over 12 years old, the typical over-the-counter dose is 200-400 mg every 4-6 hours as needed for pain or fever
2. The maximum daily dose should not exceed 1200 mg unless directed by a healthcare provider
3. For children under 12 years old, dosing is based on weight, and it is essential to follow the dosing instructions provided by a healthcare provider or the product label

Extract atomic claims."""

CONSUMER_FEW_SHOT_EXTRACT_ASSISTANT = """{
  "shared_context": {
    "topic": "safe amount of Advil to take at one time",
    "drug_a": "Advil (ibuprofen)",
    "condition": "pain or fever"
  },
  "atoms": [
    {"index": 1, "text": "The safe amount of Advil (ibuprofen) to take at one time depends on several factors.", "source_sentences": [0]},
    {"index": 2, "text": "The factors include the individual's age.", "source_sentences": [0]},
    {"index": 3, "text": "The factors include the individual's weight.", "source_sentences": [0]},
    {"index": 4, "text": "The factors include the individual's overall health.", "source_sentences": [0]},
    {"index": 5, "text": "For adults and children over 12 years old, the typical over-the-counter dose is 200-400 mg every 4-6 hours as needed for pain or fever.", "source_sentences": [1]},
    {"index": 6, "text": "The maximum daily dose for adults and children over 12 should not exceed 1200 mg unless directed by a healthcare provider.", "source_sentences": [2]},
    {"index": 7, "text": "For children under 12 years old, dosing is based on weight.", "source_sentences": [3]},
    {"index": 8, "text": "It is essential to follow the dosing instructions provided by a healthcare provider for children under 12 years old.", "source_sentences": [3]},
    {"index": 9, "text": "It is essential to follow the dosing instructions provided by the product label for children under 12 years old.", "source_sentences": [3]}
  ]
}"""

VIGNETTE_FEW_SHOT_EXTRACT_USER = """### Query
A 55-year-old male bodybuilder presents to the ED with right arm weakness over several weeks. Smoking, hypertension, weight loss, anabolic steroids. Symptoms unchanged with head/neck position. Which is the most likely diagnosis? (A) Apical lung tumor (B) Brachial plexopathy (C) Cerebral infarction (D) Scalenus anticus syndrome (E) Subclavian steal syndrome

### Sentences (10, condensed)
0. The patient's history of smoking and hypertension raises the risk for vascular events
1. The profound weakness and sensory changes suggest potential central nervous system involvement
2. Apical lung tumor (Pancoast tumor) may present with shoulder pain and muscle wasting
3. Pancoast tumor often affects the ulnar nerve distribution
4. The patient's symptoms are not consistent with Horner syndrome, which often accompanies Pancoast tumors
5. Brachial plexopathy can present with weakness and sensory deficits in the arm
6. Brachial plexopathy often results from trauma, compression, or radiation
7. The absence of positional changes somewhat argues against brachial plexopathy
8. Cerebral infarction can cause sudden onset of unilateral weakness and sensory loss
9. The most likely diagnosis is cerebral infarction (stroke)

Extract atomic claims."""

VIGNETTE_FEW_SHOT_EXTRACT_ASSISTANT = """{
  "shared_context": {
    "patient_age": 55,
    "patient_sex": "male",
    "presenting_complaint": "right arm weakness",
    "key_history": "smoking, hypertension, anabolic steroid use, weight loss",
    "exam": "decreased sensation right arm, 2/5 strength right arm",
    "question_type": "most likely diagnosis"
  },
  "atoms": [
    {"index": 1, "text": "The patient's history of smoking and hypertension raises the risk for vascular events.", "source_sentences": [0]},
    {"index": 2, "text": "The profound weakness and sensory changes suggest potential central nervous system involvement.", "source_sentences": [1]},
    {"index": 3, "text": "Apical lung tumor (Pancoast tumor) may present with shoulder pain and muscle wasting.", "source_sentences": [2]},
    {"index": 4, "text": "Pancoast tumor often affects the ulnar nerve distribution.", "source_sentences": [3]},
    {"index": 5, "text": "The patient's symptoms are not consistent with Horner syndrome, which often accompanies Pancoast tumors.", "source_sentences": [4]},
    {"index": 6, "text": "Brachial plexopathy can present with weakness and sensory deficits in the arm.", "source_sentences": [5]},
    {"index": 7, "text": "Brachial plexopathy often results from trauma, compression, or radiation.", "source_sentences": [6]},
    {"index": 8, "text": "The absence of positional changes somewhat argues against brachial plexopathy.", "source_sentences": [7]},
    {"index": 9, "text": "Cerebral infarction can cause sudden onset of unilateral weakness and sensory loss.", "source_sentences": [8]},
    {"index": 10, "text": "The most likely diagnosis is cerebral infarction (stroke).", "source_sentences": [9]}
  ]
}"""


# ---------------------------------------------------------------------------
# Public lookup
# ---------------------------------------------------------------------------

def few_shot(subset: str, step: str) -> tuple[str, str]:
    """Return the (user, assistant) one-shot example for the given step.

    `step` ∈ {"extract", "cluster", "snippet_direct"}.
    """
    if step == "extract":
        return (CONSUMER_FEW_SHOT_EXTRACT_USER, CONSUMER_FEW_SHOT_EXTRACT_ASSISTANT) \
            if subset == "consumer" \
            else (VIGNETTE_FEW_SHOT_EXTRACT_USER, VIGNETTE_FEW_SHOT_EXTRACT_ASSISTANT)
    if step == "snippet_direct":
        return (CONSUMER_FEW_SHOT_FT_USER, CONSUMER_FEW_SHOT_FT_ASSISTANT) \
            if subset == "consumer" \
            else (VIGNETTE_FEW_SHOT_FT_USER, VIGNETTE_FEW_SHOT_FT_ASSISTANT)
    # default: cluster step
    return (CONSUMER_FEW_SHOT_ATOM_USER, CONSUMER_FEW_SHOT_ATOM_ASSISTANT) \
        if subset == "consumer" \
        else (VIGNETTE_FEW_SHOT_ATOM_USER, VIGNETTE_FEW_SHOT_ATOM_ASSISTANT)


# ---------------------------------------------------------------------------
# User-message builders
# ---------------------------------------------------------------------------

def user_message_extract(query: str, sentences: list[str]) -> str:
    sent_lines = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))
    return (
        f"### Query\n{query}\n\n"
        f"### Sentences ({len(sentences)})\n{sent_lines}\n\n"
        f"Extract atomic claims."
    )


def user_message_snippet_direct(query: str, sentences: list[str]) -> str:
    sent_lines = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))
    return (
        f"### Query\n{query}\n\n"
        f"### Sentences ({len(sentences)})\n{sent_lines}\n\n"
        f"Decompose + group in one pass. Output the JSON described in the system prompt."
    )


def user_message_atom(query: str, atoms: list[dict],
                      shared_context: dict) -> str:
    atom_lines = "\n".join(f"{a['index']}. {a['text']}" for a in atoms)
    ctx_json = json.dumps(shared_context or {}, indent=2, ensure_ascii=False)
    return (
        f"### Query\n{query}\n\n"
        f"### Shared context (from extract step)\n{ctx_json}\n\n"
        f"### Atomic claims ({len(atoms)})\n{atom_lines}\n\n"
        f"Output the JSON described in the system prompt."
    )


# ---------------------------------------------------------------------------
# Subset classifier (consumer ≤ 31 words, vignette ≥ 112 words in §3 stats)
# ---------------------------------------------------------------------------

SUBSET_WORD_THRESHOLD = 60


def classify_subset(query: str) -> str:
    return "consumer" if len(query.split()) < SUBSET_WORD_THRESHOLD else "vignette"
