# Sapia pretraining data (v0)

Status: **implemented** in `sapia/sources.py` (sources, filters, formatting) and
`sapia/prepare_mix.py` (preparation). A tiny end-to-end run against the real datasets works;
the full ~15B-token preparation has not been run yet.

Goal: an English model that is **accurate** (knowledge, reasoning) and **conversational**
(natural back-and-forth, explanations to a person, everyday language), within a budget of
~15B tokens on one GH200 (see [ARCHITECTURE.md](ARCHITECTURE.md)). The exact total is set after
the throughput trial; the shares below stay fixed and scale with it.

## Main mix (first ~90% of tokens, ~13.5B)

| Share | Tokens | Source | Role |
|---:|---:|---|---|
| 45% | ~6.1B | **FineWeb-Edu** (score >= 3) | Accuracy: textbook-style explanations, the knowledge backbone |
| 25% | ~3.4B | **DCLM-Baseline** | Natural, question-and-answer-style web text (its quality filter was trained on instruction data and ELI5 answers) |
| 7% | ~0.9B | **Wikipedia** (English) | Dense, checked facts |
| 6.5% | ~0.9B | **StackExchange** (English sites except StackOverflow and meta sites; answers with score >= 1) | Expert Q&A: a question, then a careful answer, with votes as a quality signal |
| 6.5% | ~0.9B | **Reddit threads** (50 curated subreddits, threads of 2+ messages, average score >= 1) | Casual multi-turn conversation, opinions, everyday language |
| 5% | ~0.7B | **FineMath** (score 3; the 4+ tier is kept for the final phase) | Step-by-step math reasoning |
| 5% | ~0.7B | **Project Gutenberg books** (English, dialogue-rich: >= 1 quote mark per 1000 characters, as a fiction/drama proxy) | Long, coherent narrative with written dialogue; plays are almost pure dialogue |

Roughly: **57% accuracy-oriented** (Edu, Wikipedia, FineMath), **38% conversational**
(DCLM, StackExchange, Reddit), **5% narrative**.

Books are kept modest because the texts are mostly pre-1930 (public domain): more of them pushes
archaic style, spelling and outdated facts and views into a model meant to be accurate and to
chat in modern English. Books are also ~100K tokens each, so at 2048 context they are cut into
many short chunks and their long-range structure is not used. PG-19 holds ~2.5-3B tokens, so
5% is about a quarter of it; prefer fiction and plays over sermons, almanacs and reference works.
Openly licensed modern books at scale do not exist (Books3 and similar sets are pirated and
excluded).

Wikipedia gets ~0.9B tokens here plus ~0.15B in the final phase, about a quarter of English
Wikipedia, so it is still seen at most once.

## Final phase (last ~10% of tokens, ~1.5B): anneal on the best data

While the learning rate decays (warmup-stable-decay schedule), switch to the highest-quality
slices. This is a cheap, well-established gain, and it is where conversational ability is
sharpened just before chat fine-tuning.

| Share | Source |
|---:|---|
| 30% | FineWeb-Edu, top tier only (score >= 4) |
| 20% | Synthetic multi-turn dialogues (UltraChat), in the SmolLM2 chat template |
| 15% | StackExchange, high-scoring answers only |
| 10% | Reddit, high-scoring threads only |
| 10% | Wikipedia |
| 10% | FineMath, top tier (FineMath-4+) |
| 5% | DCLM-Baseline |

SmolTalk stays reserved for chat fine-tuning (SFT) after pretraining, so it is not mixed in here.

## Formatting

- Every document ends with the tokenizer's end-of-text token; documents are packed into 2048-token sequences.
- **Reddit and StackExchange**: plain text with simple speaker lines, e.g.
  `Question: ...` / `Answer (score 42): ...`, or `User A: ...` / `User B: ...`. Usernames are
  removed.
- **UltraChat (final phase only)**: SmolLM2's chat template (`<|im_start|>user ... <|im_end|>`),
  so the model already knows the turn markers when chat fine-tuning starts.
- No document appears in both phases: big sources (FineWeb-Edu, DCLM) give the final phase every
  10th file and the main mix the other nine; small sources (Wikipedia, StackExchange, Reddit) are
  split by a hash of each document's id (90% main, 10% final phase).
- If a source runs out before its target, the loader repeats it so its share holds; prep warns.

## Where each source comes from

All are parquet on the Hugging Face Hub, pinned to the commit seen on the first run.

| Source | Dataset | Files |
|---|---|---|
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | `sample/100BT/*.parquet` |
| DCLM-Baseline | `mlfoundations/dclm-baseline-1.0-parquet` | `filtered/**/*.parquet` |
| Wikipedia | `wikimedia/wikipedia` | `20231101.en/*.parquet` |
| StackExchange | `HuggingFaceH4/stack-exchange-preferences` | `data/*/*.parquet` (HTML converted to plain text) |
| Reddit | `HuggingFaceGECLM/REDDIT_threaded` | `data/*.parquet` |
| FineMath | `HuggingFaceTB/finemath` | `finemath-3plus` (main), `finemath-4plus` (final phase) |
| Gutenberg | `manu/project_gutenberg` | `data/en-*.parquet` (licence header and footer removed) |
| UltraChat | `HuggingFaceH4/ultrachat_200k` | `data/train_sft-*.parquet` |

Two changes from the first plan, found while checking the sources:

- **Reddit**: Dolma's Reddit data is single comments, not conversations. `REDDIT_threaded` has
  whole threads from 50 mostly high-quality subreddits (AskHistorians, askscience,
  explainlikeimfive, changemyview, personalfinance, ...), which fits the conversational goal better.
- **StackExchange**: StackOverflow is ~70% of the data and almost all code, so it is left out
  (v0 has no code). The remaining sites hold ~6 GB of posts, enough for both phases.

## Licenses

| Source | License | Note |
|---|---|---|
| FineWeb-Edu, FineMath | ODC-By | Attribution |
| DCLM-Baseline | CC-BY-4.0 | Attribution |
| StackExchange, Wikipedia | CC-BY-SA | Attribution, share-alike |
| Reddit (REDDIT_threaded) | Reddit's own terms apply to the content | Fine for personal/research use; reconsider before releasing weights commercially |
| Project Gutenberg | Public domain texts | — |
| UltraChat | MIT | Generated with ChatGPT; OpenAI's terms restrict using outputs to build competing models |

Sapia is internal (one user, weights not released), so all of these are fine as they are.
Reddit's terms and UltraChat's OpenAI-output restriction would only need another look if the
weights were ever released or used commercially.

## Open questions

1. **Code**: none in v0. Adding ~5-10% would give basic coding ability, at the cost of a harder
   download (Stack-Edu needs the files fetched separately from Software Heritage).
2. **Gutenberg filter**: the quote-density proxy also lets some biographies through. Raise
   `min_quote_density` in `sapia/sources.py` if books turn out too non-fictional.
