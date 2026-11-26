"""The pretraining mix: where each source comes from and how a raw row becomes a document.

A *formatter* turns one dataset row (a dict of the columns it asks for) into a document:
``None`` (skip the row), a string, or a list of strings and :class:`Special` markers
(for chat data, where turn markers must be real special tokens). Formatters are looked up
by name so that data-prep worker processes can rebuild them from plain config values.

A :class:`SourceSpec` names a Hugging Face dataset, the files to read, the formatter and
its arguments, and which documents belong to it:

* ``file_mod`` / ``file_keep``: keep source file ``i`` only if ``i % file_mod in file_keep``.
  Used to give the main mix and the final (anneal) phase disjoint files of big sources.
* ``bucket``: keep a document only if ``crc32(key) % 100`` lies in ``[lo, hi)``. Used to
  split small sources (Wikipedia, Reddit, StackExchange) between the two phases, so the
  anneal phase never repeats a main-mix document.

The shares are the decisions recorded in docs/DATA.md.
"""
from __future__ import annotations

import html
import re
import zlib
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from typing import Any, Callable, Optional, Union


@dataclass(frozen=True)
class Special:
    """A special token inside a formatted document, by name: "im_start" | "im_end" | "eot"."""
    name: str


Doc = Union[str, list]          # a string, or a list of str / Special pieces
Formatter = Callable[[dict, dict], Optional[Doc]]


# ----------------------------------------------------------------------------- formatters

def fmt_text(row: dict, args: dict) -> Optional[Doc]:
    """Plain text, optionally filtered on an ``int_score`` column (FineWeb-Edu, FineMath)."""
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    score = row.get("int_score")
    if score is not None:
        if "min_int_score" in args and score < args["min_int_score"]:
            return None
        if "max_int_score" in args and score > args["max_int_score"]:
            return None
    return text


def fmt_wikipedia(row: dict, args: dict) -> Optional[Doc]:
    text, title = row.get("text"), row.get("title")
    if not isinstance(text, str) or len(text) < args.get("min_chars", 500):
        return None
    return f"{title}\n\n{text}" if title else text


class _HTMLToText(HTMLParser):
    """Small HTML-to-plain-text converter for StackExchange posts (keeps code and lists readable)."""

    BLOCK = {"p", "div", "pre", "blockquote", "ul", "ol", "table", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "hr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.pre = 0
        self.quote = 0

    def _newline(self, n: int = 2) -> None:
        text = "".join(self.out)
        stripped = text.rstrip(" ")
        have = len(stripped) - len(stripped.rstrip("\n"))
        if stripped and have < n:
            self.out.append("\n" * (n - have))

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "pre":
            self._newline()
            self.out.append("```\n")
            self.pre += 1
        elif tag == "code" and not self.pre:
            self.out.append("`")
        elif tag == "br":
            self.out.append("\n")
        elif tag == "li":
            self._newline(1)
            self.out.append("- ")
        elif tag == "blockquote":
            self._newline()
            self.quote += 1
        elif tag in self.BLOCK:
            self._newline()
        elif tag == "img":
            alt = dict(attrs).get("alt")
            if alt:
                self.out.append(f"[image: {alt}]")

    def handle_endtag(self, tag: str) -> None:
        if tag == "pre":
            self.pre = max(0, self.pre - 1)
            if not "".join(self.out).endswith("\n"):
                self.out.append("\n")
            self.out.append("```")
            self._newline()
        elif tag == "code" and not self.pre:
            self.out.append("`")
        elif tag == "blockquote":
            self.quote = max(0, self.quote - 1)
            self._newline()
        elif tag in self.BLOCK:
            self._newline()

    def handle_data(self, data: str) -> None:
        if not self.pre:
            data = re.sub(r"\s+", " ", data)
        if self.quote and not self.pre:
            data = data.replace("\n", "\n> ")
            if "".join(self.out).endswith("\n") or not self.out:
                data = "> " + data.lstrip()
        self.out.append(data)

    def text(self) -> str:
        text = "".join(self.out)
        text = re.sub(r"[ \t]+\n", "\n", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()


def html_to_text(markup: str) -> str:
    parser = _HTMLToText()
    parser.feed(markup or "")
    parser.close()
    return html.unescape(parser.text())


def fmt_stackexchange(row: dict, args: dict) -> Optional[Doc]:
    """A question with its best answers (by score), as plain text."""
    question = html_to_text(row.get("question") or "")
    if len(question) < args.get("min_question_chars", 30):
        return None
    min_score = args.get("min_answer_score", 1)
    answers = [a for a in (row.get("answers") or []) if (a.get("pm_score") or 0) >= min_score and a.get("text")]
    if not answers:
        return None
    answers.sort(key=lambda a: (a.get("pm_score") or 0, bool(a.get("selected"))), reverse=True)
    parts = [f"Question:\n{question}"]
    for a in answers[: args.get("max_answers", 3)]:
        body = html_to_text(a["text"])
        if body:
            label = "Accepted answer" if a.get("selected") else "Answer"
            parts.append(f"{label} (score {a.get('pm_score', 0)}):\n{body}")
    return "\n\n".join(parts) if len(parts) > 1 else None


_SPEAKER = re.compile(r"(?:(?<=\n)|^)([A-Za-z0-9_-]{3,20}): ")
_NOT_SPEAKERS = {"edit", "update", "note", "source", "sources", "answer", "question", "summary", "example",
                 "step", "warning", "disclaimer", "context", "reply", "quote", "http", "https"}


def anonymize_speakers(text: str) -> str:
    """Replace "username: " message prefixes with "User A: ", "User B: ", ... in order of appearance.

    In the threaded Reddit data each message starts on a new line with "username: " (Reddit
    usernames are 3-20 characters of letters, digits, _ and -). Common line labels such as
    "Edit:" are not speakers. Names are also replaced where they recur as a whole word
    (replies often address people by name, sometimes as u/name).
    """
    names: dict[str, str] = {}

    def label(i: int) -> str:
        letters = ""
        i += 1
        while i:
            i, r = divmod(i - 1, 26)
            letters = chr(65 + r) + letters
        return f"User {letters}"

    for m in _SPEAKER.finditer(text):
        name = m.group(1)
        if name.lower() in _NOT_SPEAKERS:
            continue
        if name not in names:
            names[name] = label(len(names))
    if not names:
        return text
    pattern = re.compile(r"(?<![\w/])(?:u/)?(" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
                         + r")(?!\w)")
    return pattern.sub(lambda m: names[m.group(1)], text)


def fmt_reddit(row: dict, args: dict) -> Optional[Doc]:
    """A threaded Reddit conversation with speakers anonymised."""
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    if (row.get("num_messages") or 0) < args.get("min_messages", 2):
        return None
    if (row.get("avg_score") or 0.0) < args.get("min_avg_score", 1.0):
        return None
    if "[deleted]" in text[:200] and args.get("skip_deleted_start", True):
        return None
    return anonymize_speakers(text)


_PG_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\n", re.IGNORECASE)
_PG_END = re.compile(r"\n[^\n]*\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG", re.IGNORECASE)
_QUOTES = re.compile(r"[\"“”]")


def strip_gutenberg_boilerplate(text: str) -> str:
    m = _PG_START.search(text)
    if m:
        text = text[m.end():]
    m = _PG_END.search(text)
    if m:
        text = text[: m.start()]
    return text.strip()


def fmt_gutenberg(row: dict, args: dict) -> Optional[Doc]:
    """A Project Gutenberg book without the licence boilerplate.

    ``min_quote_density`` (quote marks per 1000 characters) keeps dialogue-rich books, which
    is a cheap proxy for fiction; reference works and sermons have few quotes.
    """
    text = row.get("text")
    if not isinstance(text, str):
        return None
    body = strip_gutenberg_boilerplate(text)
    if len(body) < args.get("min_chars", 20_000):
        return None
    density = 1000.0 * len(_QUOTES.findall(body)) / len(body)
    if density < args.get("min_quote_density", 0.0):
        return None
    return body


def fmt_chat(row: dict, args: dict) -> Optional[Doc]:
    """A multi-turn conversation in the SmolLM2 chat template.

    ``<|im_start|>role\\ncontent<|im_end|>\\n`` per message, with the markers as real special tokens.
    """
    messages = row.get("messages") or []
    pieces: list = []
    roles_seen = set()
    for msg in messages:
        role, content = msg.get("role"), msg.get("content")
        if role not in ("system", "user", "assistant") or not isinstance(content, str) or not content.strip():
            continue
        roles_seen.add(role)
        pieces += [Special("im_start"), f"{role}\n{content.strip()}", Special("im_end"), "\n"]
    if not {"user", "assistant"} <= roles_seen:
        return None
    return pieces


FORMATTERS: dict[str, Formatter] = {
    "text": fmt_text,
    "wikipedia": fmt_wikipedia,
    "stackexchange": fmt_stackexchange,
    "reddit": fmt_reddit,
    "gutenberg": fmt_gutenberg,
    "chat": fmt_chat,
}

# Columns each formatter reads (parquet sources load only these).
FORMATTER_COLUMNS: dict[str, list[str]] = {
    "text": ["text"],
    "wikipedia": ["title", "text"],
    "stackexchange": ["question", "answers"],
    "reddit": ["text", "num_messages", "avg_score"],
    "gutenberg": ["text"],
    "chat": ["messages"],
}


def bucket_of(key: Any) -> int:
    """Deterministic bucket in [0, 100) for a document key."""
    return zlib.crc32(str(key).encode("utf-8")) % 100


def make_doc_fn(formatter: str, args: Optional[dict] = None, bucket: Optional[tuple[int, int]] = None,
                key_column: Optional[str] = None) -> Callable[[dict], Optional[Doc]]:
    """Build ``row -> Doc | None`` from plain config values (picklable inputs for worker processes)."""
    if formatter not in FORMATTERS:
        raise ValueError(f"unknown formatter {formatter!r}; available: {sorted(FORMATTERS)}")
    fn, fargs = FORMATTERS[formatter], dict(args or {})
    lo, hi = bucket if bucket is not None else (0, 100)

    def doc_fn(row: dict) -> Optional[Doc]:
        if (lo, hi) != (0, 100):
            key = row.get(key_column) if key_column else None
            if key is None:
                key = row.get("text") if isinstance(row.get("text"), str) else repr(row)
            if not lo <= bucket_of(key) < hi:
                return None
        return fn(row, fargs)

    return doc_fn


def columns_for(formatter: str, key_column: Optional[str] = None, extra: Optional[list[str]] = None) -> list[str]:
    cols = list(FORMATTER_COLUMNS[formatter]) + list(extra or [])
    if key_column and key_column not in cols:
        cols.append(key_column)
    return cols


# ----------------------------------------------------------------------------- the mix

@dataclass
class SourceSpec:
    name: str
    share: float                    # share of this phase's tokens
    dataset: str                    # Hugging Face dataset repo
    files: str                      # glob inside the repo
    formatter: str = "text"
    args: dict = field(default_factory=dict)
    extra_columns: list[str] = field(default_factory=list)   # e.g. int_score for score filters
    exclude: Optional[str] = None   # regex on the file path; matching files are skipped
    file_mod: int = 1
    file_keep: tuple[int, ...] = (0,)
    bucket: tuple[int, int] = (0, 100)
    key_column: Optional[str] = None
    note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["file_keep"] = list(self.file_keep)
        d["bucket"] = list(self.bucket)
        return d


EDU = "HuggingFaceFW/fineweb-edu"
DCLM = "mlfoundations/dclm-baseline-1.0-parquet"
WIKI = "wikimedia/wikipedia"
SE = "HuggingFaceH4/stack-exchange-preferences"
REDDIT = "HuggingFaceGECLM/REDDIT_threaded"
MATH = "HuggingFaceTB/finemath"
PG = "manu/project_gutenberg"
CHAT = "HuggingFaceH4/ultrachat_200k"

# StackOverflow (code, ~70% of the dataset), meta sites and non-English sites are left out.
SE_EXCLUDE = r"/(Stackoverflow\.com|[^/]*meta\.[^/]*|(ru|es|pt|ja)\.stackoverflow\.com)/"
MAIN_FILES = dict(file_mod=10, file_keep=tuple(range(9)))     # 9 of every 10 files
ANNEAL_FILES = dict(file_mod=10, file_keep=(9,))               # the 10th file

MAIN_MIX: list[SourceSpec] = [
    SourceSpec("fineweb_edu", 0.45, EDU, "sample/100BT/*.parquet", "text", {"min_int_score": 3},
               ["int_score"], **MAIN_FILES, note="accuracy: textbook-style web pages"),
    SourceSpec("dclm", 0.25, DCLM, "filtered/**/*.parquet", "text", **MAIN_FILES,
               note="natural, explanatory web text"),
    SourceSpec("wikipedia", 0.07, WIKI, "20231101.en/*.parquet", "wikipedia", bucket=(0, 90), key_column="id",
               note="facts"),
    SourceSpec("stackexchange", 0.065, SE, "data/*/*.parquet", "stackexchange", {"min_answer_score": 1},
               exclude=SE_EXCLUDE, bucket=(0, 90), key_column="question", note="expert Q&A"),
    SourceSpec("reddit", 0.065, REDDIT, "data/*.parquet", "reddit", {"min_messages": 2, "min_avg_score": 1.0},
               bucket=(0, 90), key_column="thread_id", note="casual multi-turn conversation"),
    SourceSpec("finemath", 0.05, MATH, "finemath-3plus/*.parquet", "text", {"max_int_score": 3}, ["int_score"],
               note="step-by-step math (score 3; the 4+ tier is kept for the anneal phase)"),
    SourceSpec("gutenberg", 0.05, PG, "data/en-*.parquet", "gutenberg", {"min_quote_density": 1.0},
               note="books, dialogue-rich (fiction and drama proxy)"),
]

ANNEAL_MIX: list[SourceSpec] = [
    SourceSpec("fineweb_edu_top", 0.30, EDU, "sample/100BT/*.parquet", "text", {"min_int_score": 4},
               ["int_score"], **ANNEAL_FILES, note="top-tier educational pages"),
    SourceSpec("ultrachat", 0.20, CHAT, "data/train_sft-*.parquet", "chat", note="multi-turn chats, chat template"),
    SourceSpec("stackexchange_top", 0.15, SE, "data/*/*.parquet", "stackexchange", {"min_answer_score": 5},
               exclude=SE_EXCLUDE, bucket=(90, 100), key_column="question", note="high-scoring answers"),
    SourceSpec("reddit_top", 0.10, REDDIT, "data/*.parquet", "reddit", {"min_messages": 2, "min_avg_score": 10.0},
               bucket=(90, 100), key_column="thread_id", note="high-scoring threads"),
    SourceSpec("wikipedia_anneal", 0.10, WIKI, "20231101.en/*.parquet", "wikipedia", bucket=(90, 100),
               key_column="id", note="facts (unseen articles)"),
    SourceSpec("finemath_top", 0.10, MATH, "finemath-4plus/*.parquet", "text", note="top-tier math"),
    SourceSpec("dclm_anneal", 0.05, DCLM, "filtered/**/*.parquet", "text", **ANNEAL_FILES,
               note="natural web text (unseen files)"),
]

PHASES: dict[str, list[SourceSpec]] = {"main": MAIN_MIX, "anneal": ANNEAL_MIX}


def check_mix() -> None:
    for phase, specs in PHASES.items():
        total = sum(s.share for s in specs)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"{phase} shares add up to {total}, not 1")
        names = [s.name for s in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"{phase} has duplicate source names")


check_mix()
