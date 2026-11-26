from __future__ import annotations

import pytest

from sapia.sources import (ANNEAL_MIX, MAIN_MIX, PHASES, SE_EXCLUDE, Special, anonymize_speakers, bucket_of,
                           columns_for, fmt_chat, fmt_gutenberg, fmt_reddit, fmt_stackexchange, fmt_text,
                           fmt_wikipedia, html_to_text, make_doc_fn, strip_gutenberg_boilerplate)


def test_mix_matches_docs_data_md():
    main = {s.name: s.share for s in MAIN_MIX}
    assert main == {"fineweb_edu": 0.45, "dclm": 0.25, "wikipedia": 0.07, "stackexchange": 0.065,
                    "reddit": 0.065, "finemath": 0.05, "gutenberg": 0.05}
    anneal = {s.name: s.share for s in ANNEAL_MIX}
    assert anneal == {"fineweb_edu_top": 0.30, "ultrachat": 0.20, "stackexchange_top": 0.15, "reddit_top": 0.10,
                      "wikipedia_anneal": 0.10, "finemath_top": 0.10, "dclm_anneal": 0.05}
    for specs in PHASES.values():
        assert sum(s.share for s in specs) == pytest.approx(1.0)


def test_main_and_anneal_never_share_documents():
    by = {s.name: s for s in MAIN_MIX + ANNEAL_MIX}
    for main, anneal in [("fineweb_edu", "fineweb_edu_top"), ("dclm", "dclm_anneal")]:
        a, b = by[main], by[anneal]
        assert a.file_mod == b.file_mod and not set(a.file_keep) & set(b.file_keep)
    for main, anneal in [("wikipedia", "wikipedia_anneal"), ("stackexchange", "stackexchange_top"),
                         ("reddit", "reddit_top")]:
        assert by[main].bucket[1] <= by[anneal].bucket[0] and by[main].key_column == by[anneal].key_column
    assert by["finemath"].args == {"max_int_score": 3}        # the 4+ tier is anneal-only


def test_stackexchange_exclude_pattern():
    import re

    rx = re.compile(SE_EXCLUDE)
    base = "hf://datasets/HuggingFaceH4/stack-exchange-preferences@abc/data/{}/train-00000-of-00001.parquet"
    for site in ("Stackoverflow.com", "3dprinting.meta.stackexchange.com", "meta.mathoverflow.net",
                 "ru.stackoverflow.com", "ja.stackoverflow.com"):
        assert rx.search(base.format(site)), site
    for site in ("math.stackexchange.com", "superuser.com", "mathoverflow.net", "english.stackexchange.com"):
        assert not rx.search(base.format(site)), site


def test_text_score_filters():
    assert fmt_text({"text": "x", "int_score": 3}, {"min_int_score": 4}) is None
    assert fmt_text({"text": "x", "int_score": 4}, {"min_int_score": 4}) == "x"
    assert fmt_text({"text": "x", "int_score": 4}, {"max_int_score": 3}) is None
    assert fmt_text({"text": "  "}, {}) is None and fmt_text({"text": None}, {}) is None


def test_wikipedia_adds_title_and_drops_stubs():
    assert fmt_wikipedia({"title": "Owl", "text": "a" * 600}, {}).startswith("Owl\n\naaa")
    assert fmt_wikipedia({"title": "Stub", "text": "short"}, {}) is None


def test_html_to_text_keeps_structure():
    out = html_to_text("<p>Use <code>ls -la</code> &amp; then:</p><pre><code>a = 1\nb = 2\n</code></pre>"
                       "<ul><li>one</li><li>two</li></ul><p>See <a href='x'>docs</a>.</p>")
    assert "Use `ls -la` & then:" in out
    assert "```\na = 1\nb = 2\n```" in out
    assert "- one\n- two" in out
    assert "See docs." in out and "<" not in out


def test_stackexchange_picks_best_answers():
    row = {"question": "<p>How do owls see at night so well?</p>", "answers": [
        {"text": "<p>low</p>", "pm_score": 0, "selected": False},
        {"text": "<p>Big eyes.</p>", "pm_score": 3, "selected": False},
        {"text": "<p>Rod cells.</p>", "pm_score": 7, "selected": True},
    ]}
    doc = fmt_stackexchange(row, {"min_answer_score": 1})
    assert doc.startswith("Question:\nHow do owls see at night so well?")
    assert doc.index("Accepted answer (score 7):\nRod cells.") < doc.index("Answer (score 3):\nBig eyes.")
    assert "low" not in doc
    assert fmt_stackexchange(row, {"min_answer_score": 10}) is None


def test_reddit_speakers_are_anonymised_and_threads_filtered():
    text = ("alice_99: Why is the sky blue?\n\nbob: Rayleigh scattering, alice_99.\n\n"
            "alice_99: Thanks u/bob!")
    out = anonymize_speakers(text)
    assert out == ("User A: Why is the sky blue?\n\nUser B: Rayleigh scattering, User A.\n\n"
                   "User A: Thanks User B!")
    # Real threads also start replies after a single newline, and use "Edit:" lines.
    single = "carol-x: Is Pluto a planet?\nEdit: spelling\ndave_2: Dwarf planet since 2006, carol-x."
    assert anonymize_speakers(single) == "User A: Is Pluto a planet?\nEdit: spelling\nUser B: Dwarf planet since 2006, User A."
    row = {"text": text, "num_messages": 3, "avg_score": 5.0}
    assert fmt_reddit(row, {"min_avg_score": 1.0}) == out
    assert fmt_reddit(row, {"min_avg_score": 10.0}) is None
    assert fmt_reddit({**row, "num_messages": 1}, {"min_messages": 2}) is None


def test_gutenberg_strips_boilerplate_and_filters_on_dialogue():
    body = ('"Hello," said Ann. "Hi," said Ben. ' * 800)
    raw = ("The Project Gutenberg eBook of X\nlicence text\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\n"
           + body + "\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nmore licence")
    assert strip_gutenberg_boilerplate(raw) == body.strip()
    assert fmt_gutenberg({"text": raw}, {"min_quote_density": 1.0}) == body.strip()
    dry = "*** START OF THE PROJECT GUTENBERG EBOOK Y ***\n" + ("An almanac of tides. " * 2000)
    assert fmt_gutenberg({"text": dry}, {"min_quote_density": 1.0}) is None


def test_chat_uses_special_markers():
    row = {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
    assert fmt_chat(row, {}) == [Special("im_start"), "user\nHi", Special("im_end"), "\n",
                                 Special("im_start"), "assistant\nHello!", Special("im_end"), "\n"]
    assert fmt_chat({"messages": [{"role": "user", "content": "Hi"}]}, {}) is None


def test_buckets_split_documents_deterministically():
    keys = [f"doc-{i}" for i in range(2000)]
    main = make_doc_fn("text", bucket=(0, 90), key_column="id")
    anneal = make_doc_fn("text", bucket=(90, 100), key_column="id")
    kept_main = {k for k in keys if main({"id": k, "text": "t"}) is not None}
    kept_anneal = {k for k in keys if anneal({"id": k, "text": "t"}) is not None}
    assert not kept_main & kept_anneal and kept_main | kept_anneal == set(keys)
    assert 0.05 < len(kept_anneal) / len(keys) < 0.15
    assert bucket_of("doc-1") == bucket_of("doc-1")


def test_columns_include_key_and_extras():
    assert columns_for("reddit", "thread_id") == ["text", "num_messages", "avg_score", "thread_id"]
    assert columns_for("text", None, ["int_score"]) == ["text", "int_score"]
