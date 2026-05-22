"""
One-time enrichment: run KeyBERT over every URL in combined_metadata.json
and write the extracted keyphrases back as "bert_keywords".

Run with:
    uv run enrich_metadata_keywords.py
    uv run enrich_metadata_keywords.py --force   # re-extract even if already present
"""
import argparse
import json
from pathlib import Path

METADATA_FILE   = Path("combined_metadata.json")
CHECKPOINT_EVERY = 500


def _build_source_text(entry: dict) -> str:
    m = entry.get("metadata", entry) or {}
    parts = []
    if title := m.get("title"):
        parts.append(title)
    if desc := m.get("description"):
        parts.append(desc)
    for field in ("keywords", "categories", "tags"):
        vals = m.get(field)
        if vals and isinstance(vals, list):
            parts.extend(str(v) for v in vals if v)
    if section := m.get("section"):
        parts.append(str(section))
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if bert_keywords already present")
    args = parser.parse_args()

    print("Loading combined_metadata.json...")
    with open(METADATA_FILE) as f:
        data: dict = json.load(f)

    from keybert import KeyBERT
    from utils.embeddings import _get_model
    print("Initializing KeyBERT with shared ST backbone (all-mpnet-base-v2)...")
    kw_model = KeyBERT(model=_get_model())

    urls    = list(data.keys())
    total   = len(urls)
    updated = 0
    skipped = 0

    for i, url in enumerate(urls, 1):
        entry = data[url]

        if not args.force and entry.get("bert_keywords") is not None:
            skipped += 1
            continue

        text = _build_source_text(entry)
        if not text.strip():
            entry["bert_keywords"] = []
            updated += 1
            continue

        kws = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=10,
            use_mmr=True,
            diversity=0.5,
        )
        entry["bert_keywords"] = [kw for kw, _ in kws]
        updated += 1

        if i % CHECKPOINT_EVERY == 0:
            print(f"  {i}/{total} processed — saving checkpoint...")
            with open(METADATA_FILE, "w") as f:
                json.dump(data, f)

    print(f"Saving ({updated} enriched, {skipped} already had keywords)...")
    with open(METADATA_FILE, "w") as f:
        json.dump(data, f)
    print("Done.")


main()
