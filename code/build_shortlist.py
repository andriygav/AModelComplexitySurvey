#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


THEMES: dict[str, dict[str, object]] = {
    "classical": {
        "title_patterns": [
            (r"\bvc\b|\bvc dimension\b|\bvapnik\b|\blearnable\b|\bpac\b", 7),
            (r"\brademacher\b|\bgaussian complexities\b", 7),
            (r"\bstatistical learning theory\b", 7),
        ],
        "abstract_patterns": [
            (r"\bvc\b|\bpac\b|\brademacher\b|\buniform convergence\b", 3),
            (r"\bhypothesis class\b|\bcapacity\b", 2),
        ],
    },
    "probabilistic_information": {
        "title_patterns": [
            (r"\bpac bayes\b|\bpac-bayes\b", 8),
            (r"\binformation theoretic\b|\binformation-theoretic\b|\binformation bottleneck\b", 7),
            (r"\bstability\b|\bgeneralization bounds\b", 6),
        ],
        "abstract_patterns": [
            (r"\bpac bayes\b|\bpac-bayes\b|\bposterior\b|\bprior\b", 3),
            (r"\binformation theoretic\b|\binformation-theoretic\b|\bmutual information\b", 3),
            (r"\bstability\b|\bgeneralization bound", 2),
        ],
    },
    "landscape": {
        "title_patterns": [
            (r"\bloss landscape\b|\blandscape\b", 8),
            (r"\bhessian\b|\bcurvature\b|\bflat minima\b|\bsharp minima\b|\bsharpness\b", 8),
            (r"\bimplicit regularization\b|\bgeometry of optimization\b", 7),
        ],
        "abstract_patterns": [
            (r"\bhessian\b|\bcurvature\b|\bflat minima\b|\bsharp minima\b", 3),
            (r"\blandscape\b|\bgeometry\b|\bimplicit regularization\b", 2),
        ],
    },
    "regimes": {
        "title_patterns": [
            (r"\bdouble descent\b|\binterpolation\b", 8),
            (r"\bneural tangent kernel\b|\bntk\b", 8),
            (r"\bkernel regime\b|\bfeature learning\b|\boverparameterized\b", 7),
        ],
        "abstract_patterns": [
            (r"\bdouble descent\b|\binterpolation\b", 3),
            (r"\bneural tangent kernel\b|\bkernel regime\b|\bfeature learning\b", 3),
            (r"\boverparameterized\b", 2),
        ],
    },
    "scaling_architectures": {
        "title_patterns": [
            (r"\bscaling law\b|\bscaling laws\b", 8),
            (r"\btransformer\b|\battention is all you need\b", 7),
            (r"\bdiffusion\b|\blanguage model\b|\bfoundation model\b", 7),
        ],
        "abstract_patterns": [
            (r"\bscaling law\b|\bscaling laws\b", 3),
            (r"\btransformer\b|\bdiffusion\b|\blanguage model\b", 2),
        ],
    },
    "surveys_synthesis": {
        "title_patterns": [
            (r"\bsurvey\b|\breview\b|\boverview\b|\bperspective\b|\blecture notes\b", 6),
            (r"\bunderstanding\b|\bexplaining\b|\brecent advances\b", 5),
            (r"\bpredicting generalization\b|\bcomplexity measure\b", 5),
        ],
        "abstract_patterns": [
            (r"\bsurvey\b|\breview\b|\boverview\b", 2),
            (r"\btheoretical\b|\btheory\b|\bsynthesis\b", 2),
        ],
    },
}

GENERAL_TITLE_PATTERNS: list[tuple[str, int]] = [
    (r"\bgeneralization\b", 6),
    (r"\bcomplexity\b|\bcomplexities\b", 5),
    (r"\btheory\b|\btheoretical\b|\bbounds?\b|\banalysis\b", 4),
    (r"\bdeep learning\b|\bneural network", 2),
]

GENERAL_ABSTRACT_PATTERNS: list[tuple[str, int]] = [
    (r"\bgeneralization\b", 3),
    (r"\bcomplexity\b|\bcomplexities\b", 2),
    (r"\btheory\b|\btheoretical\b|\bbounds?\b|\banalysis\b", 2),
]

APPLICATION_PATTERNS = [
    r"\bcancer\b|\bmedical\b|\bultrasound\b|\bmri\b|\bglaucoma\b|\bbreast\b|\bskin lesion\b",
    r"\bwind power\b|\bsolar\b|\bhvac\b|\bsmart home\b|\bmariculture\b|\bwater quality\b",
    r"\bmalware\b|\bsurveillance\b|\bfruit\b|\bagriculture\b|\blithology\b|\bconcrete\b",
    r"\bchannel estimation\b|\bmimo\b|\bofdm\b|\bwireless\b|\bmetasurface\b|\bbattery\b",
    r"\bchemistry\b|\bdrug discovery\b|\bprotein\b|\bcryo\b|\btomography\b|\bunderwater\b",
    r"\bdisease\b|\bdiagnosis\b|\bforecasting\b|\bfault diagnosis\b|\barrhythmia\b",
]

STRONG_THEORY_PATTERNS = [
    r"\bgeneralization\b",
    r"\bcomplexity\b",
    r"\bpac bayes\b|\bpac-bayes\b",
    r"\brademacher\b",
    r"\bvc\b|\bvc dimension\b",
    r"\blandscape\b|\bhessian\b|\bdouble descent\b",
    r"\bscaling law\b|\bkernel regime\b|\bfeature learning\b",
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def normalize_title(text: str) -> str:
    return re.sub(r"[^a-z0-9а-яё]+", " ", normalize(text)).strip()


def to_single_line(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def is_excluded_record(record: dict) -> bool:
    authors = normalize(record.get("authors", ""))
    venue = normalize(record.get("venue", ""))
    title = normalize(record.get("title", ""))
    doi = normalize(record.get("doi", ""))

    # Filter out low-confidence anonymous/Zenodo preprints from shortlist candidates.
    if "zenodo" in venue or "zenodo" in doi:
        return True
    if authors in {"anonymous", "revista zen math 10"}:
        return True
    if title.startswith("anonymous") and "preprint" in title:
        return True
    return False


def add_pattern_score(text: str, patterns: list[tuple[str, int]], matched: list[str]) -> int:
    score = 0
    for pattern, weight in patterns:
        if re.search(pattern, text):
            score += weight
            matched.append(pattern)
    return score


def score_record(record: dict) -> dict:
    title = normalize(record.get("title", ""))
    abstract = normalize(record.get("abstract", ""))
    query_count = len(record.get("queries", []))
    source_hits = int(record.get("source_hits", 0))
    year_raw = record.get("year") or "0"
    try:
        year = int(year_raw)
    except ValueError:
        year = 0

    matched: list[str] = []
    score = 0
    score += add_pattern_score(title, GENERAL_TITLE_PATTERNS, matched)
    score += add_pattern_score(abstract, GENERAL_ABSTRACT_PATTERNS, matched)

    theme_scores: dict[str, int] = {}
    for theme, theme_patterns in THEMES.items():
        theme_score = 0
        theme_score += add_pattern_score(title, theme_patterns["title_patterns"], matched)
        theme_score += add_pattern_score(abstract, theme_patterns["abstract_patterns"], matched)
        theme_scores[theme] = theme_score
        score += theme_score

    score += min(query_count, 8)
    score += min(source_hits, 6)

    if year >= 2015:
        score += 1
    if year >= 2020:
        score += 1
    if year >= 2025 and source_hits <= 1:
        score -= 3

    has_strong_theory = any(re.search(pattern, title) or re.search(pattern, abstract) for pattern in STRONG_THEORY_PATTERNS)
    application_hits = sum(1 for pattern in APPLICATION_PATTERNS if re.search(pattern, title) or re.search(pattern, abstract))
    if application_hits:
        penalty = 4 * application_hits
        if has_strong_theory:
            penalty = max(0, penalty - 4)
        score -= penalty

    primary_theme = max(theme_scores, key=theme_scores.get)
    reason_tokens = []
    for theme, value in sorted(theme_scores.items(), key=lambda item: item[1], reverse=True):
        if value > 0:
            reason_tokens.append(f"{theme}:{value}")
    if query_count:
        reason_tokens.append(f"queries:{query_count}")
    if source_hits:
        reason_tokens.append(f"hits:{source_hits}")
    if application_hits:
        reason_tokens.append(f"application_penalty:{application_hits}")

    return {
        "record_id": record["dedup_id"],
        "title": record["title"],
        "abstract": to_single_line(record.get("abstract", "")),
        "year": record.get("year", ""),
        "score": score,
        "primary_theme": primary_theme,
        "reason": ", ".join(reason_tokens[:6]),
        "source_hits": source_hits,
        "title_key": normalize_title(record["title"]),
    }


def build_shortlist(scored_records: list[dict], target_size: int, min_score: int, per_theme_seed: int, per_theme_cap: int) -> list[dict]:
    unique_by_title: dict[str, dict] = {}
    for record in scored_records:
        if record["score"] < min_score:
            continue
        existing = unique_by_title.get(record["title_key"])
        if existing is None:
            unique_by_title[record["title_key"]] = record
            continue
        current_rank = (record["score"], record["source_hits"], record["year"], record["record_id"])
        existing_rank = (existing["score"], existing["source_hits"], existing["year"], existing["record_id"])
        if current_rank > existing_rank:
            unique_by_title[record["title_key"]] = record

    eligible = list(unique_by_title.values())
    eligible.sort(key=lambda record: (record["score"], record["year"], record["title"]), reverse=True)

    by_theme: dict[str, list[dict]] = defaultdict(list)
    for record in eligible:
        by_theme[record["primary_theme"]].append(record)

    selected: list[dict] = []
    seen_ids: set[str] = set()
    theme_counts: dict[str, int] = defaultdict(int)

    for theme in THEMES:
        for record in by_theme.get(theme, [])[:per_theme_seed]:
            if record["record_id"] in seen_ids:
                continue
            selected.append(record)
            seen_ids.add(record["record_id"])
            theme_counts[record["primary_theme"]] += 1

    for record in eligible:
        if len(selected) >= target_size:
            break
        if record["record_id"] in seen_ids:
            continue
        if theme_counts[record["primary_theme"]] >= per_theme_cap:
            continue
        selected.append(record)
        seen_ids.add(record["record_id"])
        theme_counts[record["primary_theme"]] += 1

    return selected


def write_review_md(path: Path, records: list[dict]) -> None:
    lines = [
        "| Record ID | Theme | Year | Title | Abstract |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for record in records:
        title = record["title"].replace("|", "\\|")
        abstract = to_single_line(record.get("abstract", "")).replace("|", "\\|")
        lines.append(
            f"| `{record['record_id']}` | `{record['primary_theme']}` | "
            f"{record['year']} | {title} | {abstract} |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_shortlist_json(path: Path, records: list[dict]) -> None:
    json_records = [
        {
            "record_id": record["record_id"],
            "score": record["score"],
            "theme": record["primary_theme"],
            "year": record["year"],
            "title": record["title"],
            "reason": record["reason"],
        }
        for record in records
    ]
    path.write_text(json.dumps(json_records, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a manual rereading shortlist from deduplicated records.")
    parser.add_argument(
        "--input-json",
        default=str(Path(__file__).resolve().parent / "generated" / "deduplicated_records.json"),
        help="Path to deduplicated_records.json.",
    )
    parser.add_argument(
        "--review-md",
        default=str(Path(__file__).resolve().parent / "review.md"),
        help="Markdown list for manual review (next to generated/).",
    )
    parser.add_argument(
        "--output-json",
        default=str(Path(__file__).resolve().parent / "generated" / "shortlist_500.json"),
        help="Path to output JSON shortlist.",
    )
    parser.add_argument("--target-size", type=int, default=500, help="Target shortlist size.")
    parser.add_argument("--min-score", type=int, default=24, help="Minimum score for shortlist candidacy.")
    parser.add_argument("--per-theme-seed", type=int, default=40, help="Guaranteed top records per theme.")
    parser.add_argument("--per-theme-cap", type=int, default=120, help="Maximum records per theme.")
    args = parser.parse_args()

    input_path = Path(args.input_json)
    review_md_path = Path(args.review_md)
    output_json_path = Path(args.output_json)
    records = json.loads(input_path.read_text())
    records = [record for record in records if not is_excluded_record(record)]
    scored_records = [score_record(record) for record in records]
    shortlist = build_shortlist(
        scored_records=scored_records,
        target_size=args.target_size,
        min_score=args.min_score,
        per_theme_seed=args.per_theme_seed,
        per_theme_cap=args.per_theme_cap,
    )
    write_review_md(review_md_path, shortlist)
    write_shortlist_json(output_json_path, shortlist)
    print(f"Review markdown: {review_md_path}")
    print(f"Shortlist json: {output_json_path}")
    print(f"Records in shortlist: {len(shortlist)}")


if __name__ == "__main__":
    main()
