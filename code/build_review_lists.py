#!/usr/bin/env python3

from __future__ import annotations

import argparse
import html
import json
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path


QUERIES = [
    "deep learning generalization complexity",
    "model complexity deep learning theory",
    "neural network generalization theory",
    "statistical learning theory neural networks",
    "VC dimension neural networks",
    "VC dimension deep learning",
    "Rademacher complexity neural networks",
    "Rademacher complexity deep learning",
    "PAC-Bayes deep learning",
    "PAC-Bayes neural networks",
    "information-theoretic generalization deep learning",
    "algorithmic stability deep learning",
    "stability generalization stochastic gradient descent",
    "generalization stochastic gradient descent deep learning",
    "loss landscape deep learning generalization",
    "loss landscape sharpness generalization",
    "landscape complexity neural networks",
    "loss function landscape complexity",
    "Hessian spectrum deep learning",
    "Hessian generalization deep learning",
    "Hessian based complexity measure",
    "flat minima deep learning",
    "sharp minima generalization deep learning",
    "implicit regularization deep learning",
    "geometry of optimization deep learning",
    "double descent deep learning",
    "interpolation threshold deep learning",
    "neural tangent kernel feature learning",
    "kernel regime deep learning",
    "feature learning regime neural networks",
    "overparameterization generalization neural networks",
    "scaling laws deep learning",
    "scaling laws transformers",
    "large language models scaling laws",
    "transformers generalization theory",
    "attention generalization theory",
    "diffusion models generalization theory",
    "diffusion models optimization landscape",
    "foundation models generalization theory",
    "smooth convergence loss landscapes",
    "Hessian structure loss landscapes",
    "landscape based complexity measure neural networks",
    "Hessian based model complexity deep learning",
    "optimization landscape geometry neural networks",
    "curvature generalization deep learning",
]

CHECKPOINT_VERSION = 5
REQUEST_HEADERS = {
    "User-Agent": "Python urllib",
}
ARXIV_PHRASES = [
    "statistical learning theory",
    "stochastic gradient descent",
    "information-theoretic",
    "algorithmic stability",
    "rademacher complexity",
    "vc dimension",
    "pac-bayes",
    "loss landscape",
    "loss landscapes",
    "landscape complexity",
    "hessian spectrum",
    "hessian structure",
    "flat minima",
    "sharp minima",
    "implicit regularization",
    "double descent",
    "interpolation threshold",
    "neural tangent kernel",
    "feature learning",
    "kernel regime",
    "scaling laws",
    "large language models",
    "foundation models",
    "deep learning",
    "neural networks",
    "neural network",
    "generalization theory",
    "model complexity",
    "complexity measure",
    "optimization landscape",
    "smooth convergence",
    "curvature generalization",
    "diffusion models",
]
ARXIV_STOPWORDS = {
    "and",
    "or",
    "the",
    "of",
    "for",
    "in",
    "on",
    "to",
    "with",
    "based",
}


@dataclass
class Record:
    local_id: str
    source: str
    query: str
    title: str
    authors: str
    year: str
    venue: str
    doi: str
    url: str
    source_id: str
    abstract: str


def record_to_dict(record: Record) -> dict:
    return asdict(record)


def record_from_dict(data: dict) -> Record:
    return Record(**data)


def fetch_bytes(url: str, params: dict[str, str], timeout: int = 30, retries: int = 5) -> bytes:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(f"{url}?{query}", headers=REQUEST_HEADERS)
    delays = [3, 6, 12, 24, 48]
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code == 429 and attempt < retries:
                delay = delays[min(attempt, len(delays) - 1)]
                print(f"  Retry after HTTP {error.code}: waiting {delay} s")
                time.sleep(delay)
                attempt += 1
                continue
            raise
        except (urllib.error.URLError, TimeoutError, socket.timeout) as error:
            if attempt < retries:
                delay = delays[min(attempt, len(delays) - 1)]
                print(f"  Retry after network error: {error}. Waiting {delay} s")
                time.sleep(delay)
                attempt += 1
                continue
            raise


def fetch_json(url: str, params: dict[str, str], timeout: int = 30) -> dict:
    return json.loads(fetch_bytes(url, params, timeout=timeout).decode("utf-8"))


def fetch_text(url: str, params: dict[str, str], timeout: int = 30) -> str:
    return fetch_bytes(url, params, timeout=timeout).decode("utf-8")


def build_arxiv_search_query(query: str, max_clauses: int = 4) -> str:
    normalized = re.sub(r"\s+", " ", query.strip().lower())
    clauses: list[str] = []
    used_phrases: list[str] = []

    for phrase in sorted(ARXIV_PHRASES, key=len, reverse=True):
        if len(clauses) >= max_clauses:
            break
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized):
            clauses.append(f'all:"{phrase}"')
            used_phrases.append(phrase)

    remainder = normalized
    for phrase in used_phrases:
        remainder = re.sub(rf"(?<!\w){re.escape(phrase)}(?!\w)", " ", remainder)

    for token in re.findall(r"[a-z0-9-]+", remainder):
        if len(clauses) >= max_clauses:
            break
        if token in ARXIV_STOPWORDS:
            continue
        if len(token) < 3 and token not in {"vc"}:
            continue
        clauses.append(f"all:{token}")

    if not clauses:
        return f'all:"{normalized}"'
    return " AND ".join(clauses)


def normalize_title(title: str) -> str:
    title = html.unescape(title).lower()
    title = re.sub(r"[^a-z0-9а-яё]+", " ", title)
    return re.sub(r"\s+", " ", title).strip()


def strip_tags(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def reconstruct_openalex_abstract(inverted_index: dict | None) -> str:
    if not inverted_index:
        return ""
    positions: dict[int, str] = {}
    for word, indexes in inverted_index.items():
        for index in indexes:
            positions[index] = word
    if not positions:
        return ""
    return " ".join(positions[i] for i in range(max(positions) + 1) if i in positions)


def parse_openalex_record(item: dict, query: str, index: int) -> Record:
    authors = []
    for author in item.get("authorships", [])[:8]:
        name = (author.get("author") or {}).get("display_name")
        if name:
            authors.append(name)
    return Record(
        local_id=f"OA-{index:04d}",
        source="OpenAlex",
        query=query,
        title=item.get("display_name") or "",
        authors="; ".join(authors),
        year=str(item.get("publication_year") or ""),
        venue=((item.get("primary_location") or {}).get("source") or {}).get("display_name", ""),
        doi=item.get("doi") or "",
        url=item.get("id") or "",
        source_id=item.get("id") or "",
        abstract=reconstruct_openalex_abstract(item.get("abstract_inverted_index")),
    )


def parse_arxiv_records(xml_text: str, query: str, start_index: int) -> list[Record]:
    root = ET.fromstring(xml_text)
    namespaces = {"a": "http://www.w3.org/2005/Atom"}
    records: list[Record] = []
    for offset, entry in enumerate(root.findall("a:entry", namespaces), start=0):
        authors = [
            author.findtext("a:name", default="", namespaces=namespaces)
            for author in entry.findall("a:author", namespaces)
        ]
        records.append(
            Record(
                local_id=f"ARX-{start_index + offset:04d}",
                source="arXiv",
                query=query,
                title=(entry.findtext("a:title", default="", namespaces=namespaces) or "").strip(),
                authors="; ".join(author for author in authors if author),
                year=(entry.findtext("a:published", default="", namespaces=namespaces) or "")[:4],
                venue="arXiv",
                doi="",
                url=entry.findtext("a:id", default="", namespaces=namespaces) or "",
                source_id=entry.findtext("a:id", default="", namespaces=namespaces) or "",
                abstract=(entry.findtext("a:summary", default="", namespaces=namespaces) or "").strip(),
            )
        )
    return records


def parse_crossref_record(item: dict, query: str, index: int) -> Record:
    authors = []
    for author in item.get("author", [])[:8]:
        given = (author.get("given") or "").strip()
        family = (author.get("family") or "").strip()
        name = " ".join(part for part in [given, family] if part)
        if name:
            authors.append(name)

    issued = item.get("issued", {}).get("date-parts", [])
    year = ""
    if issued and issued[0]:
        year = str(issued[0][0])

    title_list = item.get("title", []) or [""]
    container = item.get("container-title", []) or [""]
    doi = item.get("DOI") or ""
    url = item.get("URL") or (f"https://doi.org/{doi}" if doi else "")

    return Record(
        local_id=f"CR-{index:04d}",
        source="Crossref",
        query=query,
        title=(title_list[0] or "").strip(),
        authors="; ".join(authors),
        year=year,
        venue=(container[0] or "").strip(),
        doi=doi,
        url=url,
        source_id=doi or url,
        abstract=strip_tags(item.get("abstract") or ""),
    )


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {"version": CHECKPOINT_VERSION, "completed_steps": [], "raw_records": [], "config": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if "completed_steps" not in data:
        completed_steps = []
        completed_queries = data.get("completed_queries", [])
        for query in completed_queries:
            completed_steps.append(f"OpenAlex::{query}")
            completed_steps.append(f"arXiv::{query}")
        data["completed_steps"] = completed_steps
    data.setdefault("raw_records", [])
    data.setdefault("config", {})
    data["version"] = CHECKPOINT_VERSION
    return data


def save_checkpoint(path: Path, completed_steps: list[str], records: list[Record], config: dict) -> None:
    payload = {
        "version": CHECKPOINT_VERSION,
        "completed_steps": completed_steps,
        "raw_records": [record_to_dict(record) for record in records],
        "config": config,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_all_records(
    openalex_per_query: int,
    arxiv_per_query: int,
    crossref_per_query: int,
    sleep_seconds: float,
    checkpoint_path: Path,
) -> list[Record]:
    checkpoint = load_checkpoint(checkpoint_path)
    config = {
        "queries": QUERIES,
        "openalex_per_query": openalex_per_query,
        "arxiv_per_query": arxiv_per_query,
        "crossref_per_query": crossref_per_query,
        "arxiv_query_builder": "phrases-v1",
    }
    checkpoint_config = checkpoint.get("config", {})
    if checkpoint_config and checkpoint_config != config and checkpoint["completed_steps"]:
        raise ValueError(
            "Existing checkpoint was created with different query set or per-query limits. "
            "Use --reset or a different output directory."
        )
    completed_steps: list[str] = list(checkpoint["completed_steps"])
    records = [record_from_dict(item) for item in checkpoint["raw_records"]]
    openalex_index = 1 + sum(1 for record in records if record.source == "OpenAlex")
    arxiv_index = 1 + sum(1 for record in records if record.source == "arXiv")
    crossref_index = 1 + sum(1 for record in records if record.source == "Crossref")

    if completed_steps:
        print(f"Resuming from checkpoint: {len(completed_steps)} completed steps")
        print(f"Loaded raw records from checkpoint: {len(records)}")

    for position, query in enumerate(QUERIES, start=1):
        print(f"[{position}/{len(QUERIES)}] Query: {query}")
        openalex_step = f"OpenAlex::{query}"
        if openalex_step in completed_steps:
            print("  OpenAlex: skipped")
        else:
            openalex_response = fetch_json(
                "https://api.openalex.org/works",
                {
                    "search": query,
                    "per-page": str(openalex_per_query),
                },
            )
            openalex_results = openalex_response.get("results", [])
            print(f"  OpenAlex: {len(openalex_results)} records")
            for item in openalex_results:
                records.append(parse_openalex_record(item, query, openalex_index))
                openalex_index += 1
            completed_steps.append(openalex_step)
            save_checkpoint(checkpoint_path, completed_steps, records, config)
            print(f"  Checkpoint saved: {checkpoint_path}")
            time.sleep(sleep_seconds)

        arxiv_step = f"arXiv::{query}"
        if arxiv_step in completed_steps:
            print("  arXiv:    skipped")
        else:
            arxiv_search_query = build_arxiv_search_query(query)
            arxiv_xml = fetch_text(
                "https://export.arxiv.org/api/query",
                {
                    "search_query": arxiv_search_query,
                    "start": "0",
                    "max_results": str(arxiv_per_query),
                },
            )
            batch = parse_arxiv_records(arxiv_xml, query, arxiv_index)
            print(f"  arXiv:    {len(batch)} records")
            records.extend(batch)
            arxiv_index += len(batch)
            completed_steps.append(arxiv_step)
            save_checkpoint(checkpoint_path, completed_steps, records, config)
            print(f"  Checkpoint saved: {checkpoint_path}")
            time.sleep(sleep_seconds)

        crossref_step = f"Crossref::{query}"
        if crossref_step in completed_steps:
            print("  Crossref: skipped")
        else:
            crossref_response = fetch_json(
                "https://api.crossref.org/works",
                {
                    "query.bibliographic": query,
                    "rows": str(crossref_per_query),
                },
            )
            crossref_results = crossref_response.get("message", {}).get("items", [])
            print(f"  Crossref: {len(crossref_results)} records")
            for item in crossref_results:
                records.append(parse_crossref_record(item, query, crossref_index))
                crossref_index += 1
            completed_steps.append(crossref_step)
            save_checkpoint(checkpoint_path, completed_steps, records, config)
            print(f"  Checkpoint saved: {checkpoint_path}")
            time.sleep(sleep_seconds)

        print(f"  Raw total so far: {len(records)}")

    return records


def deduplicate_records(records: list[Record]) -> list[dict]:
    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        key = record.doi.lower().strip() if record.doi else normalize_title(record.title)
        grouped[key].append(record)

    deduplicated: list[dict] = []
    for index, (key, group) in enumerate(grouped.items(), start=1):
        first = group[0]
        deduplicated.append(
            {
                "dedup_id": f"D-{index:04d}",
                "dedup_key": key,
                "title": first.title,
                "authors": first.authors,
                "year": first.year,
                "venue": first.venue,
                "doi": first.doi,
                "url": first.url,
                "sources": sorted({item.source for item in group}),
                "queries": sorted({item.query for item in group}),
                "source_hits": len(group),
                "abstract": first.abstract,
            }
        )
    return deduplicated


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build search lists for the review.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "generated"),
        help="Directory for generated files.",
    )
    parser.add_argument("--openalex-per-query", type=int, default=100, help="OpenAlex results per query.")
    parser.add_argument("--arxiv-per-query", type=int, default=100, help="arXiv results per query.")
    parser.add_argument("--crossref-per-query", type=int, default=100, help="Crossref results per query.")
    parser.add_argument("--sleep-seconds", type=float, default=3.0, help="Delay between requests.")
    parser.add_argument("--reset", action="store_true", help="Remove saved checkpoint before running.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.json"

    if args.reset and checkpoint_path.exists():
        checkpoint_path.unlink()

    print("Starting search list build")
    print(f"Output directory: {output_dir}")
    print(f"Checkpoint file: {checkpoint_path}")
    print(f"OpenAlex per query: {args.openalex_per_query}")
    print(f"arXiv per query: {args.arxiv_per_query}")
    print(f"Crossref per query: {args.crossref_per_query}")
    print(f"Delay between requests: {args.sleep_seconds} s")
    print(f"Queries: {len(QUERIES)}")

    raw_records = fetch_all_records(
        openalex_per_query=args.openalex_per_query,
        arxiv_per_query=args.arxiv_per_query,
        crossref_per_query=args.crossref_per_query,
        sleep_seconds=args.sleep_seconds,
        checkpoint_path=checkpoint_path,
    )
    deduplicated_records = deduplicate_records(raw_records)

    write_json(output_dir / "raw_records.json", [asdict(record) for record in raw_records])
    write_json(output_dir / "deduplicated_records.json", deduplicated_records)

    print("Finished")
    print(f"Raw records: {len(raw_records)}")
    print(f"Deduplicated records: {len(deduplicated_records)}")
    print("Generated files:")
    print(f"  - {output_dir / 'raw_records.json'}")
    print(f"  - {output_dir / 'deduplicated_records.json'}")
    print(f"  - {checkpoint_path}")


if __name__ == "__main__":
    main()
