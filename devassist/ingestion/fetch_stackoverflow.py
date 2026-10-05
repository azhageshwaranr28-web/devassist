"""Download a small, recent Stack Overflow Q&A dataset through Stack Exchange API.

This is intentionally small enough for a same-day hackathon. Questions and answers
are saved separately so preprocessing must join answers.question_id to questions.id.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import requests

API = "https://api.stackexchange.com/2.3"
DEFAULT_TAGS = ["python", "pandas", "numpy", "docker", "tensorflow"]


def api_get(path: str, params: dict[str, Any], attempts: int = 4) -> dict[str, Any]:
    for attempt in range(attempts):
        response = requests.get(f"{API}/{path}", params=params, timeout=45)
        if response.status_code == 400:
            raise RuntimeError(f"Stack Exchange API rejected the request: {response.text[:500]}")
        if response.status_code in {429, 502, 503, 504}:
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()
        payload = response.json()
        if payload.get("backoff"):
            time.sleep(int(payload["backoff"]) + 1)
        return payload
    raise RuntimeError(f"Stack Exchange API failed after {attempts} attempts: {path}")


def chunks(values: list[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def fetch_questions(tags: list[str], per_tag: int) -> list[dict[str, Any]]:
    found: dict[int, dict[str, Any]] = {}
    for tag in tags:
        page = 1
        while len([q for q in found.values() if tag in q.get("tags", [])]) < per_tag:
            page_size = min(100, per_tag)
            payload = api_get(
                "questions",
                {
                    "site": "stackoverflow",
                    "tagged": tag,
                    "sort": "votes",
                    "order": "desc",
                    "filter": "withbody",
                    "pagesize": page_size,
                    "page": page,
                },
            )
            for item in payload.get("items", []):
                found[int(item["question_id"])] = item
            if not payload.get("has_more") or page * page_size >= per_tag:
                break
            page += 1
    return list(found.values())


def fetch_answers(question_ids: list[int], max_answers: int) -> list[dict[str, Any]]:
    by_question: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for batch in chunks(question_ids, 80):
        page = 1
        while page <= 5:
            path = f"questions/{';'.join(map(str, batch))}/answers"
            payload = api_get(
                path,
                {
                    "site": "stackoverflow",
                    "sort": "votes",
                    "order": "desc",
                    "filter": "withbody",
                    "pagesize": 100,
                    "page": page,
                },
            )
            for item in payload.get("items", []):
                qid = int(item["question_id"])
                if int(item.get("score", 0)) > 0:
                    by_question[qid].append(item)
            if not payload.get("has_more"):
                break
            page += 1

    selected: list[dict[str, Any]] = []
    for qid, items in by_question.items():
        items.sort(key=lambda x: (bool(x.get("is_accepted")), int(x.get("score", 0))), reverse=True)
        selected.extend(items[:max_answers])
    return selected


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    parser.add_argument("--per-tag", type=int, default=40)
    parser.add_argument("--max-answers", type=int, default=2)
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()

    questions = fetch_questions(args.tags, args.per_tag)
    answers = fetch_answers([int(q["question_id"]) for q in questions], args.max_answers)
    answered = {int(a["question_id"]) for a in answers}
    questions = [q for q in questions if int(q["question_id"]) in answered]

    output = Path(args.output_dir)
    write_jsonl(output / "questions.jsonl", questions)
    write_jsonl(output / "answers.jsonl", answers)
    manifest = {
        "source": "Stack Exchange API / Stack Overflow",
        "endpoint": API,
        "tags": args.tags,
        "questions": len(questions),
        "answers": len(answers),
        "selection": f"top {args.per_tag} by votes per tag; positive-score answers; max {args.max_answers} answers/question",
        "license": "Stack Overflow user contributions are CC BY-SA; retain source links and attribution.",
        "downloaded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
