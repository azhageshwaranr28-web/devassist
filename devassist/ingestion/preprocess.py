"""Clean Stack Overflow HTML and join questions to their best answers."""
from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from config import settings


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def clean_html(raw_html: str) -> str:
    """Convert HTML to readable text while preserving code and pre blocks."""
    soup = BeautifulSoup(html.unescape(raw_html or ""), "html.parser")
    for pre in soup.find_all("pre"):
        code = pre.get_text("\n", strip=False).strip("\n")
        pre.replace_with(f"\n```\n{code}\n```\n")
    for code in soup.find_all("code"):
        code.replace_with(f"`{code.get_text(' ', strip=True)}`")
    text = soup.get_text("\n")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def owner_name(item: dict[str, Any]) -> str:
    return str(item.get("owner", {}).get("display_name", "unknown"))


def build_documents(
    questions: list[dict[str, Any]], answers: list[dict[str, Any]], max_answers: int = 2
) -> list[dict[str, Any]]:
    question_map = {int(q["question_id"]): q for q in questions}
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for answer in answers:
        if int(answer.get("score", 0)) > 0:
            grouped[int(answer["question_id"])].append(answer)

    documents: list[dict[str, Any]] = []
    for qid, answer_rows in grouped.items():
        question = question_map.get(qid)
        if not question:
            continue
        answer_rows.sort(
            key=lambda row: (bool(row.get("is_accepted")), int(row.get("score", 0))), reverse=True
        )
        for answer in answer_rows[:max_answers]:
            title = html.unescape(str(question.get("title", "Untitled")))
            qbody = clean_html(str(question.get("body", "")))
            abody = clean_html(str(answer.get("body", "")))
            aid = int(answer["answer_id"])
            content = f"Question:\n{qbody}\n\nAnswer:\n{abody}".strip()
            documents.append(
                {
                    "document_id": f"q{qid}-a{aid}",
                    "title": title,
                    "content": content,
                    "metadata": {
                        "question_id": qid,
                        "answer_id": aid,
                        "tags": question.get("tags", []),
                        "question_score": int(question.get("score", 0)),
                        "answer_score": int(answer.get("score", 0)),
                        "is_accepted": bool(answer.get("is_accepted", False)),
                        "question_author": owner_name(question),
                        "answer_author": owner_name(answer),
                        "question_url": question.get("link", f"https://stackoverflow.com/questions/{qid}"),
                        "answer_url": f"https://stackoverflow.com/a/{aid}",
                        "source": "Stack Overflow",
                    },
                }
            )
    return documents


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=settings.raw_questions)
    parser.add_argument("--answers", type=Path, default=settings.raw_answers)
    parser.add_argument("--output", type=Path, default=settings.documents_file)
    parser.add_argument("--max-answers", type=int, default=2)
    args = parser.parse_args()

    documents = build_documents(read_jsonl(args.questions), read_jsonl(args.answers), args.max_answers)
    write_jsonl(args.output, documents)
    counts = Counter(tag for d in documents for tag in d["metadata"]["tags"])
    print(f"Wrote {len(documents)} clean question-answer documents to {args.output}")
    print("Most common tags:", counts.most_common(12))


if __name__ == "__main__":
    main()
