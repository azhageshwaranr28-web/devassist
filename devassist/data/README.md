# Data

This submission includes a small hackathon sample downloaded from the Stack Exchange API on 2026-10-05.

- Source: Stack Overflow through `https://api.stackexchange.com/2.3`
- Tags: `python`, `pandas`, `numpy`, `docker`, `tensorflow`
- Selection: top 40 questions per tag by vote; up to two positive-score answers per question
- Raw files: 196 answered questions and 392 answers
- Join key: `answers.question_id = questions.question_id`
- Full details: `raw/manifest.json`

Stack Overflow user contributions are shared under the applicable CC BY-SA license. DevAssist retains question/answer IDs, authors, and source URLs and displays source links with every answer. The code is MIT-licensed; the included Q&A text is not relicensed as MIT.

To refresh the sample:

```bash
python -m ingestion.fetch_stackoverflow --tags python pandas numpy docker tensorflow --per-tag 40 --max-answers 2
python -m ingestion.preprocess
python -m ingestion.build_index
```

Do not commit full StackSample CSV files or generated Qdrant/model artifacts.
