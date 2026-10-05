from generation.citations import cited_context_numbers, sources_from_answer


def test_invalid_citation_is_ignored_and_sources_come_from_metadata():
    chunks = [{"metadata": {"question_id": 10, "answer_id": 20, "title": "T", "tags": []}}]
    answer = "Use the documented fix [1], not an invented source [999]."
    assert cited_context_numbers(answer, 1) == [1]
    sources = sources_from_answer(answer, chunks)
    assert sources[0]["question_id"] == 10
    assert sources[0]["answer_id"] == 20
