from ingestion.preprocess import clean_html


def test_clean_html_keeps_code_and_removes_markup():
    raw = "<p>Use this:</p><pre><code>docker run --rm app</code></pre>"
    result = clean_html(raw)
    assert "<p>" not in result
    assert "docker run --rm app" in result
    assert "```" in result
