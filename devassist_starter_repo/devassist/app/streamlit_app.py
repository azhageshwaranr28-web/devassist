"""Streamlit interface for DevAssist."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.llm import LLMConfigurationError  # noqa: E402
from pipeline import DevAssistPipeline  # noqa: E402

st.set_page_config(page_title="DevAssist", page_icon="🧭", layout="wide")

st.markdown(
    """
<style>
.block-container {max-width: 1100px; padding-top: 2rem;}
.hero {padding: 1.2rem 1.4rem; border-radius: 16px; background: linear-gradient(120deg,#0b1f33,#143f5f); color: white; margin-bottom: 1rem;}
.hero h1 {margin: 0; font-size: 2.2rem;}
.hero p {margin: .35rem 0 0; color: #d7e9f7;}
.source-card {border-left: 4px solid #ff7a59; padding: .55rem .8rem; margin: .5rem 0; background: #f7fafc; border-radius: 4px;}
</style>
<div class="hero"><h1>DevAssist</h1><p>Grounded developer troubleshooting from Stack Overflow evidence</p></div>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading retrieval and reranking models…")
def load_pipeline() -> DevAssistPipeline:
    return DevAssistPipeline()


with st.sidebar:
    st.subheader("What this demo does")
    st.write("Hybrid retrieval → cross-encoder reranking → grounded LLM answer → metadata-built source links.")
    st.caption("Supported sample tags: Python, Pandas, NumPy, Docker and TensorFlow.")
    st.info("For best results paste the exact exception, command or stack-trace line.")

query = st.text_area(
    "Describe your technical problem",
    height=150,
    placeholder="Example: Why does my Docker container stop immediately after it starts?",
)
search = st.button("Find solution", type="primary", use_container_width=True)

if search:
    if not query.strip():
        st.warning("Enter a problem, error message or code question first.")
        st.stop()
    try:
        pipeline = load_pipeline()
        with st.spinner("Searching, reranking and checking sources…"):
            result = pipeline.answer(query)
    except LLMConfigurationError as exc:
        st.error(str(exc))
        st.code("cp .env.example .env\n# Edit .env, then restart Streamlit", language="bash")
        st.stop()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.code("python -m ingestion.preprocess\npython -m ingestion.build_index", language="bash")
        st.stop()
    except Exception as exc:
        st.error(f"DevAssist could not complete the request: {exc}")
        st.stop()

    if result["status"] == "insufficient":
        st.warning(result["answer"])
    else:
        st.subheader("Solution")
        st.markdown(result["answer"])
        st.subheader("Verified sources")
        for source in result["sources"]:
            st.markdown(
                f"<div class='source-card'><b>{source['title']}</b><br>"
                f"Question <a href='{source['question_url']}' target='_blank'>#{source['question_id']}</a> · "
                f"Answer <a href='{source['answer_url']}' target='_blank'>#{source['answer_id']}</a></div>",
                unsafe_allow_html=True,
            )

    with st.expander("Evidence and pipeline trace"):
        st.json(result["timings_ms"])
        for index, context in enumerate(result["contexts"], start=1):
            meta = context["metadata"]
            st.markdown(
                f"**[{index}] {meta['title']}**  \n"
                f"Q#{meta['question_id']} · A#{meta['answer_id']} · reranker {context['reranker_score']:.3f}"
            )
            st.code(context["text"][:1600], language=None)
