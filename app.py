import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from hf_model_tagger import parse_model_id, fetch_model_data, fetch_config, apply_tags, TAG_DEFINITIONS

st.set_page_config(page_title="HF Model Tagger", page_icon="🏷️")
st.title("🏷️ HuggingFace Model Tagger")
st.caption("Paste a HuggingFace model URL and get automatic tags — no login required.")

url = st.text_input("Model URL or ID", placeholder="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2")

if st.button("Tag it", type="primary"):
    if not url.strip():
        st.warning("Please enter a model URL or ID.")
    else:
        try:
            repo_id = parse_model_id(url.strip())
            with st.spinner(f"Fetching metadata for {repo_id} ..."):
                model_data = fetch_model_data(repo_id)
                config = fetch_config(repo_id)
                if config:
                    model_data["config"] = {**(model_data.get("config") or {}), **config}
                tags = apply_tags(model_data)
            st.success(f"{repo_id} — {len(tags)} tag(s) found")
            for tag in tags:
                desc = TAG_DEFINITIONS[tag][0] if tag in TAG_DEFINITIONS else ""
                st.markdown(f"- **{tag}** — {desc}")
        except Exception as e:
            st.error(f"Error: {e}")

st.divider()
st.caption("Built with the HuggingFace public API")
