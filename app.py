
import streamlit as st
import requests
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="AI Content Moderation",
    page_icon="🛡️",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000/moderate"

LABEL_NAMES = {
    "toxic": "Toxic",
    "severe_toxic": "Severe Toxic",
    "obscene": "Obscene",
    "threat": "Threat",
    "insult": "Insult",
    "identity_hate": "Identity Hate"
}

# Header
st.title("🛡️ AI-Powered Content Moderation")
st.markdown(
    "### Real-Time Toxicity Detection Using DistilBERT and ONNX Runtime"
)

st.divider()

# Sidebar
st.sidebar.title("About the Project")
st.sidebar.write(
    "This application uses a fine-tuned DistilBERT model "
    "optimized with ONNX Runtime to identify potentially "
    "toxic social media comments."
)

st.sidebar.markdown("**Model:** DistilBERT V3")
st.sidebar.markdown("**Inference:** ONNX Runtime")
st.sidebar.markdown("**Backend:** FastAPI")
st.sidebar.markdown("**Frontend:** Streamlit")

# Main input section
st.subheader("Analyze a Comment")

comment = st.text_area(
    "Enter your comment below:",
    placeholder="Type or paste a social media comment here...",
    height=150
)

analyze = st.button(
    "🔍 Analyze Comment",
    type="primary",
    use_container_width=True
)

if analyze:

    if not comment.strip():
        st.warning("Please enter a comment before analyzing.")

    else:

        try:
            with st.spinner("Analyzing comment..."):

                response = requests.post(
                    API_URL,
                    json={"comment": comment},
                    timeout=30
                )

                response.raise_for_status()
                data = response.json()

            st.divider()

            # Overall result
            st.subheader("Moderation Result")

            decision = data["overall_decision"]

            if decision == "REQUIRES MODERATION":
                st.error("⚠️ REQUIRES MODERATION")
            else:
                st.success("✅ NO AUTOMATIC FLAG")

            st.write("**Analyzed Comment:**")
            st.info(data["comment"])

            # Individual category results
            st.subheader("Toxicity Analysis")

            results = data["results"]

            columns = st.columns(2)

            for index, result in enumerate(results):

                label = result["label"]
                score = result["score"]
                threshold = result["threshold"]
                status = result["decision"]

                with columns[index % 2]:

                    st.markdown(
                        f"**{LABEL_NAMES.get(label, label)}**"
                    )

                    st.progress(min(max(score, 0.0), 1.0))

                    st.write(f"Score: **{score:.4f}**")

                    if threshold is not None:
                        st.caption(
                            f"Decision threshold: {threshold:.2f}"
                        )

                    if status == "FLAGGED":
                        st.error("FLAGGED")

                    elif status == "HUMAN_REVIEW":
                        st.warning("HUMAN REVIEW")

                    else:
                        st.success("NO AUTOMATIC FLAG")

            # Results table
            st.divider()
            st.subheader("Detailed Results")

            table_data = []

            for result in results:
                table_data.append({
                    "Category": LABEL_NAMES.get(
                        result["label"], result["label"]
                    ),
                    "Score": result["score"],
                    "Threshold": result["threshold"],
                    "Decision": result["decision"]
                })

            st.dataframe(
                pd.DataFrame(table_data),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Moderation predictions are model estimates. "
                "Flagged content should be reviewed by a human; "
                "an unflagged comment is not guaranteed to be safe."
            )

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to the FastAPI server. "
                "Please make sure the backend is running."
            )

        except requests.exceptions.RequestException as error:
            st.error(f"API request failed: {error}")

        except Exception as error:
            st.error(f"An unexpected error occurred: {error}")

# Footer
st.divider()

st.caption(
    "AI-Powered Real-Time Content Moderation | "
    "DistilBERT + ONNX Runtime + FastAPI + Streamlit"
)
