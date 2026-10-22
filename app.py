"""One Streamlit entry point for both saved market-risk case studies."""
import streamlit as st


def main(default_workspace="Equity VaR"):
    st.set_page_config(page_title="Market Risk Lab", page_icon="📉", layout="wide")
    st.markdown("""<style>
    @media (max-width: 1200px) {
      [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) {flex-wrap:wrap;}
      [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="stColumn"] {flex:1 1 40%;min-width:0;}
    }
    </style>""", unsafe_allow_html=True)
    labels = {"equity": "Equity VaR", "bitcoin": "Bitcoin"}
    if "workspace" not in st.session_state:
        st.session_state.workspace = labels.get(st.query_params.get("view"), default_workspace)
    workspace = st.sidebar.radio("Workspace", list(labels.values()), key="workspace")
    slug = next(key for key, value in labels.items() if value == workspace)
    if st.query_params.get("view") != slug:
        st.query_params["view"] = slug
    st.sidebar.divider()
    if workspace == "Bitcoin":
        from views.bitcoin import render
    else:
        from views.equity import render
    render()


if __name__ == "__main__":
    main()
