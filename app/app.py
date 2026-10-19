"""Apple Health Data Pipeline Streamlit app.

Runs unchanged locally (`streamlit run app/app.py`, DuckDB over Parquet)
and as a Databricks App (HEALTH_BACKEND=databricks, SQL warehouse).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data
import ui

st.set_page_config(page_title="Health Lakehouse", page_icon=":material/monitor_heart:", layout="wide")

PAGES = {
    "Summary": [
        st.Page("views/overview.py", title="Overview", icon=":material/dashboard:", default=True),
    ],
    "Body": [
        st.Page("views/activity.py", title="Activity", icon=":material/directions_walk:"),
        st.Page("views/heart.py", title="Heart", icon=":material/favorite:"),
        st.Page("views/workouts.py", title="Workouts", icon=":material/fitness_center:"),
        st.Page("views/body.py", title="Body & gait", icon=":material/accessibility_new:"),
        st.Page("views/sleep.py", title="Sleep", icon=":material/bedtime:"),
        st.Page("views/ecg.py", title="ECG", icon=":material/ecg_heart:"),
    ],
    "Models": [
        st.Page("views/ml.py", title="ML insights", icon=":material/insights:"),
    ],
}

nav = st.navigation(PAGES)

with st.sidebar:
    try:
        ui.sidebar_period()
    except Exception as exc:  # e.g. no data yet / warehouse unreachable
        st.error(f"Could not load data ({data.backend()} backend): {exc}")
        st.stop()
    st.divider()
    if data.backend() == "databricks":
        source = data.schema_name()
    else:
        # Show the path relative to the project or home dir so screenshots don't leak the user name.
        source = data.data_dir()
        for base, prefix in ((Path(__file__).resolve().parent.parent, ""), (Path.home(), "~/")):
            if source.is_relative_to(base):
                source = prefix + str(source.relative_to(base))
                break
    st.caption(f"Backend: **{data.backend()}**  \nSource: `{source}`")

nav.run()
