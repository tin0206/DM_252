import streamlit as st
import pandas as pd
import webbrowser
import csv
import os
import re

# =========================================================
# CONFIG
# =========================================================
DATA_DIR = "data"

DATASETS = [
    {
        "name": "public_test",
        "input": os.path.join(DATA_DIR, "public_test.csv"),
        "output": os.path.join(DATA_DIR, "public_test_tin.csv"),
    },
    {
        "name": "private_test",
        "input": os.path.join(DATA_DIR, "private_test.csv"),
        "output": os.path.join(DATA_DIR, "private_test_tin.csv"),
    }
]

st.set_page_config(
    page_title="Test Abstract Collection Tool",
    layout="wide"
)

# =========================================================
# SESSION STATE
# =========================================================
if "dataset_idx" not in st.session_state:
    st.session_state.dataset_idx = 0

if "index" not in st.session_state:
    st.session_state.index = 0

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_source(path):
    if os.path.exists(path):
        return pd.read_csv(path, quoting=csv.QUOTE_ALL)
    return pd.DataFrame()

def load_dest(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path, quoting=csv.QUOTE_ALL)
        except:
            return pd.DataFrame()
    return pd.DataFrame()

# =========================================================
# CURRENT DATASET
# =========================================================
current = DATASETS[st.session_state.dataset_idx]

SOURCE_FILE = current["input"]
DEST_FILE = current["output"]

source_df = load_source(SOURCE_FILE)
dest_df = load_dest(DEST_FILE)

# =========================================================
# TITLE
# =========================================================
st.title("🛠 Abstract Collection Tool (Test Set)")

st.info(f"Dataset: **{current['name']}**")

if source_df.empty:
    st.error(f"Missing file: {SOURCE_FILE}")
    st.stop()

# =========================================================
# AUTO SWITCH DATASET
# =========================================================
if st.session_state.index >= len(source_df):

    if st.session_state.dataset_idx < len(DATASETS) - 1:

        st.success(f"Completed: {current['name']}")

        if st.button("➡️ Switch to next dataset"):
            st.session_state.dataset_idx += 1
            st.session_state.index = 0
            st.rerun()

        st.stop()

    else:
        st.success("🎉 All datasets completed!")
        st.stop()

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Navigation")

    jump = st.number_input(
        "Jump to index",
        min_value=0,
        max_value=len(source_df) - 1,
        value=st.session_state.index
    )

    if st.button("Go"):
        st.session_state.index = jump
        st.rerun()

    st.divider()

    st.metric("Progress", f"{st.session_state.index + 1} / {len(source_df)}")

    st.write(f"Saved rows: {len(dest_df)}")

# =========================================================
# CURRENT ROW
# =========================================================
row = source_df.iloc[st.session_state.index]
current_id = str(row["id"])

saved_row = pd.DataFrame()
if not dest_df.empty and "id" in dest_df.columns:
    saved_row = dest_df[dest_df["id"].astype(str) == current_id]

# =========================================================
# STATE KEY
# =========================================================
temp_key = f"temp_{current_dataset['name']}_{current_id}"
widget_key = f"widget_{current_dataset['name']}_{current_id}"

if temp_key not in st.session_state:
    if not saved_row.empty:
        val = saved_row.iloc[0].get("abstract", "")
    else:
        val = row.get("abstract", "")
    st.session_state[temp_key] = str(val) if pd.notna(val) else ""

# =========================================================
# FLATTEN
# =========================================================
def flatten():
    if widget_key in st.session_state:
        text = st.session_state[widget_key]
        text = text.replace("\n", " ").replace("\r", " ")
        text = re.sub(r"\s+", " ", text).strip()

        st.session_state[widget_key] = text
        st.session_state[temp_key] = text

# =========================================================
# UI
# =========================================================
col1, col2 = st.columns([2, 1])

# LEFT
with col1:
    st.subheader(f"{current['name']} | Index {st.session_state.index} | ID {current_id}")

    st.info(f"Title: {row.get('title','')}")

    st.write(f"Authors: {row.get('authors','')}")
    st.write(f"Venue: {row.get('venue','')}")
    st.write(f"Year: {row.get('year','')}")

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Search Title"):
            webbrowser.open_new_tab(
                f"https://www.google.com/search?q={row['title']}"
            )

    with c2:
        doi_val = str(row.get("doi", ""))
        if st.button("Search DOI", disabled=("10." not in doi_val)):
            webbrowser.open_new_tab(
                f"https://www.google.com/search?q={doi_val}"
            )

    st.text_area(
        "Abstract",
        value=st.session_state[temp_key],
        key=widget_key,
        height=300
    )

    st.button("Flatten Text", on_click=flatten)

# RIGHT
with col2:
    st.subheader("Save")

    if st.button("💾 Save & Next", type="primary"):
        final_abs = st.session_state[widget_key]

        new_row = pd.DataFrame([{
            "id": row["id"],
            "title": row.get("title", ""),
            "abstract": final_abs,
            "authors": row.get("authors", ""),
            "venue": row.get("venue", ""),
            "year": row.get("year", "")
        }])

        current_dest = load_dest(DEST_FILE)

        if not current_dest.empty and "id" in current_dest.columns:
            if str(row["id"]) in current_dest["id"].astype(str).values:
                mask = current_dest["id"].astype(str) == str(row["id"])
                current_dest.loc[mask, "abstract"] = final_abs
            else:
                current_dest = pd.concat([current_dest, new_row], ignore_index=True)
        else:
            current_dest = new_row

        current_dest.to_csv(
            DEST_FILE,
            index=False,
            quoting=csv.QUOTE_ALL,
            encoding="utf-8-sig"
        )

        st.toast(f"Saved ID {current_id}")

        st.session_state.index += 1
        st.rerun()

    if st.button("⬅️ Previous"):
        if st.session_state.index > 0:
            st.session_state.index -= 1
            st.rerun()

# =========================================================
# TABLE VIEW
# =========================================================
st.divider()
st.subheader(f"Preview: {DEST_FILE}")

df_out = load_dest(DEST_FILE)

if not df_out.empty:
    st.dataframe(df_out, use_container_width=True, height=400)
else:
    st.info("No saved data yet.")