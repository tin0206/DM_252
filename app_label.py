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
SOURCE_FILE = os.path.join(DATA_DIR, "train.csv")
DEST_FILE = os.path.join(DATA_DIR, "train_tin.csv")

st.set_page_config(
    page_title="Abstract Labeling Tool",
    layout="wide"
)

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_source_data(path):
    if os.path.exists(path):
        return pd.read_csv(path, quoting=csv.QUOTE_ALL)
    return pd.DataFrame()

def load_dest_data(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path, quoting=csv.QUOTE_ALL)
        except:
            return pd.DataFrame()
    return pd.DataFrame()

source_df = load_source_data(SOURCE_FILE)
dest_df = load_dest_data(DEST_FILE)

# =========================================================
# SESSION STATE
# =========================================================
if "index" not in st.session_state:
    st.session_state.index = 0

# =========================================================
# TITLE
# =========================================================
st.title("🛠 Abstract Labeling Tool")

if source_df.empty:
    st.error("Source file not found: train.csv")
    st.stop()

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
temp_key = f"temp_abs_{current_id}"
widget_key = f"widget_abs_{current_id}"

if temp_key not in st.session_state:
    if not saved_row.empty:
        val = saved_row.iloc[0]["abstract"]
    else:
        val = row.get("abstract", "")
    st.session_state[temp_key] = str(val) if pd.notna(val) else ""

# =========================================================
# FLATTEN FUNCTION
# =========================================================
def flatten_logic():
    if widget_key in st.session_state:
        text = st.session_state[widget_key]
        text = text.replace("\n", " ").replace("\r", " ")
        text = re.sub(r"\s+", " ", text).strip()

        st.session_state[widget_key] = text
        st.session_state[temp_key] = text

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Navigation")

    jump_idx = st.number_input(
        "Jump to index:",
        min_value=0,
        max_value=len(source_df) - 1,
        value=st.session_state.index
    )

    if st.button("Go"):
        st.session_state.index = jump_idx
        st.rerun()

    st.divider()

    st.metric("Progress", f"{st.session_state.index + 1} / {len(source_df)}")
    st.write(f"Saved rows: {len(dest_df)}")

# =========================================================
# MAIN UI
# =========================================================
col1, col2 = st.columns([2, 1])

# ---------------- LEFT ----------------
with col1:
    st.subheader(f"Index: {st.session_state.index} | ID: {current_id}")

    st.info(f"Title: {row['title']}")
    st.write(f"Authors: {row.get('authors','')}")
    st.write(f"Venue: {row.get('venue','')}")
    st.write(f"Year: {row.get('year','')}")

    # SEARCH
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

    # ABSTRACT EDITOR
    st.text_area(
        "Abstract",
        value=st.session_state[temp_key],
        key=widget_key,
        height=300
    )

    st.button("Flatten Text", on_click=flatten_logic)

# ---------------- RIGHT ----------------
with col2:
    st.subheader("Save")

    current_label = int(saved_row.iloc[0]["Label"]) if not saved_row.empty and "Label" in saved_row.columns else 0

    label = st.number_input(
        "Label",
        value=current_label,
        step=1,
        key=f"label_{current_id}"
    )

    if st.button("Save & Next", type="primary"):
        final_abs = st.session_state[widget_key]

        new_record = pd.DataFrame([{
            "id": row["id"],
            "title": row["title"],
            "abstract": final_abs,
            "authors": row.get("authors", ""),
            "venue": row.get("venue", ""),
            "year": row.get("year", ""),
            "Label": label
        }])

        current_dest = load_dest_data(DEST_FILE)

        if not current_dest.empty and "id" in current_dest.columns:
            if str(row["id"]) in current_dest["id"].astype(str).values:
                mask = current_dest["id"].astype(str) == str(row["id"])
                current_dest.loc[mask, "abstract"] = final_abs
                current_dest.loc[mask, "Label"] = label
            else:
                current_dest = pd.concat([current_dest, new_record], ignore_index=True)
        else:
            current_dest = new_record

        current_dest.to_csv(
            DEST_FILE,
            index=False,
            quoting=csv.QUOTE_ALL,
            encoding="utf-8-sig"
        )

        st.toast(f"Saved ID: {current_id}")
        st.session_state.index += 1
        st.rerun()

    if st.button("Previous"):
        if st.session_state.index > 0:
            st.session_state.index -= 1
            st.rerun()

# =========================================================
# TABLE VIEW
# =========================================================
st.divider()
st.subheader("Dataset Preview")

final_df = load_dest_data(DEST_FILE)

if not final_df.empty:
    st.dataframe(final_df, use_container_width=True, height=400)
else:
    st.info("No data saved yet.")