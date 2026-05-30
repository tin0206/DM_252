import json
import os
import re
import time
from typing import Any, Dict, Iterable, List

import pandas as pd
import requests


DATA_DIR = "data"

INPUT_SPLITS = {
    "train": os.path.join(DATA_DIR, "train_tin_enriched.csv"),
    "public_test": os.path.join(DATA_DIR, "public_test_tin_enriched.csv"),
    "private_test": os.path.join(DATA_DIR, "private_test_tin_enriched.csv"),
}

OUTPUT_SPLITS = {
    "train": os.path.join(DATA_DIR, "train_tin_s2_enriched.csv"),
    "public_test": os.path.join(DATA_DIR, "public_test_tin_s2_enriched.csv"),
    "private_test": os.path.join(DATA_DIR, "private_test_tin_s2_enriched.csv"),
}

CACHE_PATH = os.path.join(DATA_DIR, "semantic_scholar_cache.csv")
S2_API_BASE = "https://api.semanticscholar.org/graph/v1/paper/{}"

S2_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
MIN_REQUEST_INTERVAL_SECONDS = 1.0
PASS_COOLDOWN_SECONDS = 10
MAX_PASSES = 5
TIMEOUT = 5

S2_FIELDS = ",".join(
    [
        "title",
        "abstract",
        "venue",
        "year",
        "url",
        "authors",
        "citationCount",
        "influentialCitationCount",
        "referenceCount",
        "fieldsOfStudy",
        "publicationTypes",
        "isOpenAccess",
        "openAccessPdf",
        "externalIds",
        "publicationVenue",
    ]
)

DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)


class RateLimiter:
    def __init__(self, min_interval_seconds: float = 1.0):
        self.min_interval_seconds = float(min_interval_seconds)
        self.last_request_start = None

    def wait(self) -> None:
        now = time.monotonic()
        if self.last_request_start is not None:
            elapsed = now - self.last_request_start
            remaining = self.min_interval_seconds - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self.last_request_start = time.monotonic()


def safe_str(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "nat", "none"}:
        return ""
    return text


def normalize_doi(value: Any) -> str:
    text = safe_str(value).lower()
    if not text:
        return ""

    prefixes = (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
    )
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text.replace(prefix, "", 1)

    match = DOI_RE.search(text)
    if match:
        return match.group(0).lower().strip()

    return text.strip()


def extract_doi_from_text(value: Any) -> str:
    text = safe_str(value)
    if not text:
        return ""

    normalized = normalize_doi(text)
    if DOI_RE.fullmatch(normalized):
        return normalized

    match = DOI_RE.search(text)
    if match:
        return match.group(0).lower().strip()

    return ""


def extract_semantic_scholar_paper_id(value: Any) -> str:
    text = safe_str(value)
    if not text:
        return ""

    if "semanticscholar.org/paper/" not in text.lower():
        return ""

    match = re.search(r"semanticscholar\.org/paper/(?:[^/?#]+/)?([^/?#]+)", text, re.I)
    if match:
        return match.group(1).strip()

    return ""


def unique_join(values: Iterable[str]) -> str:
    seen = set()
    ordered = []
    for value in values:
        item = safe_str(value)
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ", ".join(ordered)


def blank_s2_record() -> Dict[str, str]:
    return {
        "ss_id": "",
        "ss_url": "",
        "ss_title": "",
        "ss_abstract": "",
        "ss_venue": "",
        "ss_year": "",
        "ss_authors": "",
        "ss_author_count": "",
        "ss_citation_count": "",
        "ss_influential_citation_count": "",
        "ss_reference_count": "",
        "ss_fields_of_study": "",
        "ss_publication_types": "",
        "ss_is_open_access": "",
        "ss_open_access_pdf_url": "",
        "ss_external_ids": "",
        "ss_error": "",
    }


def normalize_record_strings(record: Dict[str, Any]) -> Dict[str, str]:
    normalized = {}
    for key, value in record.items():
        if isinstance(value, (dict, list)):
            normalized[key] = json.dumps(value, ensure_ascii=False)
        else:
            normalized[key] = safe_str(value)
    return normalized


def build_cache_row(
    result: Dict[str, Any],
    cache_key: str,
    source_id: Any,
    source_doi: Any,
    source_doi_norm: Any,
) -> Dict[str, str]:
    row = dict(result)
    row["cache_key"] = cache_key
    row["source_id"] = safe_str(source_id)
    row["source_doi"] = safe_str(source_doi)
    row["source_doi_norm"] = safe_str(source_doi_norm)
    return normalize_record_strings(row)


def load_cache(cache_path: str) -> pd.DataFrame:
    if os.path.exists(cache_path):
        try:
            cache_df = pd.read_csv(cache_path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=["cache_key"])
        if "cache_key" not in cache_df.columns:
            cache_df["cache_key"] = ""
        return cache_df
    return pd.DataFrame(columns=["cache_key"])


def save_cache(cache_df: pd.DataFrame, cache_path: str) -> None:
    cache_df.to_csv(cache_path, index=False)


def clean_cache(cache_df: pd.DataFrame) -> pd.DataFrame:
    if cache_df.empty:
        return cache_df

    cache_df = cache_df.copy()
    if "ss_error" not in cache_df.columns:
        cache_df["ss_error"] = ""
    if "ss_id" not in cache_df.columns:
        cache_df["ss_id"] = ""
    if "cache_key" not in cache_df.columns:
        cache_df["cache_key"] = ""

    success_mask = (
        cache_df["ss_error"].fillna("").astype(str).str.strip().eq("")
        & cache_df["ss_id"].fillna("").astype(str).str.strip().ne("")
    )
    cleaned = cache_df.loc[success_mask].copy()
    if not cleaned.empty:
        cleaned = cleaned.drop_duplicates(subset=["cache_key"], keep="last")
    return cleaned


def load_frame(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_columns = [
        "id",
        "title",
        "abstract",
        "authors",
        "venue",
        "year",
        "doi",
        "doi_norm",
        "oa_error",
    ]
    for column_name in required_columns:
        if column_name not in df.columns:
            df[column_name] = ""

    for column_name in blank_s2_record().keys():
        if column_name not in df.columns:
            df[column_name] = ""

    for column_name in blank_s2_record().keys():
        df[column_name] = df[column_name].astype("object")

    if "cache_key" not in df.columns:
        df["cache_key"] = ""
    else:
        df["cache_key"] = df["cache_key"].astype("object")

    return df


def build_lookup_reference(row: pd.Series) -> str:
    candidates = [
        row.get("doi", ""),
        row.get("doi_norm", ""),
        row.get("oa_landing_page_url", ""),
        row.get("oa_pdf_url", ""),
    ]

    for candidate in candidates:
        doi = extract_doi_from_text(candidate)
        if doi:
            return f"DOI:{doi}"

        paper_id = extract_semantic_scholar_paper_id(candidate)
        if paper_id:
            return paper_id

    return ""


def is_success_row(result: Dict[str, Any]) -> bool:
    return safe_str(result.get("ss_error", "")) == "" and safe_str(result.get("ss_id", "")) != ""


def apply_cached_row(df: pd.DataFrame, idx: Any, cached_row: Dict[str, Any]) -> None:
    for column_name, value in cached_row.items():
        if column_name in df.columns and column_name.startswith("ss_"):
            df.at[idx, column_name] = safe_str(value)


def apply_result_row(df: pd.DataFrame, idx: Any, result: Dict[str, Any]) -> None:
    for column_name, value in result.items():
        if column_name.startswith("ss_"):
            if column_name not in df.columns:
                df[column_name] = ""
                df[column_name] = df[column_name].astype("object")
            df.at[idx, column_name] = safe_str(value)


def fetch_semantic_scholar_paper(
    ref: str,
    session: requests.Session,
    limiter: RateLimiter,
) -> Dict[str, str]:
    if not ref:
        record = blank_s2_record()
        record["ss_error"] = "EMPTY_REFERENCE"
        return record

    url = S2_API_BASE.format(requests.utils.quote(ref, safe=":/"))
    params = {"fields": S2_FIELDS}
    headers = {}
    if S2_API_KEY:
        headers["x-api-key"] = S2_API_KEY

    limiter.wait()

    try:
        response = session.get(url, params=params, headers=headers, timeout=TIMEOUT)
    except Exception as exc:
        record = blank_s2_record()
        record["ss_error"] = str(exc)
        return record

    if response.status_code == 200:
        data = response.json()

        authors = data.get("authors") or []
        author_names = [author.get("name", "") for author in authors if isinstance(author, dict)]

        fields_of_study = data.get("fieldsOfStudy") or []
        publication_types = data.get("publicationTypes") or []
        open_access_pdf = data.get("openAccessPdf") or {}
        external_ids = data.get("externalIds") or {}
        publication_venue = data.get("publicationVenue") or {}

        record = blank_s2_record()
        record.update(
            {
                "ss_id": data.get("paperId", "") or "",
                "ss_url": data.get("url", "") or "",
                "ss_title": data.get("title", "") or "",
                "ss_abstract": data.get("abstract", "") or "",
                "ss_venue": data.get("venue", "") or publication_venue.get("name", "") or "",
                "ss_year": data.get("year", "") or "",
                "ss_authors": unique_join(author_names),
                "ss_author_count": len(author_names),
                "ss_citation_count": data.get("citationCount", "") or "",
                "ss_influential_citation_count": data.get("influentialCitationCount", "") or "",
                "ss_reference_count": data.get("referenceCount", "") or "",
                "ss_fields_of_study": unique_join(fields_of_study),
                "ss_publication_types": unique_join(publication_types),
                "ss_is_open_access": int(bool(data.get("isOpenAccess", False))),
                "ss_open_access_pdf_url": safe_str(open_access_pdf.get("url", "")),
                "ss_external_ids": external_ids,
                "ss_error": "",
            }
        )
        return normalize_record_strings(record)

    record = blank_s2_record()
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After", "")
        if retry_after.isdigit():
            record["ss_error"] = f"HTTP 429 Retry-After={retry_after}"
        else:
            record["ss_error"] = "HTTP 429"
        return record

    record["ss_error"] = f"HTTP {response.status_code}"
    return record


def load_cache_index(cache_df: pd.DataFrame) -> set:
    if cache_df.empty or "cache_key" not in cache_df.columns:
        return set()

    if "ss_error" in cache_df.columns and "ss_id" in cache_df.columns:
        success_mask = (
            cache_df["ss_error"].fillna("").astype(str).str.strip().eq("")
            & cache_df["ss_id"].fillna("").astype(str).str.strip().ne("")
        )
        return set(cache_df.loc[success_mask, "cache_key"].astype(str))

    return set(cache_df["cache_key"].astype(str))


def append_cache(cache_df: pd.DataFrame, new_rows: List[Dict[str, Any]], cache_path: str) -> pd.DataFrame:
    new_df = pd.DataFrame([normalize_record_strings(row) for row in new_rows])

    if cache_df.empty:
        merged = new_df
    elif new_df.empty:
        merged = cache_df
    else:
        merged = pd.concat([cache_df, new_df], ignore_index=True)

    if not merged.empty:
        merged = merged.drop_duplicates(subset=["cache_key"], keep="last")

    save_cache(merged, cache_path)
    return merged


def get_pending_indices(df: pd.DataFrame) -> List[Any]:
    if "oa_error" not in df.columns:
        return []

    oa_retry_mask = df["oa_error"].astype(str).str.contains("404", na=False)
    ss_id_empty_mask = df["ss_id"].fillna("").astype(str).str.strip().eq("")
    pending_mask = oa_retry_mask & ss_id_empty_mask
    return df.index[pending_mask].tolist()


def ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column_name in blank_s2_record().keys():
        if column_name not in df.columns:
            df[column_name] = ""
        df[column_name] = df[column_name].astype("object")

    if "cache_key" not in df.columns:
        df["cache_key"] = ""
    df["cache_key"] = df["cache_key"].astype("object")
    return df


def process_pass(
    df: pd.DataFrame,
    cache_df: pd.DataFrame,
    cache_keys: set,
    session: requests.Session,
    limiter: RateLimiter,
    pass_number: int,
) -> tuple[pd.DataFrame, pd.DataFrame, set, bool]:
    pending_indices = get_pending_indices(df)
    if not pending_indices:
        return df, cache_df, cache_keys, False

    new_cache_rows = []
    saw_429 = False

    print(f"Pass {pass_number}: {len(pending_indices)} rows pending")

    for idx in pending_indices:
        if safe_str(df.at[idx, "ss_id"]):
            continue

        row = df.loc[idx]
        ref = build_lookup_reference(row)
        cache_key = ref or f"ROW:{safe_str(row.get('id', idx))}"

        if cache_key in cache_keys:
            cached_row = cache_df[cache_df["cache_key"].astype(str) == cache_key]
            if not cached_row.empty:
                apply_cached_row(df, idx, cached_row.iloc[-1].to_dict())
                continue

        print(f"Fetching Semantic Scholar for row id={safe_str(row.get('id'))} using {cache_key}")
        result = fetch_semantic_scholar_paper(ref, session, limiter)
        apply_result_row(df, idx, result)

        if is_success_row(result):
            cache_row = build_cache_row(
                result=result,
                cache_key=cache_key,
                source_id=row.get("id", ""),
                source_doi=row.get("doi", ""),
                source_doi_norm=row.get("doi_norm", ""),
            )
            new_cache_rows.append(cache_row)
            cache_keys.add(cache_key)
        elif safe_str(result.get("ss_error", "")).startswith("HTTP 429"):
            saw_429 = True

    if new_cache_rows:
        cache_df = append_cache(cache_df, new_cache_rows, CACHE_PATH)

    return df, cache_df, cache_keys, saw_429


def process_split(split_name: str, input_path: str, output_path: str, cache_df: pd.DataFrame) -> pd.DataFrame:
    df = load_frame(input_path)
    df = ensure_output_columns(df)

    limiter = RateLimiter(MIN_REQUEST_INTERVAL_SECONDS)
    session = requests.Session()

    try:
        cache_keys = load_cache_index(cache_df)

        for pass_number in range(1, MAX_PASSES + 1):
            pending_before = len(get_pending_indices(df))
            if pending_before == 0:
                break

            df, cache_df, cache_keys, saw_429 = process_pass(
                df=df,
                cache_df=cache_df,
                cache_keys=cache_keys,
                session=session,
                limiter=limiter,
                pass_number=pass_number,
            )

            df.to_csv(output_path, index=False)

            pending_after = len(get_pending_indices(df))
            print(
                f"{split_name} pass {pass_number} done: "
                f"pending_before={pending_before}, pending_after={pending_after}"
            )

            if pending_after == 0:
                break

            if saw_429 and pass_number < MAX_PASSES:
                time.sleep(PASS_COOLDOWN_SECONDS)
    finally:
        session.close()

    df.to_csv(output_path, index=False)
    print(f"Saved {split_name} to: {output_path}")
    print(f"Rows: {len(df)}")
    return df


def main():
    cache_df = load_cache(CACHE_PATH)
    cleaned_cache_df = clean_cache(cache_df)
    if len(cleaned_cache_df) != len(cache_df):
        print(
            f"Cleaned cache: removed {len(cache_df) - len(cleaned_cache_df)} non-success rows"
        )
        save_cache(cleaned_cache_df, CACHE_PATH)
    cache_df = cleaned_cache_df

    for split_name, input_path in INPUT_SPLITS.items():
        output_path = OUTPUT_SPLITS[split_name]
        process_split(split_name, input_path, output_path, cache_df)
        cache_df = clean_cache(load_cache(CACHE_PATH))

    print(f"Saved Semantic Scholar cache to: {CACHE_PATH}")


if __name__ == "__main__":
    main()