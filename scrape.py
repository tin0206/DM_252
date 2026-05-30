import json
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests


DATA_DIR = "data"

SPLITS = {
    "train": {
        "raw": os.path.join(DATA_DIR, "train.csv"),
        "tin": os.path.join(DATA_DIR, "train_tin.csv"),
        "output": os.path.join(DATA_DIR, "train_tin_enriched.csv"),
    },
    "public_test": {
        "raw": os.path.join(DATA_DIR, "public_test.csv"),
        "tin": os.path.join(DATA_DIR, "public_test_tin.csv"),
        "output": os.path.join(DATA_DIR, "public_test_tin_enriched.csv"),
    },
    "private_test": {
        "raw": os.path.join(DATA_DIR, "private_test.csv"),
        "tin": os.path.join(DATA_DIR, "private_test_tin.csv"),
        "output": os.path.join(DATA_DIR, "private_test_tin_enriched.csv"),
    },
}

CACHE_PATH = os.path.join(DATA_DIR, "openalex_metadata_cache.csv")
OPENALEX_API_BASE = "https://api.openalex.org/works/{}"
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "").strip()

REQUEST_SLEEP = 0.2
TIMEOUT = 20
MAX_RETRIES = 3


def normalize_doi(doi: Any) -> str:
    if pd.isna(doi):
        return ""
    doi = str(doi).strip().lower()
    if not doi:
        return ""
    prefixes = (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
    )
    for prefix in prefixes:
        if doi.startswith(prefix):
            doi = doi.replace(prefix, "", 1)
    return doi.strip()


def safe_json_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def unique_join(values: Iterable[str]) -> str:
    seen = set()
    ordered = []
    for value in values:
        value = str(value).strip()
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ", ".join(ordered)


def parse_page_number(value: Any) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"(\d+)", text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def infer_page_count(first_page: Any, last_page: Any, pages: Any) -> str:
    first_num = parse_page_number(first_page)
    last_num = parse_page_number(last_page)

    if first_num is not None and last_num is not None and last_num >= first_num:
        return str(last_num - first_num + 1)

    if isinstance(pages, str) and pages.strip():
        page_text = pages.strip()
        parts = re.split(r"\s*[-–—]\s*", page_text)
        if len(parts) == 2:
            left = parse_page_number(parts[0])
            right = parse_page_number(parts[1])
            if left is not None and right is not None and right >= left:
                return str(right - left + 1)

    return ""


def reconstruct_abstract(abstract_inverted_index: Any) -> str:
    if not isinstance(abstract_inverted_index, dict) or not abstract_inverted_index:
        return ""

    positions: List[Tuple[int, str]] = []
    for token, indices in abstract_inverted_index.items():
        if not isinstance(indices, list):
            continue
        for index in indices:
            try:
                positions.append((int(index), str(token)))
            except Exception:
                continue

    if not positions:
        return ""

    positions.sort(key=lambda item: item[0])
    return " ".join(token for _, token in positions)


def blank_record(doi_norm: str, doi: str) -> Dict[str, Any]:
    return {
        "doi_norm": doi_norm,
        "doi": doi,
        "oa_id": "",
        "oa_title": "",
        "oa_display_name": "",
        "oa_abstract": "",
        "oa_venue": "",
        "oa_source_id": "",
        "oa_source_type": "",
        "oa_source_publisher": "",
        "oa_publication_year": "",
        "oa_publication_date": "",
        "oa_type": "",
        "oa_language": "",
        "oa_cited_by_count": "",
        "oa_referenced_works_count": "",
        "oa_is_oa": "",
        "oa_pdf_url": "",
        "oa_landing_page_url": "",
        "oa_authors": "",
        "oa_author_orcids": "",
        "oa_author_affiliations": "",
        "oa_author_count": "",
        "oa_institution_count": "",
        "oa_concepts": "",
        "oa_concepts_count": "",
        "oa_keywords": "",
        "oa_keywords_count": "",
        "oa_pages": "",
        "oa_page_count": "",
        "oa_biblio_volume": "",
        "oa_biblio_issue": "",
        "oa_biblio_first_page": "",
        "oa_biblio_last_page": "",
    }


def fetch_openalex_by_doi(doi: str) -> Dict[str, Any]:
    if not doi:
        return blank_record("", "")

    doi_norm = normalize_doi(doi)
    url = OPENALEX_API_BASE.format(requests.utils.quote(f"https://doi.org/{doi_norm}", safe=":/"))
    params = {}
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(REQUEST_SLEEP)
            response = requests.get(url, params=params, timeout=TIMEOUT)

            if response.status_code == 200:
                data = response.json()

                primary_location = data.get("primary_location") or {}
                best_oa_location = data.get("best_oa_location") or {}
                location = primary_location if primary_location else best_oa_location
                source = location.get("source") or {}
                biblio = data.get("biblio") or {}

                authorships = data.get("authorships") or []
                author_names = []
                author_orcids = []
                affiliations = []
                institutions = []

                for item in authorships:
                    if not isinstance(item, dict):
                        continue

                    author = item.get("author") or {}
                    name = author.get("display_name") or ""
                    orcid = author.get("orcid") or ""

                    if name:
                        author_names.append(name)
                    if orcid:
                        author_orcids.append(orcid)

                    for inst in item.get("institutions") or []:
                        if not isinstance(inst, dict):
                            continue
                        inst_name = inst.get("display_name") or ""
                        if inst_name:
                            affiliations.append(inst_name)
                            institutions.append(inst_name)

                concepts = data.get("concepts") or []
                keywords = data.get("keywords") or []

                concept_names = [
                    c.get("display_name", "")
                    for c in concepts
                    if isinstance(c, dict) and c.get("display_name")
                ]
                keyword_names = [
                    k.get("display_name", "")
                    for k in keywords
                    if isinstance(k, dict) and k.get("display_name")
                ]

                publication_year = data.get("publication_year", "")
                publication_date = data.get("publication_date", "")
                page_count = infer_page_count(
                    biblio.get("first_page", ""),
                    biblio.get("last_page", ""),
                    biblio.get("pages", ""),
                )

                return {
                    "doi_norm": doi_norm,
                    "doi": doi,
                    "oa_id": data.get("id", "") or "",
                    "oa_title": data.get("title", "") or "",
                    "oa_display_name": data.get("display_name", "") or "",
                    "oa_abstract": reconstruct_abstract(data.get("abstract_inverted_index")),
                    "oa_venue": source.get("display_name", "") or "",
                    "oa_source_id": source.get("id", "") or "",
                    "oa_source_type": source.get("type", "") or "",
                    "oa_source_publisher": source.get("publisher", "") or "",
                    "oa_publication_year": publication_year,
                    "oa_publication_date": publication_date,
                    "oa_type": data.get("type", "") or "",
                    "oa_language": data.get("language", "") or "",
                    "oa_cited_by_count": data.get("cited_by_count", ""),
                    "oa_referenced_works_count": data.get("referenced_works_count", ""),
                    "oa_is_oa": int(bool(data.get("is_oa", False))),
                    "oa_pdf_url": location.get("pdf_url", "") or "",
                    "oa_landing_page_url": location.get("landing_page_url", "") or "",
                    "oa_authors": unique_join(author_names),
                    "oa_author_orcids": unique_join(author_orcids),
                    "oa_author_affiliations": unique_join(affiliations),
                    "oa_author_count": len(author_names),
                    "oa_institution_count": len(set(institutions)),
                    "oa_concepts": unique_join(concept_names),
                    "oa_concepts_count": len(concept_names),
                    "oa_keywords": unique_join(keyword_names),
                    "oa_keywords_count": len(keyword_names),
                    "oa_pages": biblio.get("pages", "") or "",
                    "oa_page_count": page_count,
                    "oa_biblio_volume": biblio.get("volume", "") or "",
                    "oa_biblio_issue": biblio.get("issue", "") or "",
                    "oa_biblio_first_page": biblio.get("first_page", "") or "",
                    "oa_biblio_last_page": biblio.get("last_page", "") or "",
                }

            if response.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue

            last_error = f"HTTP {response.status_code}"
        except Exception as exc:
            last_error = str(exc)
            if attempt < MAX_RETRIES - 1:
                time.sleep(1.0)
                continue

    record = blank_record(doi_norm, doi)
    if last_error:
        record["oa_error"] = last_error
    return record


def load_cache(cache_path: str) -> pd.DataFrame:
    if os.path.exists(cache_path):
        cache_df = pd.read_csv(cache_path)
        if "doi_norm" not in cache_df.columns and "doi" in cache_df.columns:
            cache_df["doi_norm"] = cache_df["doi"].apply(normalize_doi)
        return cache_df
    return pd.DataFrame()


def load_split_frame(tin_path: str, raw_path: str) -> pd.DataFrame:
    tin_df = pd.read_csv(tin_path)
    raw_df = pd.read_csv(raw_path)

    if "id" not in tin_df.columns:
        raise ValueError(f"{tin_path} must contain an id column")
    if "id" not in raw_df.columns or "doi" not in raw_df.columns:
        raise ValueError(f"{raw_path} must contain id and doi columns")

    raw_map = raw_df[["id", "doi"]].copy()
    raw_map["doi_norm"] = raw_map["doi"].apply(normalize_doi)

    if "doi" in tin_df.columns:
        merged = tin_df.merge(raw_map, on="id", how="left", suffixes=("", "_raw"))
        if "doi_raw" in merged.columns:
            merged["doi"] = merged["doi"].where(
                merged["doi"].notna() & (merged["doi"].astype(str).str.strip() != ""),
                merged["doi_raw"],
            )
            merged = merged.drop(columns=["doi_raw"])
    else:
        merged = tin_df.merge(raw_map, on="id", how="left")

    if "doi" not in merged.columns:
        merged["doi"] = ""
    merged["doi"] = merged["doi"].fillna("")
    merged["doi_norm"] = merged["doi"].apply(normalize_doi)

    return merged


def enrich_cache(doi_lookup: pd.DataFrame, cache_path: str) -> pd.DataFrame:
    cache_df = load_cache(cache_path)

    if cache_df.empty:
        cached_dois = set()
    else:
        cached_dois = set(cache_df["doi_norm"].astype(str))

    pending_df = doi_lookup[~doi_lookup["doi_norm"].isin(cached_dois)].copy()
    records = []

    for _, row in pending_df.iterrows():
        doi = row["doi"]
        doi_norm = row["doi_norm"]
        print(f"Fetching DOI: {doi_norm}")
        record = fetch_openalex_by_doi(doi)
        record["doi_norm"] = doi_norm
        record["doi"] = doi
        records.append(record)

    new_cache_df = pd.DataFrame(records)

    if cache_df.empty and new_cache_df.empty:
        merged_cache = pd.DataFrame()
    elif cache_df.empty:
        merged_cache = new_cache_df
    elif new_cache_df.empty:
        merged_cache = cache_df
    else:
        merged_cache = pd.concat([cache_df, new_cache_df], ignore_index=True)

    if not merged_cache.empty and "doi_norm" in merged_cache.columns:
        merged_cache = merged_cache.drop_duplicates(subset=["doi_norm"], keep="last")

    merged_cache.to_csv(cache_path, index=False)
    return merged_cache


def merge_enrichment(split_df: pd.DataFrame, cache_df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    if cache_df.empty:
        enriched = split_df.copy()
        enriched.to_csv(output_path, index=False)
        return enriched

    merged = split_df.merge(cache_df, on="doi_norm", how="left", suffixes=("", "_oa"))

    if "doi_oa" in merged.columns:
        merged["doi"] = merged["doi"].where(
            merged["doi"].notna() & (merged["doi"].astype(str).str.strip() != ""),
            merged["doi_oa"],
        )
        merged = merged.drop(columns=["doi_oa"])

    merged.to_csv(output_path, index=False)
    return merged


def main():
    split_frames = {}
    doi_frames = []

    for split_name, paths in SPLITS.items():
        split_frame = load_split_frame(paths["tin"], paths["raw"])
        split_frames[split_name] = split_frame

        doi_lookup = split_frame[["id", "doi", "doi_norm"]].copy()
        doi_lookup = doi_lookup[doi_lookup["doi_norm"] != ""].drop_duplicates(subset=["doi_norm"])
        doi_frames.append(doi_lookup)

    all_doi_lookup = pd.concat(doi_frames, ignore_index=True)
    all_doi_lookup = all_doi_lookup.drop_duplicates(subset=["doi_norm"]).reset_index(drop=True)

    print(f"Unique DOIs to enrich: {len(all_doi_lookup)}")
    cache_df = enrich_cache(all_doi_lookup, CACHE_PATH)

    for split_name, paths in SPLITS.items():
        output_df = merge_enrichment(split_frames[split_name], cache_df, paths["output"])
        print(f"Saved {split_name} enriched data to: {paths['output']}")
        print(f"Rows: {len(output_df)}")

    print(f"Saved OpenAlex cache to: {CACHE_PATH}")


if __name__ == "__main__":
    main()