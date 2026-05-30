import gc
import math
import os
import random
import re
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

DATA_DIR = "data"


def choose_path(*candidate_names):
    for name in candidate_names:
        candidate_path = os.path.join(DATA_DIR, name)
        if os.path.exists(candidate_path):
            return candidate_path
    return os.path.join(DATA_DIR, candidate_names[-1])


TRAIN_PATH = choose_path(
    "train_tin_s2_enriched.csv",
    "train_tin_enriched.csv",
    "train_tin.csv",
)

TEST_PATHS = {
    "public": choose_path(
        "public_test_tin_s2_enriched.csv",
        "public_test_tin.csv",
        "public_test.csv",
    ),
    "private": choose_path(
        "private_test_tin_s2_enriched.csv",
        "private_test_tin.csv",
        "private_test.csv",
    ),
}

MODEL_NAME = "allenai/specter2_base"
SEED = 42
MAX_LENGTH = 512
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 1
EPOCHS = 10
ENCODER_LR = 5e-6
HEAD_LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
DROPOUT = 0.05
PATIENCE = 3
NUM_FOLDS = 5
NUMERIC_HIDDEN = 64
REFERENCE_YEAR = 2026
MAX_POS_WEIGHT = 8.0

MONITOR_WEIGHT_ACCURACY = 0.7
MONITOR_WEIGHT_MACRO_F1 = 0.3

OUTPUT_TEMPLATE = "submission_ordinal_bce.csv"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_series(df, column):
    if column in df.columns:
        return df[column].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index)


def normalize_whitespace(text):
    return " ".join(str(text).split())


def sort_label_value(value):
    text = str(value).strip()
    try:
        return (0, int(float(text)))
    except ValueError:
        return (1, text)


def coalesce_many(df, candidate_columns):
    result = pd.Series([""] * len(df), index=df.index, dtype="object")
    for column_name in candidate_columns:
        if column_name in df.columns:
            candidate = safe_series(df, column_name)
            result = result.where(result.str.strip() != "", candidate)
    return result


def count_items(text):
    cleaned = str(text).strip()
    if not cleaned:
        return 0
    parts = [part.strip() for part in re.split(r"[;,]", cleaned) if part.strip()]
    return len(parts)


def prepare_metadata_columns(df):
    df = df.copy()

    canonical_map = {
        "canonical_title": ["title", "oa_title", "oa_display_name", "ss_title"],
        "canonical_abstract": ["abstract", "oa_abstract", "ss_abstract"],
        "canonical_venue": ["venue", "oa_venue", "ss_venue"],
        "canonical_authors": ["authors", "oa_authors", "ss_authors"],
        "canonical_year": ["year", "oa_publication_year", "ss_year"],
        "canonical_document_type": ["oa_type", "ss_publication_types"],
        "canonical_source_publisher": ["oa_source_publisher"],
        "canonical_fields_of_study": ["ss_fields_of_study"],
    }

    for target_column, candidate_columns in canonical_map.items():
        df[target_column] = coalesce_many(df, candidate_columns).map(normalize_whitespace)

    df["has_openalex_metadata"] = (
        safe_series(df, "oa_title").str.strip().ne("")
        | safe_series(df, "oa_abstract").str.strip().ne("")
        | safe_series(df, "oa_venue").str.strip().ne("")
        | safe_series(df, "oa_authors").str.strip().ne("")
        | safe_series(df, "oa_publication_year").str.strip().ne("")
        | safe_series(df, "oa_type").str.strip().ne("")
        | safe_series(df, "oa_source_publisher").str.strip().ne("")
    ).astype(np.float32)

    df["has_semanticscholar_metadata"] = (
        safe_series(df, "ss_title").str.strip().ne("")
        | safe_series(df, "ss_abstract").str.strip().ne("")
        | safe_series(df, "ss_venue").str.strip().ne("")
        | safe_series(df, "ss_authors").str.strip().ne("")
        | safe_series(df, "ss_year").str.strip().ne("")
        | safe_series(df, "ss_publication_types").str.strip().ne("")
        | safe_series(df, "ss_fields_of_study").str.strip().ne("")
    ).astype(np.float32)

    df["metadata_source_count"] = df["has_openalex_metadata"] + df["has_semanticscholar_metadata"]
    return df


def build_text(df):
    title = safe_series(df, "canonical_title").map(normalize_whitespace)
    abstract = safe_series(df, "canonical_abstract").map(normalize_whitespace)

    has_title = title.str.strip().ne("")
    has_abstract = abstract.str.strip().ne("")

    text = pd.Series([""] * len(df), index=df.index, dtype="object")
    text = text.mask(has_title & has_abstract, title + " [SEP] " + abstract)
    text = text.mask(has_title & ~has_abstract, title)
    text = text.mask(~has_title & has_abstract, abstract)

    return text.map(normalize_whitespace)


def add_numeric_feature(features, df, feature_name, candidates, use_log1p=False):
    numeric_series = None
    for candidate in candidates:
        if candidate in df.columns:
            numeric_series = pd.to_numeric(df[candidate], errors="coerce")
            break

    if numeric_series is None:
        features[feature_name] = 0.0
        features[f"{feature_name}_missing"] = 1.0
        if use_log1p:
            features[f"{feature_name}_log1p"] = 0.0
        return

    features[feature_name] = numeric_series.fillna(0.0)
    features[f"{feature_name}_missing"] = numeric_series.isna().astype(np.float32)
    if use_log1p:
        features[f"{feature_name}_log1p"] = np.log1p(features[feature_name].clip(lower=0))


def build_numeric_features(
    df,
    venue_counts: Optional[pd.Series] = None,
    venue_target_mean: Optional[dict] = None,
    default_target_mean: float = 0.0,
):
    title = safe_series(df, "canonical_title")
    abstract = safe_series(df, "canonical_abstract")
    authors = safe_series(df, "canonical_authors")
    venue = safe_series(df, "canonical_venue")
    year_raw = safe_series(df, "canonical_year")
    document_type = safe_series(df, "canonical_document_type")
    source_publisher = safe_series(df, "canonical_source_publisher")
    fields_of_study = safe_series(df, "canonical_fields_of_study")
    publication_date_raw = safe_series(df, "oa_publication_date")

    title_clean = title.map(normalize_whitespace)
    abstract_clean = abstract.map(normalize_whitespace)
    authors_clean = authors.map(normalize_whitespace)
    venue_clean = venue.map(normalize_whitespace)
    doc_clean = document_type.map(normalize_whitespace)
    publisher_clean = source_publisher.map(normalize_whitespace)
    fos_clean = fields_of_study.map(normalize_whitespace)
    combined_clean = (title_clean + " " + abstract_clean).map(normalize_whitespace)

    year_num = pd.to_numeric(year_raw, errors="coerce")
    year_missing = year_num.isna().astype(np.float32)
    year_num = year_num.fillna(0.0)

    has_year = year_num > 0
    paper_age = np.where(has_year, np.maximum(0, REFERENCE_YEAR - year_num), 0.0)

    publication_date = pd.to_datetime(publication_date_raw, errors="coerce")

    title_word_len = title_clean.str.split().str.len().fillna(0).astype(np.float32)
    abstract_word_len = abstract_clean.str.split().str.len().fillna(0).astype(np.float32)
    authors_word_len = authors_clean.str.split().str.len().fillna(0).astype(np.float32)
    venue_word_len = venue_clean.str.split().str.len().fillna(0).astype(np.float32)
    doc_word_len = doc_clean.str.split().str.len().fillna(0).astype(np.float32)
    publisher_word_len = publisher_clean.str.split().str.len().fillna(0).astype(np.float32)
    fos_word_len = fos_clean.str.split().str.len().fillna(0).astype(np.float32)
    combined_word_len = combined_clean.str.split().str.len().fillna(0).astype(np.float32)

    title_char_len = title_clean.str.len().fillna(0).astype(np.float32)
    abstract_char_len = abstract_clean.str.len().fillna(0).astype(np.float32)
    authors_char_len = authors_clean.str.len().fillna(0).astype(np.float32)
    venue_char_len = venue_clean.str.len().fillna(0).astype(np.float32)
    doc_char_len = doc_clean.str.len().fillna(0).astype(np.float32)
    publisher_char_len = publisher_clean.str.len().fillna(0).astype(np.float32)
    fos_char_len = fos_clean.str.len().fillna(0).astype(np.float32)
    combined_char_len = combined_clean.str.len().fillna(0).astype(np.float32)

    features = pd.DataFrame(index=df.index)

    features["year"] = year_num.astype(np.float32)
    features["year_missing"] = year_missing
    features["paper_age"] = paper_age.astype(np.float32)
    features["publication_year"] = year_num.astype(np.float32)
    features["publication_year_missing"] = year_missing
    features["publication_month"] = publication_date.dt.month.fillna(0).astype(np.float32)
    features["publication_day"] = publication_date.dt.day.fillna(0).astype(np.float32)
    features["publication_date_missing"] = publication_date.isna().astype(np.float32)

    features["title_word_len"] = title_word_len
    features["abstract_word_len"] = abstract_word_len
    features["authors_word_len"] = authors_word_len
    features["venue_word_len"] = venue_word_len
    features["document_type_word_len"] = doc_word_len
    features["source_publisher_word_len"] = publisher_word_len
    features["fields_of_study_word_len"] = fos_word_len
    features["combined_word_len"] = combined_word_len

    features["title_char_len"] = title_char_len
    features["abstract_char_len"] = abstract_char_len
    features["authors_char_len"] = authors_char_len
    features["venue_char_len"] = venue_char_len
    features["document_type_char_len"] = doc_char_len
    features["source_publisher_char_len"] = publisher_char_len
    features["fields_of_study_char_len"] = fos_char_len
    features["combined_char_len"] = combined_char_len

    features["title_char_per_word"] = title_char_len / (title_word_len + 1.0)
    features["abstract_char_per_word"] = abstract_char_len / (abstract_word_len + 1.0)
    features["combined_char_per_word"] = combined_char_len / (combined_word_len + 1.0)

    features["abstract_to_title_word_ratio"] = abstract_word_len / (title_word_len + 1.0)
    features["title_to_abstract_word_ratio"] = title_word_len / (abstract_word_len + 1.0)
    features["abstract_to_combined_word_ratio"] = abstract_word_len / (combined_word_len + 1.0)
    features["title_to_combined_word_ratio"] = title_word_len / (combined_word_len + 1.0)

    features["author_count"] = authors_clean.apply(count_items).astype(np.float32)
    features["fields_of_study_count"] = fos_clean.apply(count_items).astype(np.float32)

    features["has_document_type"] = doc_clean.str.strip().ne("").astype(np.float32)
    features["has_fields_of_study"] = fos_clean.str.strip().ne("").astype(np.float32)

    if venue_counts is not None:
        features["venue_count"] = venue_clean.map(venue_counts).fillna(0.0).astype(np.float32)
    else:
        features["venue_count"] = 0.0

    if venue_target_mean is not None:
        features["venue_target_mean"] = venue_clean.map(venue_target_mean).fillna(default_target_mean).astype(np.float32)
    else:
        features["venue_target_mean"] = default_target_mean

    add_numeric_feature(features, df, "oa_cited_by_count", ["oa_cited_by_count"], use_log1p=True)
    add_numeric_feature(features, df, "oa_referenced_works_count", ["oa_referenced_works_count"], use_log1p=True)
    add_numeric_feature(features, df, "oa_concepts_count", ["oa_concepts_count"], use_log1p=True)
    add_numeric_feature(features, df, "oa_keywords_count", ["oa_keywords_count"], use_log1p=True)
    add_numeric_feature(features, df, "oa_page_count", ["oa_page_count"], use_log1p=True)
    add_numeric_feature(features, df, "oa_author_count", ["oa_author_count"], use_log1p=True)
    add_numeric_feature(features, df, "oa_institution_count", ["oa_institution_count"], use_log1p=True)

    add_numeric_feature(features, df, "ss_author_count", ["ss_author_count"], use_log1p=True)
    add_numeric_feature(features, df, "ss_citation_count", ["ss_citation_count"], use_log1p=True)
    add_numeric_feature(features, df, "ss_influential_citation_count", ["ss_influential_citation_count"], use_log1p=True)
    add_numeric_feature(features, df, "ss_reference_count", ["ss_reference_count"], use_log1p=True)

    features["oa_is_oa"] = pd.to_numeric(
        safe_series(df, "oa_is_oa"),
        errors="coerce",
    ).fillna(0.0).astype(np.float32)

    features["ss_is_open_access"] = pd.to_numeric(
        safe_series(df, "ss_is_open_access"),
        errors="coerce",
    ).fillna(0.0).astype(np.float32)

    features["has_ss_open_access_pdf_url"] = safe_series(df, "ss_open_access_pdf_url").str.strip().ne("").astype(np.float32)

    features["has_openalex_metadata"] = pd.to_numeric(
        safe_series(df, "has_openalex_metadata"),
        errors="coerce",
    ).fillna(0.0).astype(np.float32)

    features["has_semanticscholar_metadata"] = pd.to_numeric(
        safe_series(df, "has_semanticscholar_metadata"),
        errors="coerce",
    ).fillna(0.0).astype(np.float32)

    features["metadata_source_count"] = pd.to_numeric(
        safe_series(df, "metadata_source_count"),
        errors="coerce",
    ).fillna(0.0).astype(np.float32)

    return features


def labels_to_ordinal_targets(labels, num_classes):
    labels = np.asarray(labels, dtype=np.int64)
    thresholds = np.arange(num_classes - 1, dtype=np.int64)
    return (labels[:, None] > thresholds[None, :]).astype(np.float32)


def build_targets(y_index, num_classes):
    return labels_to_ordinal_targets(y_index, num_classes)


def accuracy_from_indices(pred_indices, true_indices):
    true_indices = np.asarray(true_indices).reshape(-1)
    return float((pred_indices == true_indices).mean())


def evaluate_probs(probabilities, y_true):
    pred_indices = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(y_true, pred_indices)
    macro_f1 = f1_score(y_true, pred_indices, average="macro", zero_division=0)
    blended = MONITOR_WEIGHT_ACCURACY * accuracy + MONITOR_WEIGHT_MACRO_F1 * macro_f1
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "blended": blended,
        "pred_indices": pred_indices,
    }


class TextNumericDataset(Dataset):
    def __init__(self, texts, numeric_features, labels, tokenizer, max_length):
        self.texts = list(texts)
        self.numeric_features = np.asarray(numeric_features, dtype=np.float32)
        self.labels = None if labels is None else np.asarray(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
        )
        item = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "numeric_features": self.numeric_features[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


def make_collate_fn(tokenizer):
    def collate_fn(batch):
        labels = None
        numeric_features = torch.tensor(
            [item.pop("numeric_features") for item in batch],
            dtype=torch.float32,
        )

        if "labels" in batch[0]:
            raw_labels = [item.pop("labels") for item in batch]
            labels = torch.tensor(raw_labels, dtype=torch.float32)

        padded = tokenizer.pad(batch, padding=True, return_tensors="pt")
        padded["numeric_features"] = numeric_features
        if labels is not None:
            padded["labels"] = labels
        return padded

    return collate_fn


def build_loader(texts, numeric_features, labels, tokenizer, shuffle):
    dataset = TextNumericDataset(texts, numeric_features, labels, tokenizer, MAX_LENGTH)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=make_collate_fn(tokenizer),
    )


class Specter2NumericBackbone(nn.Module):
    def __init__(self, model_name, numeric_dim, dropout=DROPOUT):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        if hasattr(self.encoder.config, "use_cache"):
            self.encoder.config.use_cache = False

        self.text_hidden_size = self.encoder.config.hidden_size
        self.numeric_branch = nn.Sequential(
            nn.Linear(numeric_dim, NUMERIC_HIDDEN),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(NUMERIC_HIDDEN, NUMERIC_HIDDEN),
            nn.GELU(),
        )
        self.hidden_size = self.text_hidden_size + NUMERIC_HIDDEN

    def encode(self, input_ids, attention_mask, numeric_features):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_repr = outputs.last_hidden_state[:, 0]
        numeric_repr = self.numeric_branch(numeric_features)
        return torch.cat([text_repr, numeric_repr], dim=-1)


class OrdinalBCEModel(Specter2NumericBackbone):
    def __init__(self, model_name, num_classes, numeric_dim, dropout=DROPOUT):
        super().__init__(model_name, numeric_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, num_classes - 1),
        )

    def forward(self, input_ids, attention_mask, numeric_features):
        cls_repr = self.encode(input_ids, attention_mask, numeric_features)
        return self.classifier(cls_repr)


def build_pos_weight(y_train, num_classes, device):
    targets = labels_to_ordinal_targets(y_train, num_classes)
    positives = targets.sum(axis=0)
    negatives = targets.shape[0] - positives

    ratio = negatives / np.clip(positives, 1.0, None)
    ratio = np.sqrt(ratio)
    ratio = np.clip(ratio, 1.0, MAX_POS_WEIGHT)

    return torch.tensor(ratio, dtype=torch.float32, device=device)


def ordinal_logits_to_class_probs(logits):
    survival = torch.sigmoid(logits)

    if survival.size(1) == 0:
        return torch.ones((survival.size(0), 1), device=survival.device, dtype=survival.dtype)

    class_probs = torch.zeros(
        (survival.size(0), survival.size(1) + 1),
        device=survival.device,
        dtype=survival.dtype,
    )
    class_probs[:, 0] = 1.0 - survival[:, 0]
    for class_index in range(1, survival.size(1)):
        class_probs[:, class_index] = survival[:, class_index - 1] - survival[:, class_index]
    class_probs[:, -1] = survival[:, -1]
    class_probs = torch.clamp(class_probs, min=0.0)
    class_probs = class_probs / class_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return class_probs


def collect_class_probs(model, loader, device, num_classes):
    model.eval()
    all_probs = []

    for batch in tqdm(loader, desc="Predicting", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        numeric_features = batch["numeric_features"].to(device)

        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                numeric_features=numeric_features,
            )
            probs = ordinal_logits_to_class_probs(logits)

        all_probs.append(probs.detach().cpu().numpy())

    if not all_probs:
        return np.zeros((0, num_classes), dtype=np.float32)
    return np.concatenate(all_probs, axis=0)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    progress_bar = tqdm(loader, desc="Training", leave=False)

    for step, batch in enumerate(progress_bar, start=1):
        labels = batch["labels"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        numeric_features = batch["numeric_features"].to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            numeric_features=numeric_features,
        )
        loss = criterion(logits, labels) / GRAD_ACCUM_STEPS
        loss.backward()
        total_loss += loss.item() * GRAD_ACCUM_STEPS

        if step % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        progress_bar.set_postfix(loss=total_loss / step)

    if len(loader) % GRAD_ACCUM_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(1, len(loader))


def fit_fold(
    fold_idx,
    train_df,
    y,
    train_indices,
    val_indices,
    test_df,
    device,
    tokenizer,
    num_classes,
):
    train_split_df = prepare_metadata_columns(train_df.iloc[train_indices].reset_index(drop=True))
    val_split_df = prepare_metadata_columns(train_df.iloc[val_indices].reset_index(drop=True))
    test_split_df = test_df.reset_index(drop=True)

    y_train = y[train_indices]
    y_val = y[val_indices]

    train_texts = build_text(train_split_df).tolist()
    val_texts = build_text(val_split_df).tolist()
    test_texts = build_text(test_split_df).tolist()

    venue_series_train = safe_series(train_split_df, "canonical_venue")
    has_any_venue = venue_series_train.str.strip().ne("").any()
    venue_counts = (
        venue_series_train[venue_series_train.str.strip() != ""].value_counts()
        if has_any_venue
        else None
    )

    if has_any_venue:
        venue_target_mean = (
            pd.DataFrame(
                {
                    "venue": venue_series_train,
                    "target": y_train,
                }
            )
            .loc[lambda frame: frame["venue"].str.strip() != ""]
            .groupby("venue")["target"]
            .mean()
            .to_dict()
        )
    else:
        venue_target_mean = None

    default_target_mean = float(y_train.mean()) if len(y_train) else 0.0

    train_numeric_df = build_numeric_features(
        train_split_df,
        venue_counts=venue_counts,
        venue_target_mean=venue_target_mean,
        default_target_mean=default_target_mean,
    )
    val_numeric_df = build_numeric_features(
        val_split_df,
        venue_counts=venue_counts,
        venue_target_mean=venue_target_mean,
        default_target_mean=default_target_mean,
    )
    test_numeric_df = build_numeric_features(
        test_split_df,
        venue_counts=venue_counts,
        venue_target_mean=venue_target_mean,
        default_target_mean=default_target_mean,
    )

    feature_mean = train_numeric_df.mean(axis=0)
    feature_std = train_numeric_df.std(axis=0).replace(0, 1.0).fillna(1.0)

    X_train_num = ((train_numeric_df - feature_mean) / feature_std).fillna(0.0).astype(np.float32).to_numpy()
    X_val_num = ((val_numeric_df - feature_mean) / feature_std).fillna(0.0).astype(np.float32).to_numpy()
    X_test_num = ((test_numeric_df - feature_mean) / feature_std).fillna(0.0).astype(np.float32).to_numpy()

    train_targets = build_targets(y_train, num_classes)
    val_targets = build_targets(y_val, num_classes)

    train_loader = build_loader(train_texts, X_train_num, train_targets, tokenizer, shuffle=True)
    train_eval_loader = build_loader(train_texts, X_train_num, train_targets, tokenizer, shuffle=False)
    val_loader = build_loader(val_texts, X_val_num, val_targets, tokenizer, shuffle=False)
    test_loader = build_loader(test_texts, X_test_num, None, tokenizer, shuffle=False)

    numeric_dim = X_train_num.shape[1]
    model = OrdinalBCEModel(MODEL_NAME, num_classes, numeric_dim, dropout=DROPOUT).to(device)

    pos_weight = build_pos_weight(y_train, num_classes, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": ENCODER_LR},
            {"params": list(model.numeric_branch.parameters()) + list(model.classifier.parameters()), "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    steps_per_epoch = max(1, math.ceil(len(train_loader) / GRAD_ACCUM_STEPS))
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"\n=== Fold {fold_idx + 1} | Ordinal BCE ===")
    print(f"Encoder LR: {ENCODER_LR} | Head LR: {HEAD_LR}")
    print(f"Pos weight: {pos_weight.detach().cpu().numpy().round(3).tolist()}")

    best_state = None
    best_score = -1.0
    patience_left = PATIENCE

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
        )

        train_probs = collect_class_probs(model, train_eval_loader, device, num_classes)
        val_probs = collect_class_probs(model, val_loader, device, num_classes)

        train_metrics = evaluate_probs(train_probs, y_train)
        val_metrics = evaluate_probs(val_probs, y_val)

        print(
            f"Fold {fold_idx + 1} | Epoch {epoch:02d} | "
            f"Train loss={train_loss:.4f} | "
            f"Train acc={train_metrics['accuracy']:.4f} | "
            f"Val acc={val_metrics['accuracy']:.4f} | "
            f"Val macro_f1={val_metrics['macro_f1']:.4f} | "
            f"Val blended={val_metrics['blended']:.4f}"
        )

        if val_metrics["blended"] > best_score:
            best_score = val_metrics["blended"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"Fold {fold_idx + 1}: early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_probs = collect_class_probs(model, val_loader, device, num_classes)
    test_probs = collect_class_probs(model, test_loader, device, num_classes)

    del train_loader, train_eval_loader, val_loader, test_loader
    del model, optimizer, scheduler, criterion
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "val_probs": val_probs,
        "test_probs": test_probs,
    }


def load_combined_test_df():
    frames = []
    for split_name, path in TEST_PATHS.items():
        if not os.path.exists(path):
            print(f"Skipping {split_name}: file not found at {path}")
            continue
        frame = pd.read_csv(path)
        if "id" not in frame.columns:
            frame = frame.copy()
            frame["id"] = np.arange(len(frame))
        frames.append(frame)

    if not frames:
        raise ValueError("No test files were found.")

    return pd.concat(frames, ignore_index=True)


def main():
    set_seed()

    train_df = pd.read_csv(TRAIN_PATH)
    if "Label" not in train_df.columns:
        raise ValueError("Train file must contain a Label column.")

    train_df = prepare_metadata_columns(train_df)
    test_df = prepare_metadata_columns(load_combined_test_df())

    label_values = sorted(
        train_df["Label"].dropna().astype(str).unique().tolist(),
        key=sort_label_value,
    )
    if len(label_values) < 2:
        raise ValueError("Need at least 2 classes.")

    label_to_index = {label: index for index, label in enumerate(label_values)}
    index_to_label = np.asarray(label_values)
    y = train_df["Label"].astype(str).map(label_to_index).to_numpy(dtype=np.int64)
    num_classes = len(label_values)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or tokenizer.cls_token
    tokenizer.padding_side = "right"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Train path: {TRAIN_PATH}")
    print(f"Test rows: {len(test_df)}")
    print(f"Model: {MODEL_NAME}")
    print(f"Classes: {label_values}")

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    oof_probs = np.zeros((len(train_df), num_classes), dtype=np.float32)
    test_probs_sum = np.zeros((len(test_df), num_classes), dtype=np.float32)

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(train_df, y)):
        print(f"\nStarting fold {fold_idx + 1}/{NUM_FOLDS}")

        fold_result = fit_fold(
            fold_idx=fold_idx,
            train_df=train_df,
            y=y,
            train_indices=train_indices,
            val_indices=val_indices,
            test_df=test_df,
            device=device,
            tokenizer=tokenizer,
            num_classes=num_classes,
        )

        oof_probs[val_indices] = fold_result["val_probs"]
        test_probs_sum += fold_result["test_probs"]

    test_probs_avg = test_probs_sum / NUM_FOLDS

    oof_metrics = evaluate_probs(oof_probs, y)
    test_pred = np.argmax(test_probs_avg, axis=1)
    test_labels = index_to_label[test_pred]

    output_path = OUTPUT_TEMPLATE
    pd.DataFrame(
        {
            "id": test_df["id"].to_numpy(),
            "Label": test_labels,
        }
    ).to_csv(output_path, index=False)

    print("\n=== Final Result ===")
    print(f"OOF blended score: {oof_metrics['blended']:.4f}")
    print(f"OOF accuracy: {oof_metrics['accuracy']:.4f}")
    print(f"OOF macro_f1: {oof_metrics['macro_f1']:.4f}")
    print(f"Submission saved to: {output_path}")


if __name__ == "__main__":
    main()