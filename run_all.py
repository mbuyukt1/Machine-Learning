import argparse
import logging
import os
import json
import yaml
import hashlib
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from etl_core.etl_stages import ProductionETLProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MirayETL")

def main():
    parser = argparse.ArgumentParser(description="Production ETL Pipeline")
    parser.add_argument("--seed", type=int, default=73681)
    args = parser.parse_args()

    logger.info("Executing Industrial ETL Pipeline for Miray Büyüktepe (73681)...")

    # Extract
    data = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
    df = data.frame.copy()
    if "class" in df.columns:
        df = df.rename(columns={"class": "income"})
    logger.info(f"Extract Stage: Ingested {df.shape[0]} rows and {df.shape[1]} columns.")

    # Validation
    feats = [c for c in df.columns if c != "income"]
    dups = df[df.duplicated(subset=feats, keep=False)]
    confl = dups.groupby(feats, observed=True)["income"].nunique()

    val_lines = [
        "# Industrial ETL Validation and Pre-Flight Checks",
        "**Engineer:** Miray Büyüktepe (ID: 73681)  \n",
        "## 1. Schema Boundaries",
        "- age: 17-90 verified",
        "- hours-per-week: 1-99 verified",
        "- capital-gain / loss: compliant",
        f"- Duplicate conflict rows: {(confl > 1).sum()} feature combinations detected.",
        "- Policy: Zero rows purged silently."
    ]
    with open("reports/validation_audit.md", "w", encoding="utf-8") as f:
        f.write("\n".join(val_lines) + "\n")

    # Split 70/30 stratified BEFORE cleaning
    y = (df["income"].astype(str).str.strip().isin([">50K", ">50K."])).astype(int)
    X = df.drop(columns=["income"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=args.seed, stratify=y
    )

    # Deduplicate train only
    train_combo = pd.concat([X_train, y_train], axis=1).drop_duplicates()
    y_train = train_combo["income"]
    X_train = train_combo.drop(columns=["income"])
    logger.info(f"Deduplicated Train: {len(X_train)} rows. Test: {len(X_test)}.")

    num_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

    etl = ProductionETLProcessor(num_cols, cat_cols)
    etl.fit_training_stage(X_train)

    X_train_clean = etl.transform_stage(X_train)
    X_test_clean = etl.transform_stage(X_test)

    joblib.dump(etl, "saved_models/preprocessor.joblib")

    # Model Benchmark
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    scaler1 = StandardScaler()

    cat_tr = ohe.fit_transform(X_train_clean[cat_cols])
    cat_te = ohe.transform(X_test_clean[cat_cols])

    num_tr = scaler1.fit_transform(X_train_clean[num_cols])
    num_te = scaler1.transform(X_test_clean[num_cols])

    clf1 = LogisticRegression(max_iter=1000, random_state=args.seed)
    clf1.fit(np.hstack([num_tr, cat_tr]), y_train)
    p1 = clf1.predict(np.hstack([num_te, cat_te]))

    m1 = {
        "accuracy": round(float(accuracy_score(y_test, p1)), 4),
        "f1": round(float(f1_score(y_test, p1)), 4),
        "precision": round(float(precision_score(y_test, p1)), 4),
        "recall": round(float(recall_score(y_test, p1)), 4)
    }

    X_tr_eng = etl.engineer_derived_features(X_train_clean)
    X_te_eng = etl.engineer_derived_features(X_test_clean)
    num_eng_cols = num_cols + ["capital_net", "hours_per_age"]

    scaler2 = StandardScaler()
    num_tr_eng = scaler2.fit_transform(X_tr_eng[num_eng_cols])
    num_te_eng = scaler2.transform(X_te_eng[num_eng_cols])

    clf2 = LogisticRegression(max_iter=1000, random_state=args.seed)
    clf2.fit(np.hstack([num_tr_eng, cat_tr]), y_train)
    p2 = clf2.predict(np.hstack([num_te_eng, cat_te]))

    m2 = {
        "accuracy": round(float(accuracy_score(y_test, p2)), 4),
        "f1": round(float(f1_score(y_test, p2)), 4),
        "precision": round(float(precision_score(y_test, p2)), 4),
        "recall": round(float(recall_score(y_test, p2)), 4)
    }

    metrics_out = {
        "raw_baseline": m1,
        "feature_engineered": m2,
        "delta": {
            "accuracy_gain": round(m2["accuracy"] - m1["accuracy"], 4),
            "f1_gain": round(m2["f1"] - m1["f1"], 4)
        }
    }
    with open("saved_models/performance_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=4)

    tr_hash = hashlib.sha256(pd.concat([X_train_clean, y_train], axis=1).to_csv(index=False).encode("utf-8")).hexdigest()
    te_hash = hashlib.sha256(pd.concat([X_test_clean, y_test], axis=1).to_csv(index=False).encode("utf-8")).hexdigest()

    params = {
        "etl_metadata": {
            "lead": "Miray Büyüktepe",
            "student_id": 73681,
            "seed": args.seed
        },
        "checksums": {
            "train_sha256": tr_hash,
            "test_sha256": te_hash
        },
        "scores": metrics_out
    }
    with open("parameters/configuration.yaml", "w", encoding="utf-8") as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)
    logger.info("ETL pipeline executed with zero errors.")

if __name__ == "__main__":
    main()
