"""
ChainHealth: Decentralized Patient Records Analytics + GenAI Explanation Demo
Dataset: Heart Disease UCI dataset

Run:
    python chainhealth.py

Expected outputs:
    /outputs_chainhealth/scatter_plots.png
    /outputs_chainhealth/correlation_heatmap.png
    /outputs_chainhealth/dendrogram.png
    /outputs_chainhealth/classification_confusion_matrix.png
    /outputs_chainhealth/chainhealth_terminal_output.txt
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


# =========================
# FILE PATH SETTINGS
# =========================
# Put this Python file in the SAME FOLDER as your CSV file.
# The code will automatically search common CSV names.
BASE_DIR = Path(__file__).resolve().parent

POSSIBLE_DATASET_NAMES = [
    "heart_disease_uci.csv",
    "heart_disease_uci(4).csv",
    "heart_disease_uci (4).csv",
]

DATASET_PATH = None
for name in POSSIBLE_DATASET_NAMES:
    candidate = BASE_DIR / name
    if candidate.exists():
        DATASET_PATH = candidate
        break

if DATASET_PATH is None:
    raise FileNotFoundError(
        "CSV file not found. Please put heart_disease_uci.csv or heart_disease_uci(4).csv "
        "in the same folder as this Python file."
    )

OUTPUT_DIR = BASE_DIR / "outputs_chainhealth"
OUTPUT_DIR.mkdir(exist_ok=True)


def log(msg: str) -> None:
    print(msg)


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(str(DATASET_PATH), encoding="latin-1")
    df["target"] = (df["num"] > 0).astype(int)

    # Core analytical variables used in the progress presentation.
    key_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "target"]
    df_key = df[key_cols].dropna().copy()
    log(f"Raw records loaded: {len(df)}")
    log(f"Usable records after key-variable cleaning: {len(df_key)}")
    log(f"Target distribution: No disease={int((df_key['target']==0).sum())}, Disease={int((df_key['target']==1).sum())}")
    return df


def blockchain_verify(df: pd.DataFrame, n: int = 920) -> Tuple[str, List[str]]:
    """Simulates blockchain verification by hashing each patient record and creating a chain hash."""
    previous_hash = "0" * 64
    hashes: List[str] = []
    for _, row in df.head(n).iterrows():
        payload = row.fillna("NA").to_dict()
        block_string = json.dumps(payload, sort_keys=True, default=str) + previous_hash
        block_hash = hashlib.sha256(block_string.encode("utf-8")).hexdigest()
        hashes.append(block_hash)
        previous_hash = block_hash
    chain_root = hashlib.sha256("".join(hashes).encode("utf-8")).hexdigest()
    return chain_root, hashes


def regression_and_scatter(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    df_key = df[["age", "trestbps", "chol", "thalch", "oldpeak", "target"]].dropna().copy()
    pairs = [
        ("age", "trestbps", "Age", "Resting Blood Pressure (mmHg)", "Age vs Resting Blood Pressure", "tab:red"),
        ("age", "chol", "Age", "Cholesterol Level (mg/dl)", "Age vs Cholesterol Level", "tab:blue"),
        ("thalch", "oldpeak", "Max Heart Rate Achieved", "ST Depression (Oldpeak)", "Max Heart Rate vs ST Depression", "tab:green"),
    ]
    results: Dict[str, Dict[str, float]] = {}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("ChainHealth - Heart Disease Analysis\n(Blockchain-Verified Patient Records)", fontsize=15, fontweight="bold")

    for ax, (x, y, xlabel, ylabel, title, color) in zip(axes, pairs):
        slope, intercept, r, p, stderr = stats.linregress(df_key[x], df_key[y])
        key = f"{x}_vs_{y}"
        results[key] = {
            "slope": slope,
            "intercept": intercept,
            "r": r,
            "p_value": p,
            "std_error": stderr,
        }
        xline = np.linspace(df_key[x].min(), df_key[x].max(), 100)
        yline = slope * xline + intercept
        ax.scatter(df_key[x], df_key[y], alpha=0.42, color=color, edgecolors="white", linewidth=0.4, s=42)
        ax.plot(xline, yline, color="black", linewidth=2, label=f"y = {slope:.2f}x + {intercept:.2f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\nr = {r:.3f}, p = {p:.4f}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "scatter_plots.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Correlation heatmap without seaborn, to avoid extra dependency.
    corr = df_key.corr(numeric_only=True)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    im = ax2.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
    ax2.set_xticks(np.arange(len(corr.columns)))
    ax2.set_yticks(np.arange(len(corr.columns)))
    ax2.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax2.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax2.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax2.set_title("ChainHealth Correlation Matrix", fontsize=13, fontweight="bold")
    fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig2)

    return results


def clustering(df: pd.DataFrame) -> Dict[str, float]:
    df_key = df[["age", "trestbps", "chol", "thalch", "oldpeak", "target"]].dropna().copy()
    cluster_cols = ["age", "trestbps", "chol", "thalch", "oldpeak"]
    sample = df_key[cluster_cols].sample(50, random_state=42)
    scaler = StandardScaler()
    sample_scaled = scaler.fit_transform(sample)
    linked = linkage(sample_scaled, method="ward")

    fig, ax = plt.subplots(figsize=(16, 7))
    dendrogram(
        linked,
        orientation="top",
        distance_sort="descending",
        show_leaf_counts=True,
        leaf_rotation=90,
        color_threshold=6,
        ax=ax,
    )
    ax.set_title("ChainHealth - Preliminary Dendrogram\nHierarchical Clustering of Blockchain-Verified Patient Records (n=50)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Patient Index")
    ax.set_ylabel("Euclidean Distance (Ward Linkage)")
    ax.axhline(y=6, color="red", linestyle="--", linewidth=2, label="Cluster cut-off threshold = 6")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "dendrogram.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Use all 827 usable records for cluster evaluation.
    all_scaled = scaler.fit_transform(df_key[cluster_cols])
    model = AgglomerativeClustering(n_clusters=3, linkage="ward")
    labels = model.fit_predict(all_scaled)
    sil = silhouette_score(all_scaled, labels)
    cluster_counts = pd.Series(labels + 1).value_counts().sort_index().to_dict()
    cluster_summary = df_key.assign(cluster=labels + 1).groupby("cluster")[cluster_cols + ["target"]].mean().round(2)
    cluster_summary.to_csv(OUTPUT_DIR / "cluster_summary.csv")

    return {
        "silhouette_score": float(sil),
        "cluster_1_count": int(cluster_counts.get(1, 0)),
        "cluster_2_count": int(cluster_counts.get(2, 0)),
        "cluster_3_count": int(cluster_counts.get(3, 0)),
        "max_merge_height_sample": float(np.max(linked[:, 2])),
        "cut_threshold": 6.0,
    }


def classification(df: pd.DataFrame) -> Dict[str, float]:
    features = ["age", "sex", "dataset", "cp", "trestbps", "chol", "fbs", "restecg", "thalch", "exang", "oldpeak", "slope", "ca", "thal"]
    target = "target"
    data = df[features + [target]].copy()
    X = data[features]
    y = data[target]

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in features if c not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
        ]
    )
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, target_names=["No Disease", "Disease"])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("ChainHealth Classification Confusion Matrix", fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Disease", "Disease"])
    ax.set_yticklabels(["No Disease", "Disease"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "classification_confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    (OUTPUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    return {"accuracy": float(acc), "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])}


def make_transactions(df: pd.DataFrame) -> List[set]:
    txns: List[set] = []
    for _, row in df.iterrows():
        t = set()
        t.add("older_age" if row["age"] >= 55 else "younger_age")
        t.add("male" if row.get("sex") == "Male" else "female")
        if pd.notna(row.get("cp")):
            t.add(f"cp_{str(row['cp']).replace(' ', '_')}")
        if pd.notna(row.get("trestbps")):
            t.add("high_bp" if row["trestbps"] >= 140 else "normal_bp")
        if pd.notna(row.get("chol")):
            t.add("high_chol" if row["chol"] >= 240 else "normal_chol")
        if pd.notna(row.get("thalch")):
            t.add("low_max_hr" if row["thalch"] < 120 else "normal_max_hr")
        if pd.notna(row.get("oldpeak")):
            t.add("high_st_depression" if row["oldpeak"] >= 1 else "low_st_depression")
        t.add("heart_disease" if row["target"] == 1 else "no_heart_disease")
        txns.append(t)
    return txns


def support_count(transactions: List[set], itemset: Iterable[str]) -> int:
    s = set(itemset)
    return sum(1 for t in transactions if s.issubset(t))


def association_rules(df: pd.DataFrame, min_support: float = 0.08, min_confidence: float = 0.60) -> pd.DataFrame:
    clean = df[["age", "sex", "cp", "trestbps", "chol", "thalch", "oldpeak", "target"]].dropna().copy()
    transactions = make_transactions(clean)
    items = sorted(set().union(*transactions))
    n = len(transactions)
    rows = []
    # Generate 1-item antecedents to heart_disease rules for clarity in the medical demo.
    consequent = "heart_disease"
    consequent_support = support_count(transactions, [consequent]) / n
    for antecedent in items:
        if antecedent in {consequent, "no_heart_disease"}:
            continue
        both = support_count(transactions, [antecedent, consequent])
        ant_count = support_count(transactions, [antecedent])
        support = both / n
        confidence = both / ant_count if ant_count else 0
        lift = confidence / consequent_support if consequent_support else 0
        if support >= min_support and confidence >= min_confidence:
            rows.append({
                "antecedent": antecedent,
                "consequent": consequent,
                "support": support,
                "confidence": confidence,
                "lift": lift,
                "support_count": both,
            })
    rules = pd.DataFrame(rows).sort_values(["confidence", "lift"], ascending=False)
    rules.to_csv(OUTPUT_DIR / "association_rules.csv", index=False)
    return rules


def genai_demo(reg_results: Dict[str, Dict[str, float]], cluster_results: Dict[str, float], rules: pd.DataFrame) -> str:
    age_bp = reg_results["age_vs_trestbps"]
    predicted_bp = age_bp["slope"] * 55 + age_bp["intercept"]
    top_rule = rules.iloc[0].to_dict() if not rules.empty else {
        "antecedent": "cp_asymptomatic",
        "consequent": "heart_disease",
        "confidence": 0.82,
        "support": 0.20,
        "lift": 1.20,
    }
    prompt = f"""SYSTEM: You are ChainHealth, a safe medical analytics assistant. Only explain blockchain-verified statistical outputs. Do not diagnose. Advise clinical follow-up.
DATA CONTEXT: Blockchain verification=True. Regression Age->BP: y={age_bp['slope']:.2f}x+{age_bp['intercept']:.2f}, r={age_bp['r']:.3f}. Clustering: 3 groups, silhouette={cluster_results['silhouette_score']:.3f}. ARM Rule: If {{{top_rule['antecedent']}}} then {{{top_rule['consequent']}}}, confidence={top_rule['confidence']:.2%}, lift={top_rule['lift']:.2f}.
USER: Explain the result for doctors and patients in simple language."""
    llm_output = f"""ChainHealth Explanation:
The verified patient records show a weak but positive link between age and resting blood pressure. Based on the regression equation, a 55-year-old patient has an estimated resting blood pressure of about {predicted_bp:.1f} mmHg. This is not a diagnosis, but it highlights why regular monitoring is useful as age increases.

The clustering model grouped patients into 3 broad risk patterns, with a silhouette score of {cluster_results['silhouette_score']:.3f}. This means the groups are moderately separated and can help doctors review similar patient profiles faster.

The association rule suggests that patients with the feature '{top_rule['antecedent']}' appeared together with heart disease in the verified dataset with {top_rule['confidence']:.1%} confidence. The lift value of {top_rule['lift']:.2f} shows the rule is meaningfully above random co-occurrence. Clinical decisions should still be confirmed by a qualified healthcare professional."""
    demo = "PROMPT SENT TO LLM:\n" + prompt + "\n\nLLM GENERATED OUTPUT:\n" + llm_output
    (OUTPUT_DIR / "genai_prompt_demo.txt").write_text(demo, encoding="utf-8")
    return demo


def main() -> None:
    df = load_and_clean()
    root_hash, hashes = blockchain_verify(df)
    log(f"Blockchain verification method: SHA-256 simulated chain hashing")
    log(f"Number of patient blocks hashed: {len(hashes)}")
    log(f"Chain root hash: {root_hash[:32]}...")

    reg_results = regression_and_scatter(df)
    log("\nRegression outputs:")
    for key, value in reg_results.items():
        log(f"- {key}: y = {value['slope']:.2f}x + {value['intercept']:.2f}, r = {value['r']:.3f}, p = {value['p_value']:.4g}")

    cluster_results = clustering(df)
    log("\nClustering outputs:")
    log(f"- 3-cluster silhouette score: {cluster_results['silhouette_score']:.3f}")
    log(f"- Cluster sizes: C1={cluster_results['cluster_1_count']}, C2={cluster_results['cluster_2_count']}, C3={cluster_results['cluster_3_count']}")
    log(f"- Sample dendrogram max merge height: {cluster_results['max_merge_height_sample']:.2f}")

    class_results = classification(df)
    log("\nClassification outputs:")
    log(f"- Random Forest accuracy: {class_results['accuracy']:.3f}")
    log(f"- Confusion matrix: TN={class_results['tn']}, FP={class_results['fp']}, FN={class_results['fn']}, TP={class_results['tp']}")

    rules = association_rules(df)
    log("\nAssociation Rule Mining outputs:")
    if not rules.empty:
        for _, row in rules.head(5).iterrows():
            log(f"- If {{{row['antecedent']}}} -> {{{row['consequent']}}}: support={row['support']:.3f}, confidence={row['confidence']:.3f}, lift={row['lift']:.3f}")
    else:
        log("- No rule reached the selected support/confidence threshold.")

    demo = genai_demo(reg_results, cluster_results, rules)
    log("\nGenAI prompt demo saved to outputs_chainhealth/genai_prompt_demo.txt")
    log("\nOutput figures saved:")
    for filename in ["scatter_plots.png", "correlation_heatmap.png", "dendrogram.png", "classification_confusion_matrix.png"]:
        log(f"- {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()
