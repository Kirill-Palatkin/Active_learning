import os
import random
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


DATA_DIR = "data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")

RESULTS_CSV = "results.csv"
RESULTS_PLOT = "results.png"

RANDOM_SEED = 42
N_CLASSES = 4

TRAIN_TEST_SPLIT = 0.1  # 10% на тест
FRACTIONS = [0.01, 0.10, 0.20]  # 1%, 10%, 20%
N_RANDOM_RUNS = 5

MAX_FEATURES = 20000  # TF-IDF признаков
NGRAM_RANGE = (1, 2)

EPOCHS = 5
BATCH_SIZE = 64
LR = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_agnews_from_kaggle(train_csv_path: str) -> pd.DataFrame:
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(
            f"Нет датасета {train_csv_path}."
        )

    df = pd.read_csv(train_csv_path, header=None)
    first_val = df.iloc[0, 0]
    if not isinstance(first_val, (int, float, np.integer, np.floating)):
        df = pd.read_csv(train_csv_path)
        cols = df.columns.tolist()
        if len(cols) < 3:
            raise ValueError(
                f"Ошибка"
            )
        label_col = cols[0]
        title_col = cols[1]
        desc_col = cols[2]
        df["label"] = df[label_col].astype(int) - 1  # 1..4 -> 0..3
        df["text"] = df[title_col].fillna("") + " " + df[desc_col].fillna("")
    else:
        if df.shape[1] < 3:
            raise ValueError(
                f"Ошибка"
            )
        df.columns = ["label", "title", "description"] + [
            f"extra_{i}" for i in range(df.shape[1] - 3)
        ]
        df["label"] = df["label"].astype(int) - 1  # 1..4 -> 0..3
        df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")

    df = df[["text", "label"]].dropna()
    df = df.reset_index(drop=True)
    return df


class TfidfDataset(Dataset):
    def __init__(self, X_csr, y: np.ndarray):
        self.X = X_csr
        self.y = y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x_dense = self.X[idx].toarray().astype(np.float32).squeeze(0)
        y_val = self.y[idx]
        return torch.from_numpy(x_dense), torch.tensor(y_val, dtype=torch.long)


class MLPTextClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_model_torch(
        X_train,
        y_train: np.ndarray,
        X_eval,
        y_eval: np.ndarray,
        num_classes: int,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        lr: float = LR,
        seed: int = RANDOM_SEED,
        verbose: bool = False,
) -> Tuple[MLPTextClassifier, float]:
    set_seed(seed)

    input_dim = X_train.shape[1]
    model = MLPTextClassifier(input_dim, num_classes).to(DEVICE)

    train_ds = TfidfDataset(X_train, y_train)
    eval_ds = TfidfDataset(X_eval, y_eval)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        if verbose:
            avg_loss = total_loss / len(train_ds)
            print(f"[Epoch {epoch}/{epochs}] loss = {avg_loss:.4f}")

    f1 = evaluate_f1(model, eval_loader)
    return model, f1


def evaluate_f1(model: MLPTextClassifier, eval_loader: DataLoader) -> float:
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in eval_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = yb.numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return f1


def predict_proba_on_pool(
        model: MLPTextClassifier,
        X_pool,
        batch_size: int = 256
) -> np.ndarray:
    
    dummy_labels = np.zeros(X_pool.shape[0], dtype=np.int64)
    pool_ds = TfidfDataset(X_pool, dummy_labels)
    pool_loader = DataLoader(pool_ds, batch_size=batch_size, shuffle=False)

    model.eval()
    all_probs = []
    with torch.no_grad():
        for xb, _ in pool_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    return np.vstack(all_probs)


def run_random_baselines(
        X_train_all,
        y_train_all,
        X_test,
        y_test
) -> Dict[str, Dict[str, float]]:
    results = {}

    n_train = X_train_all.shape[0]
    indices_all = np.arange(n_train)

    for frac in FRACTIONS:
        n_samples = max(int(frac * n_train), N_CLASSES * 10)
        f1_scores = []

        print(
            f"\n\033[33m[Random]\033[0m \033[1mДоля обучающих данных:\033[0m {frac * 100:.1f}% ({n_samples} образцов)")

        for run in range(N_RANDOM_RUNS):
            set_seed(RANDOM_SEED + run)
            chosen_idx = np.random.choice(indices_all, size=n_samples, replace=False)

            X_sub = X_train_all[chosen_idx]
            y_sub = y_train_all[chosen_idx]

            _, f1 = train_model_torch(
                X_sub,
                y_sub,
                X_test,
                y_test,
                num_classes=N_CLASSES,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                lr=LR,
                seed=RANDOM_SEED + run,
                verbose=False,
            )
            f1_scores.append(f1)
            print(f"  - Запуск {run + 1}/{N_RANDOM_RUNS}: F1_macro = {f1:.4f}")

        mean_f1 = float(np.mean(f1_scores))
        std_f1 = float(np.std(f1_scores))
        results[f"random_{frac}"] = {"mean_f1": mean_f1, "std_f1": std_f1}

        print(f"\033[1mRandom {frac * 100:.1f}%: среднее F1_macro = {mean_f1:.4f}, отклонение F1 = {std_f1:.4f}\033[0m")

    return results


def active_learning_least_confidence(
        X_train_all,
        y_train_all,
        X_test,
        y_test,
        frac: float,
        init_frac: float = 0.005,
        seed: int = RANDOM_SEED,
) -> Dict[str, float]:
    set_seed(seed)

    n_train = X_train_all.shape[0]
    target_size = max(int(frac * n_train), N_CLASSES * 10)

    init_size = max(int(init_frac * n_train), N_CLASSES * 10)
    if init_size >= target_size:
        init_size = max(int(target_size * 0.5), N_CLASSES * 10)

    indices_all = np.arange(n_train)
    rng = np.random.RandomState(seed)
    init_idx = rng.choice(indices_all, size=init_size, replace=False)

    unlabeled_mask = np.ones(n_train, dtype=bool)
    unlabeled_mask[init_idx] = False
    unlabeled_idx = indices_all[unlabeled_mask]

    X_init = X_train_all[init_idx]
    y_init = y_train_all[init_idx]

    print(
        f"\n\033[35m[Active LC]\033[0m \033[1mДоля:\033[0m {frac * 100:.1f}% | init_size = {init_size}, target_size = {target_size}")

    model_init, f1_init = train_model_torch(
        X_init,
        y_init,
        X_test,
        y_test,
        num_classes=N_CLASSES,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        seed=seed,
        verbose=False,
    )
    print(f"  F1_macro на тесте после начального обучения (init set): {f1_init:.4f}")

    X_unlabeled = X_train_all[unlabeled_idx]
    probs = predict_proba_on_pool(model_init, X_unlabeled, batch_size=256)

    max_conf = probs.max(axis=1)
    lc_scores = 1.0 - max_conf  
    order = np.argsort(-lc_scores)  

    remaining_budget = target_size - init_size
    if remaining_budget > 0:
        chosen_unlabeled_local = order[:remaining_budget]
        chosen_unlabeled_global = unlabeled_idx[chosen_unlabeled_local]
        final_indices = np.concatenate([init_idx, chosen_unlabeled_global])
    else:
        final_indices = init_idx[:target_size]

    X_final = X_train_all[final_indices]
    y_final = y_train_all[final_indices]

    model_final, f1_final = train_model_torch(
        X_final,
        y_final,
        X_test,
        y_test,
        num_classes=N_CLASSES,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        seed=seed + 1,
        verbose=False,
    )

    print(f"  Итоговый размер размеченного набора: {len(final_indices)} "
          f"({len(final_indices) / n_train * 100:.2f}% от train)")
    print(f"\n\033[1mActive LC {frac * 100:.1f}%: F1_macro на тесте = {f1_final:.4f}\033[0m")

    return {
        "f1": float(f1_final),
        "n_labeled": int(len(final_indices)),
        "effective_fraction": float(len(final_indices) / n_train),
    }


def active_learning_margin_sampling(
        X_train_all,
        y_train_all,
        X_test,
        y_test,
        frac: float,
        init_frac: float = 0.005,
        seed: int = RANDOM_SEED,
) -> Dict[str, float]:
    set_seed(seed)

    n_train = X_train_all.shape[0]
    target_size = max(int(frac * n_train), N_CLASSES * 10)

    init_size = max(int(init_frac * n_train), N_CLASSES * 10)
    if init_size >= target_size:
        init_size = max(int(target_size * 0.5), N_CLASSES * 10)

    indices_all = np.arange(n_train)
    rng = np.random.RandomState(seed)
    init_idx = rng.choice(indices_all, size=init_size, replace=False)

    unlabeled_mask = np.ones(n_train, dtype=bool)
    unlabeled_mask[init_idx] = False
    unlabeled_idx = indices_all[unlabeled_mask]

    X_init = X_train_all[init_idx]
    y_init = y_train_all[init_idx]

    print(
        f"\n\033[36m[Active Margin]\033[0m \033[1mДоля:\033[0m {frac * 100:.1f}% | init_size = {init_size}, target_size = {target_size}")

    model_init, f1_init = train_model_torch(
        X_init,
        y_init,
        X_test,
        y_test,
        num_classes=N_CLASSES,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        seed=seed,
        verbose=False,
    )
    print(f"  F1_macro на тесте после начального обучения (init set): {f1_init:.4f}")

    X_unlabeled = X_train_all[unlabeled_idx]
    probs = predict_proba_on_pool(model_init, X_unlabeled, batch_size=256)

    probs_sorted = np.sort(probs, axis=1)

    probs_top = probs_sorted[:, -1]

    probs_second = probs_sorted[:, -2]

    margin_scores = probs_top - probs_second

    order = np.argsort(margin_scores) 

    remaining_budget = target_size - init_size
    if remaining_budget > 0:
        chosen_unlabeled_local = order[:remaining_budget]
        chosen_unlabeled_global = unlabeled_idx[chosen_unlabeled_local]
        final_indices = np.concatenate([init_idx, chosen_unlabeled_global])
    else:
        final_indices = init_idx[:target_size]

    X_final = X_train_all[final_indices]
    y_final = y_train_all[final_indices]

    model_final, f1_final = train_model_torch(
        X_final,
        y_final,
        X_test,
        y_test,
        num_classes=N_CLASSES,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        seed=seed + 1,
        verbose=False,
    )

    print(f"  Итоговый размер размеченного набора: {len(final_indices)} "
          f"({len(final_indices) / n_train * 100:.2f}% от train)")
    print(f"\n\033[1mActive Margin {frac * 100:.1f}%: F1_macro на тесте = {f1_final:.4f}\033[0m")

    return {
        "f1": float(f1_final),
        "n_labeled": int(len(final_indices)),
        "effective_fraction": float(len(final_indices) / n_train),
    }


def main():
    print(f"\n{'-' * 25}")
    print(f"\033[32mЗагрузка датасета\033[0m \033[34mAG News\033[0m")
    print(f"{'-' * 25}")

    df = load_agnews_from_kaggle(TRAIN_CSV)
    print(f"\n\033[1mВсего образцов\033[0m: {len(df)}")

    print(f"\n{'-' * 63}")
    print("\033[32mРазделение на train/test\033[0m (90/10; пропорции классов сохраняются)")
    print(f"{'-' * 63}")
    train_df, test_df = train_test_split(
        df,
        test_size=TRAIN_TEST_SPLIT,
        stratify=df["label"],
        random_state=RANDOM_SEED,
    )
    print(f"\n\033[1mРазмер train:\033[0m {len(train_df)}\n\033[1mРазмер test:\033[0m {len(test_df)}")

    print(f"\n{'-' * 27}")
    print("\033[32mПостроение TF-IDF признаков\033[0m")
    print(f"{'-' * 27}")
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
    )
    X_train_all = vectorizer.fit_transform(train_df["text"].values)
    X_test = vectorizer.transform(test_df["text"].values)

    y_train_all = train_df["label"].values
    y_test = test_df["label"].values

    print(
        f"\nX_train_all: {X_train_all.shape}. {X_train_all.shape[0]} новостей в обучающем наборе; {X_train_all.shape[1]} признаков.")
    print(f"X_test: {X_test.shape}. {X_test.shape[0]} новостей в тестовом наборе; {X_test.shape[1]} признаков.")

    print(f"\n{'-' * 36}")
    print("\033[32mОбучение на 100% обучающего датасета\033[0m")
    print(f"{'-' * 36}")
    _, f1_full = train_model_torch(
        X_train_all,
        y_train_all,
        X_test,
        y_test,
        num_classes=N_CLASSES,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        seed=RANDOM_SEED,
        verbose=False,
    )
    print(f"\n\033[1mF1_macro (100% обучающего датасета) = {f1_full:.4f}\033[0m")

    print(f"\n{'-' * 61}")
    print("\033[32mСлучайные выборки из обучающего датасета 1%, 10% и 20% данных\033[0m")
    print(f"{'-' * 61}")
    random_results = run_random_baselines(
        X_train_all,
        y_train_all,
        X_test,
        y_test
    )

    print(f"\n{'-' * 81}")
    print("\033[32mАктивное обучение: Least Confidence и Margin Sampling на 1%, 10% и 20% данных\033[0m")
    print(f"{'-' * 81}")

    al_results = {"lc": {}, "margin": {}}

    for frac in FRACTIONS:
        # Least Confidence (LC)
        al_results["lc"][frac] = active_learning_least_confidence(
            X_train_all,
            y_train_all,
            X_test,
            y_test,
            frac=frac,
            init_frac=0.005,
            seed=RANDOM_SEED,
        )

        al_results["margin"][frac] = active_learning_margin_sampling(
            X_train_all,
            y_train_all,
            X_test,
            y_test,
            frac=frac,
            init_frac=0.005,
            seed=RANDOM_SEED + 10,
        )

    rows = []

    rows.append({
        "setting": "full_100",
        "train_fraction": 1.0,
        "strategy": "full_train",
        "mean_f1": f1_full,
        "std_f1": 0.0,
        "n_labeled": len(y_train_all),
    })

    for frac in FRACTIONS:
        key = f"random_{frac}"
        rows.append({
            "setting": f"random_{frac}",
            "train_fraction": frac,
            "strategy": "random",
            "mean_f1": random_results[key]["mean_f1"],
            "std_f1": random_results[key]["std_f1"],
            "n_labeled": int(frac * len(y_train_all)),
        })

    for frac in FRACTIONS:
        rows.append({
            "setting": f"activeLC_{frac}",
            "train_fraction": al_results["lc"][frac]["effective_fraction"],
            "strategy": "active_least_confidence",
            "mean_f1": al_results["lc"][frac]["f1"],
            "std_f1": 0.0,
            "n_labeled": al_results["lc"][frac]["n_labeled"],
        })

    for frac in FRACTIONS:
        rows.append({
            "setting": f"activeMargin_{frac}",
            "train_fraction": al_results["margin"][frac]["effective_fraction"],
            "strategy": "active_margin_sampling",
            "mean_f1": al_results["margin"][frac]["f1"],
            "std_f1": 0.0,
            "n_labeled": al_results["margin"][frac]["n_labeled"],
        })

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values(by=["strategy", "train_fraction"]).reset_index(drop=True)

    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\n{'-' * 23}")
    print(f"\033[32mРезультаты:\033[0m \033[34m{RESULTS_CSV}\033[0m")
    print(f"{'-' * 23}")

    plt.figure(figsize=(8, 5))
    random_mask = results_df["strategy"] == "random"
    plt.errorbar(
        results_df[random_mask]["train_fraction"],
        results_df[random_mask]["mean_f1"],
        yerr=results_df[random_mask]["std_f1"],
        fmt="o-",
        label="Случайная выборка",
        color='green'
    )

    al_lc_mask = results_df["strategy"] == "active_least_confidence"
    plt.plot(
        results_df[al_lc_mask]["train_fraction"],
        results_df[al_lc_mask]["mean_f1"],
        "o-",
        label="Активное обучение (Least Confidence)",
        color='red'
    )

    al_margin_mask = results_df["strategy"] == "active_margin_sampling"
    plt.plot(
        results_df[al_margin_mask]["train_fraction"],
        results_df[al_margin_mask]["mean_f1"],
        "s--",  # s - квадратный маркер, -- - пунктирная линия
        label="Активное обучение (Margin Sampling)",
        color='orange'
    )

    full_mask = results_df["strategy"] == "full_train"
    plt.axhline(
        y=results_df[full_mask]["mean_f1"].values[0],
        linestyle="-",
        label="Полное обучение (100%)",
        color='blue'
    )

    plt.xlabel("Доля обучающих данных")
    plt.ylabel("F1-macro на тесте")
    plt.title("Сравнение случайного выбора и активного обучения")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_PLOT, dpi=300)
    print(f"\n{'-' * 19}")
    print(f"\033[32mГрафик:\033[0m \033[34m{RESULTS_PLOT}\033[0m")
    print(f"{'-' * 19}")


if __name__ == "__main__":
    main()
