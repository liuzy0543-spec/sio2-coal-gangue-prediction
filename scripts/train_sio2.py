# -*- coding: utf-8 -*-
"""SiO2（二氧化硅）含量预测 —— 自包含训练 / 评估 / 保存脚本

==========================================================================
1. 模型规格
==========================================================================
  目标变量 : SiO2（wt%）
  输入特征 : LOI（烧失量）、_den（闭合分母）、K2O
             以及 4 个类别字段 province / city / gangue_source_type / material_stage
             （OneHot 编码进入模型）
  模型     : RidgeCV，alpha 在 logspace(-2, 4, 12) 中折内自选
  预处理   : 中位数插补（含缺失指示器）→ StandardScaler → OneHot(min_frequency=2)
  样本权重 : 均匀（评估指标为样本加权 R²，故权重与目标一致）
  输出裁剪 : clip(预测, 0, 100)

  _den 定义：_den = MgO + K2O + Na2O + TiO2 + SO3 + LOI
  物理含义：煤矸石全分析满足 Σ四氧化物 + Σ非目标氧化物 + LOI ≈ 100，
            故 (100 - _den) 约为四氧化物的总量份额。
            该特征只由**非目标**字段构成，不使用任何目标氧化物，无标签泄露。

==========================================================================
2. 防泄露设计
==========================================================================
  ① 交叉验证按 source_group_id 分组：同一文献 / 同一矿区的样品不跨折
  ② 插补 / 标准化 / OneHot 全部只在训练折 fit
  ③ RidgeCV 的 alpha 折内自选
  ④ 5 个不同分组种子 [42, 7, 2024, 13, 99] 重复评估，报 均值 ± 标准差
  ⑤ 不使用目标编码（243 条样本、136 个来源组下风险高且无增益，已实测验证）

==========================================================================
3. 用法
==========================================================================
  python train_sio2.py                       # 训练 + 评估 + 保存模型
  python train_sio2.py --predict new.xlsx    # 用已训练模型批量预测
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_XLSX = ROOT / "data" / "SiO2_建模数据.xlsx"
MODEL_PKL = ROOT / "models" / "sio2_model.joblib"
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(exist_ok=True)

TARGET = "SiO2"
NUM_FEATS = ["LOI烧失量", "_den闭合分母", "K2O"]
CAT_FEATS = ["province", "city", "gangue_source_type", "material_stage"]
GROUP = "source_group_id"
SEEDS = [42, 7, 2024, 13, 99]
NFOLD = 5


# ----------------------------------------------------------------------
def load_data(path: Path = DATA_XLSX) -> pd.DataFrame:
    """读入建模数据并做类型规整。"""
    df = pd.read_excel(path, sheet_name="建模数据")
    for c in NUM_FEATS + [TARGET]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CAT_FEATS:
        if c not in df.columns:
            df[c] = "缺失"
        df[c] = df[c].fillna("缺失").astype(str)
    return df


def group_folds(groups, nfolds: int = NFOLD, seed: int = 42) -> dict:
    """按来源组贪心分折：大组优先放入当前样本最少的折。"""
    rng = np.random.default_rng(seed)
    uniq = list(pd.unique(groups))
    rng.shuffle(uniq)
    sizes = Counter(groups)
    uniq.sort(key=lambda g: -sizes[g])
    load = np.zeros(nfolds)
    out = {}
    for g in uniq:
        f = int(np.argmin(load))
        out[g] = f + 1
        load[f] += sizes[g]
    return out


def build_pipeline() -> Pipeline:
    """数值 + 类别的联合预处理（OneHot 与数值标准化并置）。"""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import FeatureUnion  # noqa: F401

    num = Pipeline([
        ("imp", SimpleImputer(strategy="median", add_indicator=True)),
        ("sc", StandardScaler()),
    ])
    cat = Pipeline([
        ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=2,
                             sparse_output=False, dtype=np.float32)),
    ])
    pre = ColumnTransformer([
        ("num", num, NUM_FEATS),
        ("cat", cat, CAT_FEATS),
    ], remainder="drop")
    return Pipeline([
        ("pre", pre),
        ("ridge", RidgeCV(alphas=np.logspace(-2, 4, 12))),
    ])


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """在一折上训练并预测。预处理只在 train 上 fit。"""
    pipe = build_pipeline()
    pipe.fit(train[NUM_FEATS + CAT_FEATS], train[TARGET].to_numpy(float))
    return np.clip(pipe.predict(test[NUM_FEATS + CAT_FEATS]), 0, 100)


def cross_validate(df: pd.DataFrame, verbose: bool = True):
    """5 折 × 5 个分组种子。返回 (汇总指标, 种子42的OOF预测)。"""
    df = df.copy().reset_index(drop=True)
    per_seed, oof42 = [], None
    for seed in SEEDS:
        fold = df[GROUP].map(group_folds(df[GROUP].to_numpy(), NFOLD, seed)).astype(int)
        pred = np.full(len(df), np.nan)
        for f in range(1, NFOLD + 1):
            m = (fold == f).to_numpy()
            if m.sum() < 5:
                continue
            pred[m] = fit_predict(df.loc[~m].reset_index(drop=True),
                                  df.loc[m].reset_index(drop=True))
        y = df[TARGET].to_numpy(float)
        ok = np.isfinite(pred) & np.isfinite(y)
        per_seed.append({
            "seed": seed,
            "R2": float(r2_score(y[ok], pred[ok])),
            "MAE": float(mean_absolute_error(y[ok], pred[ok])),
            "RMSE": float(np.sqrt(mean_squared_error(y[ok], pred[ok]))),
            "n_te": int(ok.sum()),
        })
        if seed == SEEDS[0]:
            oof42 = pred
        if verbose:
            print(f"    seed={seed:<5} R²={per_seed[-1]['R2']:+.4f}  "
                  f"MAE={per_seed[-1]['MAE']:.3f}  RMSE={per_seed[-1]['RMSE']:.3f}")
    r2s = [x["R2"] for x in per_seed]
    summary = {
        "R2_mean": float(np.mean(r2s)),
        "R2_sd": float(np.std(r2s, ddof=1)),
        "R2_min": float(np.min(r2s)),
        "R2_max": float(np.max(r2s)),
        "MAE_mean": float(np.mean([x["MAE"] for x in per_seed])),
        "RMSE_mean": float(np.mean([x["RMSE"] for x in per_seed])),
        "n": int(len(df)),
        "groups": int(df[GROUP].nunique()),
        "per_seed": per_seed,
    }
    return summary, oof42


def train_full(df: pd.DataFrame) -> dict:
    """用全部数据训练最终模型并打包。"""
    pipe = build_pipeline()
    pipe.fit(df[NUM_FEATS + CAT_FEATS], df[TARGET].to_numpy(float))
    return {
        "target": TARGET,
        "num_feats": NUM_FEATS,
        "cat_feats": CAT_FEATS,
        "pipeline": pipe,
        "ridge_alpha": float(pipe.named_steps["ridge"].alpha_),
        "n_train": int(len(df)),
        "groups_train": int(df[GROUP].nunique()),
        "den_definition": "MgO + K2O + Na2O + TiO2 + SO3 + LOI",
        "note": "RidgeCV + OneHot(4类别) + 3数值特征；均匀样本权重；无目标编码",
    }


def predict(model: dict, df: pd.DataFrame) -> np.ndarray:
    d = df.copy()
    for c in NUM_FEATS:
        if c not in d.columns:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in CAT_FEATS:
        if c not in d.columns:
            d[c] = "缺失"
        d[c] = d[c].fillna("缺失").astype(str)
    return np.clip(model["pipeline"].predict(d[NUM_FEATS + CAT_FEATS]), 0, 100)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="SiO2 预测模型")
    ap.add_argument("--predict", type=str, default=None,
                    help="待预测文件（xlsx/csv），需含 LOI烧失量/_den闭合分母/K2O 及 4 个类别列")
    ap.add_argument("--retrain", action="store_true", help="强制重新训练")
    args = ap.parse_args()

    if args.predict:
        if not MODEL_PKL.exists():
            raise SystemExit("未找到模型，请先运行 python train_sio2.py")
        model = joblib.load(MODEL_PKL)
        p = Path(args.predict)
        df = pd.read_excel(p) if p.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(p)
        df["SiO2_预测"] = np.round(predict(model, df), 2)
        out = OUTDIR / "predictions.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(df[["SiO2_预测"]].head(20).to_string())
        print(f"\n[已写出] {out}")
        return

    print("=" * 76)
    print(" SiO2 预测模型 —— 训练与评估")
    print("=" * 76)
    df = load_data()
    print(f"  数据文件 : {DATA_XLSX.name}")
    print(f"  样本     : {len(df)} 条 / {df[GROUP].nunique()} 个来源组")
    print(f"  数值特征 : {' + '.join(NUM_FEATS)}")
    print(f"  类别特征 : {', '.join(CAT_FEATS)}")
    print(f"  目标     : {TARGET}，均值 {df[TARGET].mean():.2f}，"
          f"标准差 {df[TARGET].std():.2f}")
    print("\n  评估：5 折（按来源组分组）× 5 个分组种子")
    res, oof = cross_validate(df)

    print("\n" + "-" * 76)
    print(f"  R²   = {res['R2_mean']:+.4f}  (sd {res['R2_sd']:.4f}, "
          f"范围 {res['R2_min']:+.4f} ~ {res['R2_max']:+.4f})")
    print(f"  MAE  = {res['MAE_mean']:.3f} wt%")
    print(f"  RMSE = {res['RMSE_mean']:.3f} wt%")
    y = df[TARGET].to_numpy(float)
    ok = np.isfinite(oof) & np.isfinite(y)
    rel = (oof[ok] - y[ok]) / y[ok]
    print(f"  相对误差 (预测−真实)/真实：均值 {np.mean(rel):+.2%}，"
          f"中位数 {np.median(rel):+.2%}")
    print(f"  |相对误差| ≤ 10% : {np.mean(np.abs(rel) <= 0.10):.1%}")
    print(f"  |相对误差| ≤ 20% : {np.mean(np.abs(rel) <= 0.20):.1%}")
    print(f"  |相对误差| ≤ 30% : {np.mean(np.abs(rel) <= 0.30):.1%}")
    print("-" * 76)

    pd.DataFrame(res["per_seed"]).to_csv(OUTDIR / "cv_results.csv",
                                         index=False, encoding="utf-8-sig")
    oofdf = pd.DataFrame({
        "sample_id": df["sample_id"], "source_group_id": df[GROUP],
        "province": df["province"], "真实值": y, "OOF预测": oof,
    })
    oofdf["误差"] = oofdf["OOF预测"] - oofdf["真实值"]
    oofdf["相对误差"] = np.where(oofdf["真实值"] != 0,
                              oofdf["误差"] / oofdf["真实值"], np.nan)
    oofdf.to_csv(OUTDIR / "oof_predictions.csv", index=False, encoding="utf-8-sig")

    final = train_full(df)
    final["cv_metrics"] = {k: v for k, v in res.items() if k != "per_seed"}
    joblib.dump(final, MODEL_PKL)
    print(f"\n  最终模型（全量 {final['n_train']} 条 / {final['groups_train']} 组）")
    print(f"    RidgeCV 选定 alpha = {final['ridge_alpha']:.4g}")
    print(f"    [已保存] {MODEL_PKL}")

    metrics = {
        "target": TARGET,
        "model": "RidgeCV(alphas=logspace(-2,4,12)) + OneHot(4 cat) + 3 numeric",
        "cv_protocol": "5-fold by source_group_id × 5 group seeds [42,7,2024,13,99]",
        "n": res["n"], "groups": res["groups"],
        "num_feats": NUM_FEATS, "cat_feats": CAT_FEATS,
        "R2_mean": round(res["R2_mean"], 4),
        "R2_sd": round(res["R2_sd"], 4),
        "R2_min": round(res["R2_min"], 4),
        "R2_max": round(res["R2_max"], 4),
        "MAE_wt%": round(res["MAE_mean"], 3),
        "RMSE_wt%": round(res["RMSE_mean"], 3),
        "rel_err_mean": round(float(np.mean(rel)), 4),
        "rel_err_median": round(float(np.median(rel)), 4),
        "within_10pct": round(float(np.mean(np.abs(rel) <= 0.10)), 4),
        "within_20pct": round(float(np.mean(np.abs(rel) <= 0.20)), 4),
        "within_30pct": round(float(np.mean(np.abs(rel) <= 0.30)), 4),
        "ridge_alpha_selected": final["ridge_alpha"],
    }
    (OUTDIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    [已写出] {OUTDIR / 'metrics.json'}")
    print(f"    [已写出] {OUTDIR / 'cv_results.csv'}")
    print(f"    [已写出] {OUTDIR / 'oof_predictions.csv'}")


if __name__ == "__main__":
    main()
