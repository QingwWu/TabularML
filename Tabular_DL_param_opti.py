"""
用于表格深度学习模型的网格参数搜索及性能验证
"""

import pandas as pd
import numpy as np
import os
import itertools
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix, cohen_kappa_score)
from pytorch_tabular import TabularModel
from pytorch_tabular.models import (CategoryEmbeddingModelConfig, TabNetModelConfig,
                                    GANDALFConfig, FTTransformerConfig)
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.getLogger("pytorch_tabular").setLevel(logging.ERROR)


# ======================== 辅助函数 ========================

def get_feature_types(df, cat_threshold=5):
    numerical_features = []
    categorical_features = []
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            converted = pd.to_numeric(df[col], errors='coerce')
            non_nan_ratio = converted.notna().mean()
            if non_nan_ratio > 0.5:
                unique_count = converted.nunique(dropna=True)
                if unique_count > cat_threshold:
                    numerical_features.append(col)
                else:
                    categorical_features.append(col)
            else:
                categorical_features.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            unique_count = df[col].nunique(dropna=True)
            if unique_count <= cat_threshold:
                categorical_features.append(col)
            else:
                numerical_features.append(col)
    return numerical_features, categorical_features


def compute_metrics(y_true, y_pred, y_prob):
    """计算所有评估指标"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    youden = sensitivity + specificity - 1
    kappa = cohen_kappa_score(y_true, y_pred)

    return {
        'test_roc_auc': roc_auc_score(y_true, y_prob),
        'test_accuracy': accuracy_score(y_true, y_pred),
        'test_precision': precision_score(y_true, y_pred, zero_division=0),
        'test_recall': sensitivity,
        'test_f1': f1_score(y_true, y_pred, zero_division=0),
        'test_tp': int(tp),
        'test_tn': int(tn),
        'test_fp': int(fp),
        'test_fn': int(fn),
        'test_npv': npv,
        'test_Kappa': kappa,
        'test_Youden': youden,
        'test_speci': specificity
    }

metric_mapping = {
    'test_roc_auc': 'AUC',
    'test_accuracy': 'ACC',
    'test_precision': 'PPV',
    'test_recall': 'Recall',
    'test_f1': 'F1',
    'test_tp': 'TP',
    'test_tn': 'TN',
    'test_fp': 'FP',
    'test_fn': 'FN',
    'test_npv': 'NPV',
    'test_Kappa': 'kappa',
    'test_Youden': 'Youden',
    'test_speci': 'Speci'
}

# ======================== 参数搜索空间 ========================
param_grids = {
    "FTTrans": {
        "num_attn_blocks": [2, 4, 6],
        "num_heads": [2, 4, 8],
        "attn_dropout": [0.0, 0.1, 0.2],
    },
    "TabNet": {
        'n_d': [8, 16, 32],
        'n_a': [8, 16, 32],
        'n_steps': [3, 5, 7],
        "gamma": [1.0, 1.5, 2.0],
    },
    "GANDALF": {
        "gflu_stages": [4, 6, 8, 10],
        "gflu_dropout": [0.0, 0.1, 0.2],
    },
    "Category": {
        "layers": ["128-64", "256-128-64", "512-256-128"],
        "dropout": [0.0, 0.1, 0.2],
    }
}

# ======================== 模型构建函数 ========================

def build_model_config(model_name, params, task="classification"):
    """根据模型名和参数字典构建模型配置"""
    if model_name == "FTTrans":
        return FTTransformerConfig(
            task=task,
            num_attn_blocks=params.get("num_attn_blocks", 4),
            num_heads=params.get("num_heads", 4),
            attn_dropout=params.get("attn_dropout", 0.0),
        )
    elif model_name == "TabNet":
        return TabNetModelConfig(
            task=task,
            n_d=params.get("n_d", 32),
            n_a=params.get("n_a", 32),
            n_steps=params.get("n_steps", 3),
            gamma=params.get("gamma", 1.5),
            n_independent=1,
            n_shared=2,
        )
    elif model_name == "GANDALF":
        return GANDALFConfig(
            task=task,
            gflu_stages=params.get("gflu_stages", 6),
            gflu_dropout=params.get("gflu_dropout", 0.0),
        )
    elif model_name == "Category":
        return CategoryEmbeddingModelConfig(
            task=task,
            layers=params.get("layers", "256-128-64"),
            dropout=params.get("dropout", 0.0),
        )
    else:
        raise ValueError(f"未知模型: {model_name}")

def get_trainer_config():
    return TrainerConfig(
        auto_lr_find=True,
        batch_size=32,
        max_epochs=100,
        load_best=False,
        trainer_kwargs=dict(enable_progress_bar=False, enable_model_summary=False),
        accelerator="cpu",
    )


def train_and_predict(train_df, test_df, model_config, data_config, label_col="label"):
    """训练模型并返回预测结果，避免标签泄露"""
    optimizer_config = OptimizerConfig()
    trainer_config = get_trainer_config()

    model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    model.fit(train=train_df)
    pred_df = model.predict(test_df)
    prob_col = [c for c in pred_df.columns if 'probability' in c.lower() and '1' in c]
    if prob_col:
        y_prob = pred_df[prob_col[0]].values
    else:
        prob_cols = [c for c in pred_df.columns if 'prob' in c.lower()]
        y_prob = pred_df[prob_cols[-1]].values if prob_cols else np.zeros(len(pred_df))
    pred_label_col = [c for c in pred_df.columns if 'prediction' in c.lower()]
    if pred_label_col:
        y_pred = pred_df[pred_label_col[0]].values.astype(int)
    else:
        y_pred = (y_prob >= 0.5).astype(int)
    return y_pred, y_prob

# ======================== 网格搜索（内层CV，避免标签泄露）========================

def grid_search_best_params(X_train, y_train, model_name, numerical_features,
                            categorical_features, n_splits=3, random_state=42):
    """
    在训练集内部进行3折交叉验证网格搜索，返回最优参数。
    注意：此函数只在训练集上运行，不接触测试集，避免标签泄露。
    """
    param_grid = param_grids[model_name]
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_score = -np.inf
    best_params = all_combinations[0]

    data_config = DataConfig(
        target=["label"],
        continuous_cols=numerical_features,
        categorical_cols=categorical_features
    )

    for params in all_combinations:
        fold_scores = []
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train, y_train):
            X_inner_train = X_train.iloc[inner_train_idx].copy()
            X_inner_val = X_train.iloc[inner_val_idx].copy()
            y_inner_train = y_train.iloc[inner_train_idx]
            y_inner_val = y_train.iloc[inner_val_idx]

            train_df = X_inner_train.copy()
            train_df['label'] = y_inner_train.values
            val_df = X_inner_val.copy()
            val_df['label'] = y_inner_val.values

            try:
                model_config = build_model_config(model_name, params)
                y_pred, y_prob = train_and_predict(train_df, val_df, model_config, data_config)
                score = roc_auc_score(y_inner_val.values, y_prob)
                fold_scores.append(score)
            except Exception as e:
                print(f"    [网格搜索] 参数 {params} 内层折失败: {e}")
                fold_scores.append(0.0)

        mean_score = np.mean(fold_scores)
        print(f"  [网格搜索] 模型={model_name}, 参数={params}, 内层AUC={mean_score:.4f}")
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    print(f"  [网格搜索] 最优参数={best_params}, 最优内层AUC={best_score:.4f}")
    return best_params

# ======================== 主流程 ========================

folder_path = "/data/Medical_ML/Tabular/all_data"
output_path = "/data/Medical_ML/Tabular/results_DL_param_opti/"
os.makedirs(output_path, exist_ok=True)

all_data = []
all_numerical_features = []
all_categorical_features = []
file_list = os.listdir(folder_path)
file_list_order = sorted(file_list, key=lambda x: x.lower())

for filename in file_list_order:
    if filename.endswith(('.xlsx', '.xls')):
        file_path = os.path.join(folder_path, filename)
        try:
            df = pd.read_excel(file_path)
            print(f"原始数据：{df.shape[0]}行，{df.shape[1]}列")
            numerical_feature, categorical_feature = get_feature_types(
                df.drop(columns=['label', "source_file"])
            )
            all_numerical_features.append(numerical_feature)
            all_categorical_features.append(categorical_feature)
            all_data.append(df)
            print(f"成功读取：{filename}，数据行数：{len(df)}")
        except Exception as e:
            print(f"读取 {filename} 失败：{str(e)}")

# 10次5折交叉验证
N_REPEATS = 10
N_SPLITS = 5

for dID in range(1):
    print(f"\n{'='*60}")
    print(f"  开始第 {dID} 个数据集")
    print(f"{'='*60}")

    numerical_features = all_numerical_features[dID]
    categorical_features = all_categorical_features[dID]
    X = all_data[dID].drop(columns=['label', "source_file"])
    y = all_data[dID]['label']

    data_config = DataConfig(
        target=["label"],
        continuous_cols=numerical_features,
        categorical_cols=categorical_features
    )

    model_names = ["FTTrans", "TabNet", "GANDALF", "Category"]

    all_results = {m: [] for m in model_names}

    for repeat in range(N_REPEATS):
        print(f"\n--- 第 {repeat+1}/{N_REPEATS} 次重复 ---")
        outer_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                   random_state=repeat * 100 + 42)

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            print(f"\n  [折 {fold_idx+1}/{N_SPLITS}]")
            X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            test_df = X_test.copy()
            test_df['label'] = y_test.values

            for model_name in model_names:
                print(f"\n  模型: {model_name}")

                # 网格搜索最优参数（仅在训练集上，避免标签泄露）
                best_params = grid_search_best_params(
                    X_train, y_train, model_name,
                    numerical_features, categorical_features,
                    n_splits=3, random_state=repeat * 100 + fold_idx
                )

                # 用最优参数在完整训练集上训练，然后在测试集上评估
                train_df = X_train.copy()
                train_df['label'] = y_train.values

                try:
                    model_config = build_model_config(model_name, best_params)
                    y_pred, y_prob = train_and_predict(
                        train_df, test_df, model_config, data_config
                    )
                    metrics = compute_metrics(y_test.values, y_pred, y_prob)
                    metrics['repeat'] = repeat + 1
                    metrics['fold'] = fold_idx + 1
                    metrics['best_params'] = str(best_params)
                    all_results[model_name].append(metrics)

                    print(f"    AUC={metrics['test_roc_auc']:.4f}, "
                          f"ACC={metrics['test_accuracy']:.4f}, "
                          f"F1={metrics['test_f1']:.4f}")
                except Exception as e:
                    print(f"    模型 {model_name} 训练/预测失败: {e}")

    # ======================== 汇总与保存结果 ========================

    print(f"\n{'='*60}")
    print(f"  数据集 {dID} 结果汇总")
    print(f"{'='*60}")

    summary_rows = []
    for model_name in model_names:
        results = all_results[model_name]
        if not results:
            continue
        results_df = pd.DataFrame(results)

        # 保存每折详细结果
        detail_save_path = os.path.join(output_path, f"dataset{dID}_{model_name}_fold_results.csv")
        results_df.to_csv(detail_save_path, index=False)

        # 计算均值和标准差
        metric_cols = list(metric_mapping.keys())
        row = {"Model": model_name}
        for metric_key in metric_cols:
            if metric_key in results_df.columns:
                mean_val = results_df[metric_key].mean()
                std_val = results_df[metric_key].std()
                display_name = metric_mapping[metric_key]
                row[f"{display_name}_mean"] = round(mean_val, 4)
                row[f"{display_name}_std"] = round(std_val, 4)
                row[f"{display_name}"] = f"{mean_val:.4f}±{std_val:.4f}"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_save_path = os.path.join(output_path, f"dataset{dID}_summary.csv")
    summary_df.to_csv(summary_save_path, index=False)
    print(summary_df[["Model"] + [metric_mapping[k] for k in metric_mapping.keys()]].to_string(index=False))

    # ======================== 可视化 ========================
    fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 5))
    if len(model_names) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_names):
        results = all_results[model_name]
        if results:
            auc_values = [r['test_roc_auc'] for r in results]
            ax.boxplot(auc_values, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', color='navy'))
            ax.set_title(model_name)
            ax.set_ylabel('AUC')
            ax.set_ylim(0, 1)
            ax.axhline(y=np.mean(auc_values), color='red', linestyle='--',
                       label=f'Mean={np.mean(auc_values):.4f}')
            ax.legend(fontsize=8)

    plt.suptitle(f'Dataset {dID} - AUC Distribution (10×5-Fold CV)', fontsize=14)
    plt.tight_layout()
    fig_path = os.path.join(output_path, f"dataset{dID}_auc_boxplot.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 多指标对比柱状图
    key_metrics = ['AUC', 'ACC', 'F1', 'Recall', 'Speci', 'PPV']
    x = np.arange(len(key_metrics))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model_name in enumerate(model_names):
        results = all_results[model_name]
        if not results:
            continue
        means = []
        stds = []
        for km in key_metrics:
            rev_map = {v: k for k, v in metric_mapping.items()}
            mkey = rev_map.get(km)
            if mkey:
                vals = [r[mkey] for r in results]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            else:
                means.append(0)
                stds.append(0)

        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, label=model_name,
                      yerr=stds, capsize=3, alpha=0.8)

    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title(f'Dataset {dID} - Model Comparison (10×5-Fold CV)')
    ax.set_xticks(x)
    ax.set_xticklabels(key_metrics)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig_path2 = os.path.join(output_path, f"dataset{dID}_metrics_comparison.png")
    plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n结果已保存至: {output_path}")
    print(f"  详细结果: dataset{dID}_<model>_fold_results.csv")
    print(f"  汇总结果: dataset{dID}_summary.csv")
    print(f"  图表: dataset{dID}_auc_boxplot.png, dataset{dID}_metrics_comparison.png")

print("\n所有数据集处理完成！")