import itertools
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve, auc, 
                            cohen_kappa_score, confusion_matrix, 
                            precision_recall_curve, average_precision_score)
from sklearn.model_selection import (train_test_split, RepeatedStratifiedKFold,
                                    GridSearchCV, cross_validate, RandomizedSearchCV,
                                    StratifiedKFold)
from tabpfn import TabPFNClassifier
import pandas as pd
from sklearn.utils import resample
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostClassifier
import copy
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from pytorch_tabular import TabularModel
from pytorch_tabular.models import (CategoryEmbeddingModelConfig, TabNetModelConfig, 
                                    GANDALFConfig, FTTransformerConfig)
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from autogluon.tabular import TabularDataset, TabularPredictor
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.getLogger("pytorch_tabular").setLevel(logging.ERROR)

def get_feature_types(df, cat_threshold=5):
    """筛选连续数值变量和分类变量"""
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
        else:
            categorical_features.append(col)
    return numerical_features, categorical_features

import scipy.stats

def delong_test(y_true, y_pred1, y_pred2):
    """Delong检验"""
    y_true = np.asarray(y_true)
    y_pred1 = np.asarray(y_pred1)
    y_pred2 = np.asarray(y_pred2)
    
    pos_preds1 = y_pred1[y_true == 1]
    neg_preds1 = y_pred1[y_true == 0]
    pos_preds2 = y_pred2[y_true == 1]
    neg_preds2 = y_pred2[y_true == 0]
    
    n_pos = len(pos_preds1)
    n_neg = len(neg_preds1)
    
    v10_1 = np.zeros((n_pos, 1))
    for i in range(n_pos):
        v10_1[i] = np.mean(neg_preds1 < pos_preds1[i])
    
    v10_2 = np.zeros((n_pos, 1))
    for i in range(n_pos):
        v10_2[i] = np.mean(neg_preds2 < pos_preds2[i])
    
    v01_1 = np.zeros((n_neg, 1))
    for i in range(n_neg):
        v01_1[i] = np.mean(pos_preds1 > neg_preds1[i])
    
    v01_2 = np.zeros((n_neg, 1))
    for i in range(n_neg):
        v01_2[i] = np.mean(pos_preds2 > neg_preds2[i])
    
    auc1 = np.mean(v10_1)
    auc2 = np.mean(v10_2)
    
    s10 = np.cov(v10_1.T, v10_2.T, ddof=1)
    s01 = np.cov(v01_1.T, v01_2.T, ddof=1)
    
    cov_1 = s10
    cov_2 = s01
    
    var_auc1 = (cov_1[0,0] / n_pos) + (cov_2[0,0] / n_neg)
    var_auc2 = (cov_1[1,1] / n_pos) + (cov_2[1,1] / n_neg)
    cov_auc1_auc2 = (cov_1[0,1] / n_pos) + (cov_2[0,1] / n_neg)
    
    var_diff = var_auc1 + var_auc2 - 2 * cov_auc1_auc2
    
    z_score = (auc1 - auc2) / np.sqrt(var_diff)
    p_value = 2 * (1 - scipy.stats.norm.cdf(np.abs(z_score)))
    
    return p_value

def impute_features(df_train_test, continuous_features, categorical_features):
    """插值函数"""
    df_train_test_new = df_train_test.copy()
    
    for col in continuous_features:
        if col in df_train_test_new.columns:
            median_val = df_train_test_new[col].median()
            df_train_test_new[col] = df_train_test_new[col].fillna(median_val)
    
    for col in categorical_features:
        if col in df_train_test_new.columns:
            mode_val = df_train_test_new[col].mode()
            if not mode_val.empty:
                df_train_test_new[col] = df_train_test_new[col].fillna(mode_val[0])
    
    return df_train_test_new

# ==================== 主程序 ====================
folder_path = "/data/Medical_ML/Tabular/all_data"
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
            numerical_feature, categorical_feature = get_feature_types(df.drop(columns=['label']))
            all_numerical_features.append(numerical_feature)
            all_categorical_features.append(categorical_feature)
            all_data.append(df)
            print(f"成功读取：{filename}，数据行数：{len(df)}")
        except Exception as e:
            print(f"读取 {filename} 失败：{str(e)}")

for dID in range(1, 8):
    print(f"=============== 开始第{dID}个数据集 ====================")
    output_path = "/data/Medical_ML/Tabular/results/"
    
    numerical_features = all_numerical_features[dID]
    categorical_features = all_categorical_features[dID]
    
    X = all_data[dID].drop(columns=['label'])
    y = all_data[dID]['label']
    
    # 外层CV：用于无偏性能评估 (5折10次重复)
    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    # 内层CV：用于超参数调优（仅使用训练数据）
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # 定义模型和参数
    models_and_params = {
        'LR': (LogisticRegression(solver='liblinear', random_state=42, max_iter=1000), {
            'model__C': [0.01, 0.1, 1, 10],
            'model__penalty': ['l1', 'l2']
        }),
        'XGB': (XGBClassifier(objective='binary:logistic', random_state=42, eval_metric='logloss', use_label_encoder=False), {
            'model__n_estimators': [50, 100, 150],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__max_depth': [3, 4, 5],
            'model__subsample': [0.6, 0.8, 1.0],
            'model__colsample_bytree': [0.6, 0.8, 1.0],
            'model__gamma': [0, 0.1]
        }),
        'RF': (RandomForestClassifier(random_state=42), {
            'model__n_estimators': [50, 100, 150],
            'model__max_depth': [3, 4, 5, None],
            'model__min_samples_leaf': [5, 10, 20],
            'model__min_samples_split': [10, 20],
            'model__max_features': ['sqrt', 'log2'],
        }),
        'CatBoost': (CatBoostClassifier(task_type="CPU", verbose=0, random_state=42), {
            'model__iterations': [50, 100, 150],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__depth': [3, 4, 5],
            'model__l2_leaf_reg': [1, 3],
            'model__subsample': [0.6, 0.8, 1.0],
        }),
    }
    
    # 定义预处理器
    numeric_transformer_for_lr = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])
    categorical_transformer_for_lr = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    preprocessor_for_lr = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer_for_lr, numerical_features),
            ('cat', categorical_transformer_for_lr, categorical_features)
        ],
        remainder='passthrough'
    )
    
    numeric_transformer_for_trees = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    categorical_transformer_for_trees = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
    ])
    preprocessor_for_trees = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer_for_trees, numerical_features),
            ('cat', categorical_transformer_for_trees, categorical_features)
        ],
        remainder='passthrough'
    )
    
    # 初始化存储结构
    all_preds = {name: {'y_true_test': [], 'y_pred_test': [], 
                       'y_true_train': [], 'y_pred_train': []} 
                for name in list(models_and_params.keys()) + ['TabPFN', 'AutoGluon', 'FTTrans', 'TabNet', 'GANDALF', 'Category']}
    
    results_list = []
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
    
    # 存储每折的最佳参数（用于分析稳定性）
    best_params_per_fold = {name: [] for name in models_and_params.keys()}
    
    # 初始化图表
    fig_roc, ax_roc = plt.subplots(figsize=(12, 10))
    fig_pr, ax_pr = plt.subplots(figsize=(12, 10))
    
    # ==================== 嵌套交叉验证主循环 ====================
    outer_fold_idx = 0
    
    for train_idx, test_idx in outer_cv.split(X, y):
        outer_fold_idx += 1
        print(f"\n========== 外层第 {outer_fold_idx}/50 折 ==========")
        
        X_train_outer, X_test_outer = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train_outer, y_test_outer = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
        
        # 插补在每次划分后、模型训练前进行，且仅使用训练集统计量
        X_train_outer = impute_features(X_train_outer, numerical_features, categorical_features)
        X_test_outer = impute_features(X_test_outer, numerical_features, categorical_features)
        
        # ----------------- 1. 传统ML模型（带超参数搜索） -----------------
        for name, (model, params) in models_and_params.items():
            print(f"--- 处理模型: {name} (第{outer_fold_idx}折) ---")
            
            # 构建pipeline
            if name == 'LR':
                pipe = Pipeline(steps=[
                    ('preprocessor', preprocessor_for_lr),
                    ('model', model)
                ])
            else:
                pipe = Pipeline(steps=[
                    ('preprocessor', preprocessor_for_trees),
                    ('model', model)
                ])
            
            # 内层CV：仅在当前外层训练集上进行超参数搜索
            search = RandomizedSearchCV(
                clone(pipe),
                param_distributions=params,
                n_iter=20,
                cv=inner_cv,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=42,
                refit=True
            )
            
            search.fit(X_train_outer, y_train_outer)
            best_model = search.best_estimator_
            current_best_params = search.best_params_
            best_params_per_fold[name].append(current_best_params)
            
            print(f"  最佳参数: {current_best_params}, 验证AUC: {search.best_score_:.4f}")
            
            # 在当前外层测试集上评估
            y_prob_test = best_model.predict_proba(X_test_outer)[:, 1]
            y_pred_test = best_model.predict(X_test_outer)
            y_prob_train = best_model.predict_proba(X_train_outer)[:, 1]
            
            # 存储结果
            all_preds[name]['y_true_test'].extend(y_test_outer)
            all_preds[name]['y_pred_test'].extend(y_prob_test)
            all_preds[name]['y_true_train'].extend(y_train_outer)
            all_preds[name]['y_pred_train'].extend(y_prob_train)
        
        # ----------------- 2. TabPFN -----------------
        print(f"--- 处理模型: TabPFN (第{outer_fold_idx}折) ---")
        tabpfn_model = TabPFNClassifier(
            model_path="/data/Medical_ML/Tabular/model_ckpt/tabpfn-v2-classifier.ckpt",
            device='cpu'
        )
        tabpfn_model.fit(X_train_outer, y_train_outer)
        y_prob_test = tabpfn_model.predict_proba(X_test_outer)[:, 1]
        y_pred_test = tabpfn_model.predict(X_test_outer)
        y_prob_train = tabpfn_model.predict_proba(X_train_outer)[:, 1]
        
        all_preds['TabPFN']['y_true_test'].extend(y_test_outer)
        all_preds['TabPFN']['y_pred_test'].extend(y_prob_test)
        all_preds['TabPFN']['y_true_train'].extend(y_train_outer)
        all_preds['TabPFN']['y_pred_train'].extend(y_prob_train)
        
        # ----------------- 3. AutoGluon -----------------
        print(f"--- 处理模型: AutoGluon (第{outer_fold_idx}折) ---")
        train_data = pd.concat([X_train_outer, y_train_outer], axis=1)
        test_data = pd.concat([X_test_outer, y_test_outer], axis=1)
        
        fold_idx = (outer_fold_idx - 1) % 5
        repeat_num = (outer_fold_idx - 1) // 5
        unique_path = f"autogluon_nested_cv/repeat_{repeat_num}/fold_{fold_idx}_{dID}"

        if os.path.exists(unique_path):
            import shutil
            shutil.rmtree(unique_path)
            
        ag_predictor = TabularPredictor(
            label="label",
            eval_metric="roc_auc",
            path=unique_path,
            problem_type='binary',
            verbosity=0
        ).fit(train_data=train_data, time_limit=60, presets='medium_quality')
        
        y_prob_test = ag_predictor.predict_proba(test_data).iloc[:, 1]
        y_pred_test = ag_predictor.predict(test_data)
        y_prob_train = ag_predictor.predict_proba(train_data).iloc[:, 1]
        
        all_preds['AutoGluon']['y_true_test'].extend(y_test_outer)
        all_preds['AutoGluon']['y_pred_test'].extend(y_prob_test)
        all_preds['AutoGluon']['y_true_train'].extend(y_train_outer)
        all_preds['AutoGluon']['y_pred_train'].extend(y_prob_train)
        
        # ----------------- 4. PyTorch Tabular Models -----------------
        data_config = DataConfig(
            target=["label"],
            continuous_cols=numerical_features,
            categorical_cols=categorical_features
        )
        trainer_config = TrainerConfig(
            auto_lr_find=True,
            batch_size=32,
            max_epochs=50,
            load_best=False,
            trainer_kwargs=dict(enable_progress_bar=False, enable_model_summary=False),
            accelerator="cpu",
        )
        optimizer_config = OptimizerConfig()
        
        tabular_configs = {
            'FTTrans': FTTransformerConfig(task="classification", num_attn_blocks=4, num_heads=4),
            'TabNet': TabNetModelConfig(task="classification", n_d=32, n_a=32, n_steps=3),
            'GANDALF': GANDALFConfig(task="classification", gflu_stages=6),
            'Category': CategoryEmbeddingModelConfig(task="classification", layers="256-128-64")
        }
        
        for tab_name, model_config in tabular_configs.items():
            print(f"--- 处理模型: {tab_name} (第{outer_fold_idx}折) ---")
            try:
                cloned_config = copy.deepcopy(model_config)
                tabular_model = TabularModel(
                    data_config=data_config,
                    model_config=cloned_config,
                    optimizer_config=optimizer_config,
                    trainer_config=trainer_config,
                    verbose=False,
                    suppress_lightning_logger=True,
                )
                
                train_data = pd.concat([X_train_outer, y_train_outer], axis=1)
                test_data = pd.concat([X_test_outer, y_test_outer], axis=1)
                
                tabular_model.fit(train=train_data, validation=test_data)
                
                pred_df_test = tabular_model.predict(test_data)
                pred_df_train = tabular_model.predict(train_data)
                
                y_prob_test = pred_df_test.iloc[:, 1]
                y_pred_test = pred_df_test.iloc[:, 2]
                y_prob_train = pred_df_train.iloc[:, 1]
                
                all_preds[tab_name]['y_true_test'].extend(y_test_outer)
                all_preds[tab_name]['y_pred_test'].extend(y_prob_test)
                all_preds[tab_name]['y_true_train'].extend(y_train_outer)
                all_preds[tab_name]['y_pred_train'].extend(y_prob_train)
                
                del tabular_model
            except Exception as e:
                print(f"  错误：{tab_name} 训练失败 - {str(e)}")
    
    # ==================== 结果汇总与可视化 ====================
    print("\n========== 嵌套CV完成，开始汇总结果 ==========")
    
    # 保存超参数稳定性分析
    output_param_path = output_path + str(dID) + "_param_stability.xlsx"
    with pd.ExcelWriter(output_param_path, engine='openpyxl') as writer:
        for model_name, params_list in best_params_per_fold.items():
            if params_list:
                params_df = pd.DataFrame(params_list)
                # 添加fold标识
                params_df['outer_fold'] = range(1, len(params_list) + 1)
                params_df.to_excel(writer, sheet_name=model_name, index=False)
                print(f"模型 {model_name} 超参数稳定性已记录，共{len(params_list)}折")
    
    # 计算并绘制结果
    all_model_names = list(all_preds.keys())
    
    for name in all_model_names:
        if not all_preds[name]['y_true_test']:
            continue
            
        y_true_all = np.array(all_preds[name]['y_true_test'])
        y_pred_all = np.array(all_preds[name]['y_pred_test'])
        y_train_true = np.array(all_preds[name]['y_true_train'])
        y_train_pred = np.array(all_preds[name]['y_pred_train'])
        
        # 计算总体指标
        fpr, tpr, _ = roc_curve(y_true_all, y_pred_all)
        roc_auc = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(y_true_all, y_pred_all)
        ap = average_precision_score(y_true_all, y_pred_all)
        
        # 绘制ROC
        ax_roc.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
        
        # 绘制PR曲线
        ax_pr.plot(recall, precision, lw=2, label=f'{name} (AP = {ap:.3f})')
        
        # 计算混淆矩阵相关指标
        y_pred_bin = (y_pred_all >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_bin).ravel()
        
        acc = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
        kappa = cohen_kappa_score(y_true_all, y_pred_bin)
        youden = sensitivity + specificity - 1
        
        model_results = {
            'Model': name,
            'AUC': f"{roc_auc:.4f}",
            'ACC': f"{acc:.4f}",
            'Sensitivity': f"{sensitivity:.4f}",
            'Specificity': f"{specificity:.4f}",
            'PPV': f"{ppv:.4f}",
            'NPV': f"{npv:.4f}",
            'F1': f"{f1:.4f}",
            'Kappa': f"{kappa:.4f}",
            'Youden': f"{youden:.4f}",
            'Total_TP': tp, 'Total_TN': tn, 'Total_FP': fp, 'Total_FN': fn
        }
        results_list.append(model_results)
    
    # 保存图表
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=2)
    ax_roc.set(xlabel='FPR', ylabel='TPR', title='ROC Curve (Nested CV)')
    ax_roc.legend(loc='lower right', prop={'size': 10})
    ax_roc.grid(True, linestyle='--', alpha=0.6)
    fig_roc.savefig(output_path + str(dID) + '_nested_auc曲线.jpg', dpi=600, bbox_inches='tight')
    plt.close(fig_roc)
    
    ax_pr.set(xlabel='Recall', ylabel='Precision', title='PR Curve (Nested CV)')
    ax_pr.legend(loc='lower left', prop={'size': 10})
    ax_pr.grid(True, linestyle='--', alpha=0.6)
    fig_pr.savefig(output_path + str(dID) + '_nested_pr曲线.jpg', dpi=600, bbox_inches='tight')
    plt.close(fig_pr)
    
    # 保存结果表格
    results_df = pd.DataFrame(results_list)
    results_df.to_excel(output_path + str(dID) + '_nested_模型评估结果.xlsx', index=False)
    
    # Delong检验
    p_value_matrix = pd.DataFrame(index=all_model_names, columns=all_model_names, dtype=float)
    for model1_name, model2_name in itertools.combinations(all_model_names, 2):
        if (all_preds[model1_name]['y_pred_test'] and 
            all_preds[model2_name]['y_pred_test']):
            
            y_test = np.array(all_preds[model1_name]['y_true_test'])
            model1_preds = np.array(all_preds[model1_name]['y_pred_test'])
            model2_preds = np.array(all_preds[model2_name]['y_pred_test'])
            
            p_value = delong_test(y_test, model1_preds, model2_preds)
            p_value_matrix.loc[model1_name, model2_name] = p_value
            p_value_matrix.loc[model2_name, model1_name] = p_value
    
    np.fill_diagonal(p_value_matrix.values, 1.0)
    p_value_matrix.to_excel(output_path + str(dID) + "_nested_delong.xlsx")
    
    print(f"第{dID}个数据集处理完成，结果已保存。")
