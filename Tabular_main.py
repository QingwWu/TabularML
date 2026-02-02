import itertools
from sklearn.metrics import accuracy_score, roc_auc_score,roc_curve,auc,cohen_kappa_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
import pandas as pd
from sklearn.utils import resample
import os
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, cross_validate, RandomizedSearchCV 
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
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve,average_precision_score
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig, TabNetModelConfig, GANDALFConfig, FTTransformerConfig
from pytorch_tabular.config import DataConfig,OptimizerConfig,TrainerConfig
from autogluon.tabular import TabularDataset, TabularPredictor

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.getLogger("pytorch_tabular").setLevel(logging.ERROR)


def get_feature_types(df, cat_threshold=5):
    """
    筛选：
      连续数值变量和分类变量
    """
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
    var_auc_diff = (s10[0, 0] / n_pos) + (s01[0, 0] / n_neg) - 2 * (s10[0, 1] / n_pos) + \
                (s10[1, 1] / n_pos) + (s01[1, 1] / n_neg) - 2 * (s01[0, 1] / n_neg)
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

folder_path = "/data/Medical_ML/Tabular/all_data" # 存放了8个excel表格
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

for dID in range(1,8):
    print(f"=============== 开始第{dID}个数据集 =====================")
    output_path = "/data/Medical_ML/Tabular/results/"
    numerical_features = all_numerical_features[dID]
    categorical_features = all_categorical_features[dID] 
    X = all_data[dID].drop(columns=['label'])
    y = all_data[dID]['label']

    data_config = DataConfig(
        target=["label"],
        continuous_cols=numerical_features,
        categorical_cols=categorical_features
    )
    trainer_config = TrainerConfig(
        auto_lr_find=True, 
        batch_size=32,
        max_epochs=100,
        load_best=False,
        trainer_kwargs=dict(enable_progress_bar=False,enable_model_summary=False),
        accelerator= "cpu",
    )
    optimizer_config = OptimizerConfig()

    CategoryEmbeddingModel = CategoryEmbeddingModelConfig(
        task="classification",
        layers="256-128-64",
    )
    TabNetModel = TabNetModelConfig(
        task="classification",
        n_d= 32,
        n_a = 32,
        n_steps = 3,
        gamma = 1.5,
        n_independent = 1,
        n_shared = 2
    )
    GANDALFModel = GANDALFConfig(
        task="classification",
        gflu_stages= 6
    )
    FTTransformer = FTTransformerConfig(
        task="classification",
        num_attn_blocks=4,
        num_heads=4,
    )
    tabular_models = {}
    tabular_models["FTTrans"] = FTTransformer
    tabular_models["TabNet"] = TabNetModel
    tabular_models["GANDALF"] = GANDALFModel
    tabular_models["Category"] = CategoryEmbeddingModel

    models_and_params = {
        'LR': (LogisticRegression(solver='liblinear', random_state=42), {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2']
        }),
        
        'XGB': (XGBClassifier(objective='binary:logistic', random_state=42, eval_metric='logloss'), {
            'n_estimators': [50, 100, 150, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6],
            'subsample': [0.6, 0.8,1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2]
        }),

        'RF': (RandomForestClassifier(random_state=42), {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [3, 4, 5, 6],
            'min_samples_leaf': [5, 10, 15, 20],
            'min_samples_split': [10, 20, 30],
            'max_features': ['sqrt', 'log2', 0.5],
            'bootstrap': [True, False]
        }),

        'CatBoost': (CatBoostClassifier(
            task_type="CPU",
            verbose=0,
            random_state=42
        ), {
            'iterations': [50, 100 , 150, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'depth': [3, 4, 5, 6],
            'l2_leaf_reg': [1, 3],
            'subsample': [0.6, 0.8, 1.0],
            'min_data_in_leaf': [5, 10, 20, 30],
            'bootstrap_type': ['Bernoulli', 'MVS'], 
        }),
    }
    cv_strategy = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

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

    best_estimators_pipe = {}
    output_param_path = output_path+str(dID)+"_param.xlsx"
    empty_df = pd.DataFrame()
    empty_df.to_excel(output_param_path, index=False, engine="openpyxl")

    for name, (model, params) in models_and_params.items():
        print(f"--- 正在处理模型: {name} ---")
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

        param_grid = {f'model__{key}': value for key, value in params.items()}
        grid_search = RandomizedSearchCV(
            pipe, 
            param_distributions=param_grid,
            n_iter=50,
            cv=cv_strategy, 
            scoring='roc_auc', 
            n_jobs=-1, 
            return_train_score=True, 
            error_score='raise',
            random_state=42
        )        
        grid_search.fit(X, y) 
        print(f"GridSearchCV找到的'官方'最佳参数: {grid_search.best_params_}")
        print(f"GridSearchCV中的'官方'最佳AUC: {grid_search.best_score_:.4f}")
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df = results_df[['params', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']]
        results_df['gap'] = results_df['mean_train_score'] - results_df['mean_test_score']
        print(f"\n--- {name} 模型的详细分析 ---")
        GAP_THRESHOLD = 0.1 
        robust_models = results_df[results_df['gap'] < GAP_THRESHOLD].copy()
        final_best_estimator = None
        if not robust_models.empty:
            robust_models_sorted = robust_models.sort_values(by='mean_test_score', ascending=False)
            best_robust_params = robust_models_sorted.iloc[0]['params']
            print(f"在差距小于 {GAP_THRESHOLD} 的模型中，找到最佳稳健参数: {best_robust_params}")
            print(f"其验证集AUC为: {robust_models_sorted.iloc[0]['mean_test_score']:.4f}, 差距为: {robust_models_sorted.iloc[0]['gap']:.4f}")
            final_best_estimator = clone(pipe)
            final_best_estimator.set_params(**best_robust_params)
            try:
                with pd.ExcelWriter(output_param_path, mode='a', engine='openpyxl') as writer:
                    pd.DataFrame([best_robust_params]).to_excel(writer, sheet_name= f'{name}_best_p')
            except Exception as e:
                print(f"\n添加最优参数到Excel失败。原因: {e}")

        else:
            print(f"\n--- 警告：未找到满足条件（差距<{GAP_THRESHOLD}）的模型 ---")
            print("将回退使用GridSearchCV的官方最佳模型进行评估。")
            final_best_estimator = grid_search.best_estimator_   
            try:
                with pd.ExcelWriter(output_param_path, mode='a', engine='openpyxl') as writer:
                    pd.DataFrame([grid_search.best_params_]).to_excel(writer, sheet_name= f'{name}_nobest_p')
            except Exception as e:
                print(f"\n添加最优参数到Excel失败。原因: {e}")

        best_estimators_pipe[name] = final_best_estimator

    TabPFN = TabPFNClassifier(model_path="/data/Medical_ML/Tabular/model_ckpt/tabpfn-v2-classifier.ckpt")
    best_estimators_pipe["TabPFN"] = TabPFN
    best_estimators_pipe["AutoGluon"] = "auto"

    model_lists = [best_estimators_pipe,tabular_models]
    all_model_names = []
    for model_dict in model_lists:
        all_model_names.extend(model_dict.keys())
    all_model_names = list(set(all_model_names))
    all_preds = {
        name: {'y_true_test': [], 'y_pred_test': [],'y_true_train': [], 'y_pred_train': []} 
        for name in all_model_names
    }
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

    fig_roc, ax_roc = plt.subplots(figsize=(12, 10))
    fig_pr, ax_pr = plt.subplots(figsize=(12, 10))

    for model_dict in model_lists:
        if model_dict == best_estimators_pipe:
            for name, model in model_dict.items():
                print(f"--- 正在为模型 {name} 生成ROC曲线数据 ---")
                accuracies_test, precisions_test, recalls_test, f1s_test = [], [], [], []
                tps_test, tns_test, fps_test, fns_test = [], [], [], []
                npvs_test = []
                kappa_test,youden_test, speci_test = [],[],[]

                tprs_test, aucs_test = [], []  
                tprs_train, aucs_train = [], []
                mean_fpr = np.linspace(0, 1, 100)

                precisions_test_interp, aps_test = [], []
                precisions_train_interp, aps_train = [], []
                mean_recall = np.linspace(0, 1, 100)

                for repeat_idx, (train_index, test_index) in enumerate(cv_strategy.split(X, y)):
                    X_train, X_test = X.iloc[train_index].copy(), X.iloc[test_index].copy()
                    y_train, y_test = y.iloc[train_index].copy(), y.iloc[test_index].copy()
                    X_train = impute_features(X_train,numerical_features,categorical_features)
                    X_test = impute_features(X_test,numerical_features,categorical_features)
                    if name == "AutoGluon":
                        train_data = pd.concat([X_train, y_train], axis=1)
                        test_data = pd.concat([X_test, y_test], axis=1)
                        fold_idx = repeat_idx % 5
                        repeat_num = repeat_idx // 5
                        unique_path = f"autogluon_cv/repeat_{repeat_num}/fold_{fold_idx}"
                        fold_predictor = TabularPredictor(
                            label="label",
                            eval_metric="roc_auc",
                            path=unique_path,
                            problem_type = 'binary'
                        ).fit(train_data=train_data, time_limit=120, presets='medium_quality',verbosity=1) 
                        y_prob_test = fold_predictor.predict_proba(test_data).iloc[:, 1]
                        y_pred_test = fold_predictor.predict(test_data)

                    else:
                        cloned_model = clone(model)
                        cloned_model.fit(X_train, y_train)
                        y_prob_test = cloned_model.predict_proba(X_test)[:, 1]
                        y_pred_test = cloned_model.predict(X_test)
                    
                    fpr_test, tpr_test, _ = roc_curve(y_test, y_prob_test)
                    aucs_test.append(auc(fpr_test, tpr_test))
                    tprs_test.append(np.interp(mean_fpr, fpr_test, tpr_test))
                    y_prob_train = cloned_model.predict_proba(X_train)[:, 1]
                    fpr_train, tpr_train, _ = roc_curve(y_train, y_prob_train)
                    aucs_train.append(auc(fpr_train, tpr_train))
                    tprs_train.append(np.interp(mean_fpr, fpr_train, tpr_train))

                    precision_test, recall_test, _ = precision_recall_curve(y_test, y_prob_test)
                    aps_test.append(average_precision_score(y_test, y_prob_test))

                    precisions_test_interp.append(np.interp(mean_recall, recall_test[::-1], precision_test[::-1]))

                    precision_train, recall_train, _ = precision_recall_curve(y_train, y_prob_train)
                    aps_train.append(average_precision_score(y_train, y_prob_train))
                    precisions_train_interp.append(np.interp(mean_recall, recall_train[::-1], precision_train[::-1]))
                    
                    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
                    tps_test.append(tp)
                    tns_test.append(tn)
                    fps_test.append(fp)
                    fns_test.append(fn)

                    kappa = cohen_kappa_score(y_test, y_pred_test)
                    kappa_test.append(kappa)

                    ppv = tp / (tp + fp) if (tp + fp) != 0 else 0.0
                    precisions_test.append(ppv)
                    
                    npv = tn / (tn + fn) if (tn + fn) != 0 else 0.0
                    npvs_test.append(npv)
                    
                    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
                    f1 = 2 * (ppv * recall) / (ppv + recall) if (ppv + recall) != 0 else 0.0
                    accuracies_test.append(acc)
                    recalls_test.append(recall)
                    f1s_test.append(f1)

                    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
                    speci_test.append(specificity)
                    youden = recall + specificity - 1
                    youden_test.append(youden)

                    all_preds[name]['y_true_test'].extend(y_test)
                    all_preds[name]['y_pred_test'].extend(y_prob_test)
                    all_preds[name]['y_true_train'].extend(y_train)
                    all_preds[name]['y_pred_train'].extend(y_prob_train)
                    
                mean_tpr_test = np.mean(tprs_test, axis=0)
                mean_auc_test = np.mean(aucs_test)
                std_auc_test = np.std(aucs_test)
                n_samples = len(aucs_test)
                auc_ci_lower = mean_auc_test - 1.96 * (std_auc_test / np.sqrt(n_samples))
                auc_ci_upper = mean_auc_test + 1.96 * (std_auc_test / np.sqrt(n_samples))
                label_test_roc = f'{name} Test (AUC = {mean_auc_test:.3f} [{auc_ci_lower:.3f}-{auc_ci_upper:.3f}])'
                line, = ax_roc.plot(mean_fpr, mean_tpr_test, label=label_test_roc, lw=2, alpha=0.9)
                
                mean_tpr_train = np.mean(tprs_train, axis=0)
                mean_auc_train = np.mean(aucs_train)
                label_train_roc = f'{name} Train (AUC = {mean_auc_train:.3f})'
                ax_roc.plot(mean_fpr, mean_tpr_train, linestyle=':', color=line.get_color(), label=label_train_roc, lw=2)

                mean_precision_test = np.mean(precisions_test_interp, axis=0)
                std_precision_test = np.std(precisions_test_interp, axis=0)
                mean_ap_test = np.mean(aps_test)
                std_ap_test = np.std(aps_test)
                ap_ci_lower = mean_ap_test - 1.96 * (std_ap_test / np.sqrt(n_samples))
                ap_ci_upper = mean_ap_test + 1.96 * (std_ap_test / np.sqrt(n_samples))
                label_test_pr = f'{name} Test (AP = {mean_ap_test:.3f} [{ap_ci_lower:.3f}-{ap_ci_upper:.3f}])'
                line_pr, = ax_pr.plot(mean_recall, mean_precision_test, label=label_test_pr, lw=2, alpha=0.9)

                mean_precision_train = np.mean(precisions_train_interp, axis=0)
                mean_ap_train = np.mean(aps_train)
                label_train_pr = f'{name} Train (AP = {mean_ap_train:.3f})'
                ax_pr.plot(mean_recall, mean_precision_train, linestyle=':', color=line_pr.get_color(), label=label_train_pr, lw=2)

                model_results = {'Model': name}
                metrics_to_calculate = {
                    'test_roc_auc': (aucs_test, 'AUC'),
                    'test_accuracy': (accuracies_test, 'ACC'),
                    'test_precision': (precisions_test, 'PPV'),
                    'test_recall': (recalls_test, 'Recall'),
                    'test_f1': (f1s_test, 'F1'),
                    'test_tp': (tps_test, 'TP'),
                    'test_tn': (tns_test, 'TN'),
                    'test_fp': (fps_test, 'FP'),
                    'test_fn': (fns_test, 'FN'),
                    'test_npv': (npvs_test, 'NPV'),
                    'test_Kappa': (kappa_test, 'kappa'),
                    'test_Youden': (speci_test, 'Youden'),
                    'test_speci': (youden_test, 'Speci'),
                }
                
                for key, (values, display_name) in metrics_to_calculate.items():
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    n = len(values)
                    std_err = std_val / np.sqrt(n) if n > 0 else 0.0
                    ci_lower = mean_val - 1.96 * std_err
                    ci_upper = mean_val + 1.96 * std_err
                    
                    model_results[f'{display_name} (mean)'] = f"{mean_val:.4f}"
                    model_results[f'{display_name} (95% CI)'] = f"[{ci_lower:.4f}, {ci_upper:.4f}]"
                    model_results[f'{display_name} (std)'] = f"{std_err:.4f}"

                results_list.append(model_results)
        
        elif model_dict == tabular_models:
            for name, base_model_config in model_dict.items():
                print(f"--- 正在为模型 {name} 生成ROC曲线数据 ---")
                accuracies_test, precisions_test, recalls_test, f1s_test = [], [], [], []
                tps_test, tns_test, fps_test, fns_test = [], [], [], []
                npvs_test = []
                kappa_test,youden_test, speci_test = [],[],[]
                tprs_test, aucs_test = [], []  
                tprs_train, aucs_train = [], []
                mean_fpr = np.linspace(0, 1, 100)

                precisions_test_interp, aps_test = [], []
                precisions_train_interp, aps_train = [], []
                mean_recall = np.linspace(0, 1, 100)
                
                for train_index, test_index in cv_strategy.split(X, y):
                    cloned_model_config = copy.deepcopy(base_model_config)
                    tabular_model = TabularModel(
                        data_config=data_config,
                        model_config=cloned_model_config,
                        optimizer_config=optimizer_config,
                        trainer_config=trainer_config,
                        verbose=False,
                        suppress_lightning_logger=False,
                    )
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    X_train = impute_features(X_train,numerical_features,categorical_features)
                    X_test = impute_features(X_test,numerical_features,categorical_features)

                    X_train = impute_features(X_train,numerical_features,categorical_features)
                    train_data = pd.concat([X_train, y_train], axis=1)
                    test_data = pd.concat([X_test, y_test], axis=1)
                    tabular_model.fit(train=train_data, validation=test_data)
                    pred_df_test = tabular_model.predict(test_data)
                    pred_df_train = tabular_model.predict(train_data)
                    y_prob_test = pred_df_test.iloc[:, 1]
                    y_pred_test = pred_df_test.iloc[:, 2]
                    
                    fpr_test, tpr_test, _ = roc_curve(y_test, y_prob_test)
                    aucs_test.append(auc(fpr_test, tpr_test))
                    tprs_test.append(np.interp(mean_fpr, fpr_test, tpr_test))
                    y_prob_train = pred_df_train.iloc[:, 1]
                    fpr_train, tpr_train, _ = roc_curve(y_train, y_prob_train)
                    aucs_train.append(auc(fpr_train, tpr_train))
                    tprs_train.append(np.interp(mean_fpr, fpr_train, tpr_train))
                    precision_test, recall_test, _ = precision_recall_curve(y_test, y_prob_test)
                    aps_test.append(average_precision_score(y_test, y_prob_test))
                    precisions_test_interp.append(np.interp(mean_recall, recall_test[::-1], precision_test[::-1]))

                    precision_train, recall_train, _ = precision_recall_curve(y_train, y_prob_train)
                    aps_train.append(average_precision_score(y_train, y_prob_train))
                    precisions_train_interp.append(np.interp(mean_recall, recall_train[::-1], precision_train[::-1]))

                    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
                    tps_test.append(tp)
                    tns_test.append(tn)
                    fps_test.append(fp)
                    fns_test.append(fn)

                    kappa = cohen_kappa_score(y_test, y_pred_test)
                    kappa_test.append(kappa)                    
                    
                    ppv = tp / (tp + fp) if (tp + fp) != 0 else 0.0
                    precisions_test.append(ppv)
                    npv = tn / (tn + fn) if (tn + fn) != 0 else 0.0
                    npvs_test.append(npv)
                    
                    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
                    f1 = 2 * (ppv * recall) / (ppv + recall) if (ppv + recall) != 0 else 0.0
                    accuracies_test.append(acc)
                    recalls_test.append(recall)
                    f1s_test.append(f1)    

                    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
                    speci_test.append(specificity)
                    youden = recall + specificity - 1
                    youden_test.append(youden)

                    all_preds[name]['y_true_test'].extend(y_test)
                    all_preds[name]['y_pred_test'].extend(y_prob_test)
                    all_preds[name]['y_true_train'].extend(y_train)
                    all_preds[name]['y_pred_train'].extend(y_prob_train)    

                    del tabular_model        
            
                mean_tpr_test = np.mean(tprs_test, axis=0)
                mean_auc_test = np.mean(aucs_test)
                std_auc_test = np.std(aucs_test)
                n_samples = len(aucs_test)
                auc_ci_lower = mean_auc_test - 1.96 * (std_auc_test / np.sqrt(n_samples))
                auc_ci_upper = mean_auc_test + 1.96 * (std_auc_test / np.sqrt(n_samples))
                label_test_roc = f'{name} Test (AUC = {mean_auc_test:.3f} [{auc_ci_lower:.3f}-{auc_ci_upper:.3f}])'
                line, = ax_roc.plot(mean_fpr, mean_tpr_test, label=label_test_roc, lw=2, alpha=0.9)
                
                mean_tpr_train = np.mean(tprs_train, axis=0)
                mean_auc_train = np.mean(aucs_train)
                label_train_roc = f'{name} Train (AUC = {mean_auc_train:.3f})'
                ax_roc.plot(mean_fpr, mean_tpr_train, linestyle=':', color=line.get_color(), label=label_train_roc, lw=2)
                mean_precision_test = np.mean(precisions_test_interp, axis=0)
                std_precision_test = np.std(precisions_test_interp, axis=0)
                mean_ap_test = np.mean(aps_test)
                std_ap_test = np.std(aps_test)
                ap_ci_lower = mean_ap_test - 1.96 * (std_ap_test / np.sqrt(n_samples))
                ap_ci_upper = mean_ap_test + 1.96 * (std_ap_test / np.sqrt(n_samples))
                label_test_pr = f'{name} Test (AP = {mean_ap_test:.3f} [{ap_ci_lower:.3f}-{ap_ci_upper:.3f}])'
                line_pr, = ax_pr.plot(mean_recall, mean_precision_test, label=label_test_pr, lw=2, alpha=0.9)
 
                mean_precision_train = np.mean(precisions_train_interp, axis=0)
                mean_ap_train = np.mean(aps_train)
                label_train_pr = f'{name} Train (AP = {mean_ap_train:.3f})'
                ax_pr.plot(mean_recall, mean_precision_train, linestyle=':', color=line_pr.get_color(), label=label_train_pr, lw=2)
  
                model_results = {'Model': name}
                metrics_to_calculate = {
                    'test_roc_auc': (aucs_test, 'AUC'),
                    'test_accuracy': (accuracies_test, 'ACC'),
                    'test_precision': (precisions_test, 'PPV'),
                    'test_recall': (recalls_test, 'Recall'),
                    'test_f1': (f1s_test, 'F1'),
                    'test_tp': (tps_test, 'TP'),
                    'test_tn': (tns_test, 'TN'),
                    'test_fp': (fps_test, 'FP'),
                    'test_fn': (fns_test, 'FN'),
                    'test_npv': (npvs_test, 'NPV'),
                    'test_Kappa': (kappa_test, 'kappa'),
                    'test_Youden': (speci_test, 'Youden'),
                    'test_speci': (youden_test, 'Speci'),
                }
                
                for key, (values, display_name) in metrics_to_calculate.items():
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    n = len(values)
                    std_err = std_val / np.sqrt(n) if n > 0 else 0.0
                    ci_lower = mean_val - 1.96 * std_err
                    ci_upper = mean_val + 1.96 * std_err
                    
                    model_results[f'{display_name} (mean)'] = f"{mean_val:.4f}"
                    model_results[f'{display_name} (95% CI)'] = f"[{ci_lower:.4f}, {ci_upper:.4f}]"
                    model_results[f'{display_name} (std)'] = f"{std_err:.4f}"
            
                results_list.append(model_results)

    ax_roc.set(xlabel='FPR (False Positive Rate)', ylabel='TPR (True Positive Rate)', title='ROC Curve')
    ax_roc.legend(loc='lower right', prop={'size': 12})
    ax_roc.grid(True, linestyle='--', alpha=0.6)
    ax_roc.xaxis.label.set_fontsize(16)
    ax_roc.yaxis.label.set_fontsize(16)
    ax_roc.title.set_fontsize(18)

    ax_pr.set(xlabel='Recall', ylabel='Precision', title='Precision-Recall Curve')
    ax_pr.legend(loc='lower left', prop={'size': 12})
    ax_pr.grid(True, linestyle='--', alpha=0.6)
    ax_pr.xaxis.label.set_fontsize(16)
    ax_pr.yaxis.label.set_fontsize(16)
    ax_pr.title.set_fontsize(18)

    plt.tight_layout()
    fig_roc.savefig(output_path +str(dID) +'_auc曲线.jpg', dpi=1200, bbox_inches='tight')
    fig_pr.savefig(output_path +str(dID) +'_pr曲线.jpg', dpi=1200, bbox_inches='tight')

    columns_ordered = ['Model']
    for display_name in metric_mapping.values():
        columns_ordered.append(f'{display_name} (mean)')
        columns_ordered.append(f'{display_name} (95% CI)')
        columns_ordered.append(f'{display_name} (std)')
    results_df = pd.DataFrame(results_list)[columns_ordered]
    results_df.to_excel(output_path +str(dID) + '_模型评估结果.xlsx', index=False)
    print("模型评估结果已保存到Excel。")

    from sklearn.calibration import calibration_curve as calib_curve
    from sklearn.metrics import brier_score_loss
    plt.figure(figsize=(8, 8))
    plt.title('Calibration Curve', fontsize=18)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray',label= 'Reference')
    plt.xlabel('Predicted Probability', fontsize=16)
    plt.ylabel('True Probability', fontsize=16)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    for name in all_model_names:
        y_true_test = np.array(all_preds[name]['y_true_test'])
        y_pred_test = np.array(all_preds[name]['y_pred_test'])
        prob_true, prob_pred = calib_curve(y_true_test, y_pred_test, n_bins=5)
        bri = brier_score_loss(y_true_test,y_pred_test, pos_label=y.max())
        plt.plot(prob_pred, prob_true, label=f'{name}({bri:.3})')
    plt.legend(loc="lower right",fontsize=12)
    plt.grid(linestyle='--')
    plt.savefig(output_path+str(dID)+'_校准曲线.jpg',dpi=1200, bbox_inches='tight')

    p_value_matrix = pd.DataFrame(index=all_model_names, columns=all_model_names, dtype=float)
    for model1_name, model2_name in itertools.combinations(all_model_names, 2):
        model1_preds = np.array(all_preds[model1_name]['y_pred_test'])
        model2_preds = np.array(all_preds[model2_name]['y_pred_test'])
        y_test = np.array(all_preds[model1_name]['y_true_test'])
        p_value = delong_test(y_test, model1_preds, model2_preds)
        p_value_matrix.loc[model1_name, model2_name] = p_value
        p_value_matrix.loc[model2_name, model1_name] = p_value

    np.fill_diagonal(p_value_matrix.values, 1.0)
    print("\n--- Delong检验P值矩阵 ---")
    print(p_value_matrix.round(4))

    n_comparisons = len(list(itertools.combinations(all_model_names, 2))) # 15次比较
    alpha = 0.05
    bonferroni_alpha = alpha / n_comparisons

    print(f"\n--- Bonferroni校正分析 (α = {alpha}, 比较次数 = {n_comparisons}) ---")
    print(f"校正后的显著性水平 (Bonferroni α) = {bonferroni_alpha:.4f}")
    significance_matrix = p_value_matrix.applymap(lambda p: f"p < {bonferroni_alpha:.4f} (显著)" if p < bonferroni_alpha else f"p = {p:.4f} (不显著)")
    np.fill_diagonal(significance_matrix.values, "-")

    print("\n--- 校正后的显著性判断矩阵 ---")
    print(significance_matrix)

    output_report_path = output_path+str(dID)+"_delong.xlsx"
    try:
        with pd.ExcelWriter(output_report_path, mode='w', engine='openpyxl') as writer:
            p_value_matrix.round(4).to_excel(writer, sheet_name='Delong_Test_P_Values')
            significance_matrix.to_excel(writer, sheet_name='Delong_Test_Significance')
        print(f"\nDelong检验结果已添加到文件: {output_report_path}")
    except Exception as e:
        print(f"\n添加Delong检验结果到Excel失败。原因: {e}")

    result_df = pd.DataFrame()
    for name in all_model_names:
        y_true_test = np.array(all_preds[name]['y_true_test'])
        y_pred_test = np.array(all_preds[name]['y_pred_test'])

        y_true_test_col = f"{name}_y_true_test"
        y_pred_test_col = f"{name}_y_pred_test"
        result_df[y_true_test_col] = y_true_test
        result_df[y_pred_test_col] = y_pred_test

    result_df.to_excel(output_path+str(dID)+'_测试集结果.xlsx', index=False)

    result_df = pd.DataFrame()
    for name in all_model_names:
        y_true_test = np.array(all_preds[name]['y_true_train'])
        y_pred_test = np.array(all_preds[name]['y_pred_train'])
        y_true_test_col = f"{name}_y_true_train"
        y_pred_test_col = f"{name}_y_pred_train"
        result_df[y_true_test_col] = y_true_test
        result_df[y_pred_test_col] = y_pred_test

    result_df.to_excel(output_path+str(dID)+'_训练集结果.xlsx', index=False)

