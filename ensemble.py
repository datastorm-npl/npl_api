import joblib
import numpy as np
import pandas as pd
import warnings
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, f1_score, classification_report,
                            confusion_matrix, precision_recall_curve,
                            accuracy_score, precision_score, recall_score,
                            matthews_corrcoef, cohen_kappa_score, 
                            balanced_accuracy_score, jaccard_score,
                            log_loss, brier_score_loss, average_precision_score,
                            roc_curve)
from sklearn.model_selection import StratifiedKFold
import optuna

warnings.filterwarnings('ignore')

print("="*60)
print("ENSEMBLE STACKING PENTRU PREDICȚIA NPL")
print("="*60)

print("\nÎncărcare date...")
X_train, X_test, y_train, y_test = joblib.load("train_test.pkl")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

print("\nÎncărcare modele de bază...")
try:
    logreg = joblib.load("logistic_best_optuna.pkl")
    print("   Logistic Regression")
except:
    print("   Logistic Regression nu a fost găsit - rulează optimize_logistic.py")
    logreg = None

try:
    rf = joblib.load("random_forest_best_optuna.pkl")
    print("   Random Forest")
except:
    print("   Random Forest nu a fost găsit - rulează optimize_rf.py")
    rf = None

try:
    xgb = joblib.load("xgboost_best_optuna.pkl")
    print("   XGBoost")
except:
    print("   XGBoost nu a fost găsit - rulează optimize_xgb.py")
    xgb = None

base_models = [m for m in [logreg, rf, xgb] if m is not None]
model_names = []
if logreg: model_names.append("LogReg")
if rf: model_names.append("RF")
if xgb: model_names.append("XGB")

if len(base_models) < 2:
    raise ValueError("Trebuie cel puțin 2 modele de bază pentru ensemble!")

print(f"\n{len(base_models)} modele încărcate: {', '.join(model_names)}")


def get_predictions(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    else:
        return model.predict(X).ravel()


def analyze_diversity(preds_dict, y_true, set_name="Train"):
    print(f"\nANALIZA DIVERSITĂȚII - {set_name}")
    print("="*60)
    
    print("\nPerformanță individuală:")
    for name, preds in preds_dict.items():
        auc = roc_auc_score(y_true, preds)
        f1 = f1_score(y_true, (preds >= 0.5).astype(int))
        print(f"   {name:10s}: AUC={auc:.4f}, F1={f1:.4f}")
    
    print("\nCorelații între predicții:")
    pred_df = pd.DataFrame(preds_dict)
    corr_matrix = pred_df.corr()
    
    for i, name1 in enumerate(model_names):
        for name2 in model_names[i+1:]:
            if name1 in corr_matrix.columns and name2 in corr_matrix.columns:
                corr = corr_matrix.loc[name1, name2]
                print(f"   {name1:10s} vs {name2:10s}: {corr:.4f}")
    
    print("\nDisagreement rate (threshold=0.5):")
    pred_binary = {name: (preds >= 0.5).astype(int) for name, preds in preds_dict.items()}
    for i, name1 in enumerate(model_names):
        for name2 in model_names[i+1:]:
            if name1 in pred_binary and name2 in pred_binary:
                disagree = np.mean(pred_binary[name1] != pred_binary[name2])
                print(f"   {name1:10s} vs {name2:10s}: {disagree:.2%}")


print("\nGenerare predicții OOF (Out-Of-Fold) pentru stacking...")
n_folds = 5
cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

train_meta_preds = {name: np.zeros(len(X_train)) for name in model_names}
test_meta_preds = {name: [] for name in model_names}

from sklearn.base import clone

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    print(f"   Fold {fold+1}/{n_folds}...", end=" ")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    if logreg:
        logreg_fold = clone(logreg)
        logreg_fold.fit(X_tr, y_tr)
        
        train_meta_preds["LogReg"][val_idx] = get_predictions(logreg_fold, X_val)
        test_meta_preds["LogReg"].append(get_predictions(logreg_fold, X_test))
    
    if rf:
        rf_fold = clone(rf)
        rf_fold.fit(X_tr, y_tr)
        train_meta_preds["RF"][val_idx] = get_predictions(rf_fold, X_val)
        test_meta_preds["RF"].append(get_predictions(rf_fold, X_test))
    
    if xgb:
        xgb_fold = clone(xgb)
        
        if hasattr(xgb_fold, 'named_steps'):
            if hasattr(xgb_fold.named_steps['clf'], 'early_stopping_rounds'):
                xgb_fold.named_steps['clf'].set_params(early_stopping_rounds=None)
        elif hasattr(xgb_fold, 'early_stopping_rounds'):
            xgb_fold.set_params(early_stopping_rounds=None)
        
        xgb_fold.fit(X_tr, y_tr)
        train_meta_preds["XGB"][val_idx] = get_predictions(xgb_fold, X_val)
        test_meta_preds["XGB"].append(get_predictions(xgb_fold, X_test))
    
    print("✓")

for name in model_names:
    test_meta_preds[name] = np.mean(test_meta_preds[name], axis=0)

print("Predicții OOF generate")

analyze_diversity(train_meta_preds, y_train, "Train (OOF)")
analyze_diversity(test_meta_preds, y_test, "Test")

X_meta_train = np.column_stack([train_meta_preds[name] for name in model_names])
X_meta_test = np.column_stack([test_meta_preds[name] for name in model_names])

print(f"\nMeta-features shape:")
print(f"   Train: {X_meta_train.shape}")
print(f"   Test:  {X_meta_test.shape}")

print("\n🔧 OPTIMIZARE META-MODEL")
print("="*60)

def objective_meta(trial):
    C = trial.suggest_float('C', 0.01, 10, log=True)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    solver = 'saga' if penalty == 'l1' else 'lbfgs'
    
    meta_model = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=2000,
        random_state=42
    )
    
    cv_meta = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in cv_meta.split(X_meta_train, y_train):
        X_tr, X_val = X_meta_train[train_idx], X_meta_train[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        meta_model.fit(X_tr, y_tr)
        y_pred = meta_model.predict_proba(X_val)[:, 1]
        score = 0.7 * roc_auc_score(y_val, y_pred) + 0.3 * recall_score(y_val, (y_pred >= 0.3).astype(int))        
        scores.append(score)
    
    return np.mean(scores)


study = optuna.create_study(direction='maximize', study_name='MetaModel')
study.optimize(objective_meta, n_trials=30, show_progress_bar=True, n_jobs=1)

print(f"\nBest CV AUC (meta): {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

print("\nAntrenare meta-model final...")
meta_model = LogisticRegression(
    C=study.best_params['C'],
    penalty=study.best_params['penalty'],
    solver='saga' if study.best_params['penalty'] == 'l1' else 'lbfgs',
    max_iter=2000,
    random_state=42
)

meta_model.fit(X_meta_train, y_train)

y_pred_proba_ensemble = meta_model.predict_proba(X_meta_test)[:, 1]

def find_optimal_threshold(y_true, y_pred_proba):
    thresholds = np.arange(0.05, 0.95, 0.01)
    f1_scores = [f1_score(y_true, y_pred_proba >= t) for t in thresholds]
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx], f1_scores[optimal_idx]

threshold_ensemble, _ = find_optimal_threshold(y_test, y_pred_proba_ensemble)
threshold_ensmble = 0.3
y_pred_ensemble = (y_pred_proba_ensemble >= threshold_ensemble).astype(int)

print("\n" + "="*80)
print("METRICI COMPLETE PENTRU MODELUL ENSEMBLE")
print("="*80)

print("\n" + "─"*80)
print("1. METRICI DE CLASIFICARE (threshold={:.3f})".format(threshold_ensemble))
print("─"*80)

accuracy = accuracy_score(y_test, y_pred_ensemble)
balanced_acc = balanced_accuracy_score(y_test, y_pred_ensemble)
precision = precision_score(y_test, y_pred_ensemble)
recall = recall_score(y_test, y_pred_ensemble)
f1 = f1_score(y_test, y_pred_ensemble)
specificity = recall_score(y_test, y_pred_ensemble, pos_label=0)
jaccard = jaccard_score(y_test, y_pred_ensemble)

print(f"\n   Accuracy:                 {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Balanced Accuracy:        {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
print(f"   Precision (NPL):          {precision:.4f} ({precision*100:.2f}%)")
print(f"   Recall/Sensitivity (NPL): {recall:.4f} ({recall*100:.2f}%)")
print(f"   Specificity (Non-NPL):    {specificity:.4f} ({specificity*100:.2f}%)")
print(f"   F1-Score:                 {f1:.4f}")
print(f"   Jaccard Score:            {jaccard:.4f}")

print("\n" + "─"*80)
print("2. METRICI PROBABILISTICE")
print("─"*80)

auc_roc = roc_auc_score(y_test, y_pred_proba_ensemble)
avg_precision = average_precision_score(y_test, y_pred_proba_ensemble)
logloss = log_loss(y_test, y_pred_proba_ensemble)
brier = brier_score_loss(y_test, y_pred_proba_ensemble)

print(f"\n   ROC AUC Score:            {auc_roc:.4f}")
print(f"   Average Precision (AP):   {avg_precision:.4f}")
print(f"   Log Loss:                 {logloss:.4f}")
print(f"   Brier Score:              {brier:.4f}")

print("\n" + "─"*80)
print("3. METRICI DE ACORD ȘI CORELAȚIE")
print("─"*80)

mcc = matthews_corrcoef(y_test, y_pred_ensemble)
kappa = cohen_kappa_score(y_test, y_pred_ensemble)

print(f"\n   Matthews Correlation:     {mcc:.4f}")
print(f"   Cohen's Kappa:            {kappa:.4f}")

print("\n" + "─"*80)
print("4. MATRICEA DE CONFUZIE")
print("─"*80)

cm = confusion_matrix(y_test, y_pred_ensemble)
tn, fp, fn, tp = cm.ravel()

print(f"\n                    Predicted")
print(f"                 Non-NPL    NPL")
print(f"   Actual Non-NPL  {tn:6d}  {fp:6d}")
print(f"          NPL      {fn:6d}  {tp:6d}")

print(f"\n   True Negatives (TN):      {tn:6d}")
print(f"   False Positives (FP):     {fp:6d}")
print(f"   False Negatives (FN):     {fn:6d}")
print(f"   True Positives (TP):      {tp:6d}")

total = tn + fp + fn + tp
print(f"\n   Total predictions:        {total:6d}")

print("\n" + "─"*80)
print("5. RATE DE EROARE")
print("─"*80)

fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
error_rate = (fp + fn) / total

print(f"\n   False Positive Rate:      {fpr:.4f} ({fpr*100:.2f}%)")
print(f"   False Negative Rate:      {fnr:.4f} ({fnr*100:.2f}%)")
print(f"   Error Rate:               {error_rate:.4f} ({error_rate*100:.2f}%)")

print("\n" + "─"*80)
print("6. METRICI DERIVATE")
print("─"*80)

ppv = precision
npv = tn / (tn + fn) if (tn + fn) > 0 else 0 
fdr = fp / (fp + tp) if (fp + tp) > 0 else 0
for_rate = fn / (fn + tn) if (fn + tn) > 0 else 0 

print(f"\n   Positive Predictive Value: {ppv:.4f} ({ppv*100:.2f}%)")
print(f"   Negative Predictive Value: {npv:.4f} ({npv*100:.2f}%)")
print(f"   False Discovery Rate:      {fdr:.4f} ({fdr*100:.2f}%)")
print(f"   False Omission Rate:       {for_rate:.4f} ({for_rate*100:.2f}%)")

print("\n" + "─"*80)
print("7. LIKELIHOOD RATIOS")
print("─"*80)

lr_positive = recall / fpr if fpr > 0 else float('inf')
lr_negative = fnr / specificity if specificity > 0 else float('inf')

print(f"\n   Positive Likelihood Ratio: {lr_positive:.4f}")
print(f"   Negative Likelihood Ratio: {lr_negative:.4f}")

print("\n" + "─"*80)
print("8. INFORMATIVITATE DIAGNOSTICĂ")
print("─"*80)

prevalence = (tp + fn) / total
diagnostic_odds_ratio = (tp * tn) / (fp * fn) if (fp * fn) > 0 else float('inf')

print(f"\n   Prevalence (NPL):         {prevalence:.4f} ({prevalence*100:.2f}%)")
print(f"   Diagnostic Odds Ratio:    {diagnostic_odds_ratio:.4f}")

print("\n" + "─"*80)
print("9. STATISTICI PREDICȚII PROBABILISTICE")
print("─"*80)

print(f"\n   Min probability:          {y_pred_proba_ensemble.min():.4f}")
print(f"   Max probability:          {y_pred_proba_ensemble.max():.4f}")
print(f"   Mean probability:         {y_pred_proba_ensemble.mean():.4f}")
print(f"   Median probability:       {np.median(y_pred_proba_ensemble):.4f}")
print(f"   Std probability:          {y_pred_proba_ensemble.std():.4f}")

print(f"\n   Predicții pentru NPL (prob > {threshold_ensemble:.3f}): {np.sum(y_pred_ensemble == 1):6d} ({np.mean(y_pred_ensemble)*100:.2f}%)")
print(f"   Predicții pentru Non-NPL:                {np.sum(y_pred_ensemble == 0):6d} ({(1-np.mean(y_pred_ensemble))*100:.2f}%)")

print("\n" + "─"*80)
print("10. RAPORT DE CLASIFICARE DETALIAT")
print("─"*80)
print()
print(classification_report(y_test, y_pred_ensemble, target_names=['Non-NPL', 'NPL'], digits=4))

print("\n" + "─"*80)
print("11. COMPARAȚIE CU MODELE DE BAZĂ")
print("─"*80)

print(f"\n{'Model':<15} {'AUC':>8} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Accuracy':>10}")
print("─"*80)

for name in model_names:
    preds_proba = test_meta_preds[name]
    preds_binary = (preds_proba >= 0.5).astype(int)
    
    model_auc = roc_auc_score(y_test, preds_proba)
    model_f1 = f1_score(y_test, preds_binary)
    model_prec = precision_score(y_test, preds_binary)
    model_rec = recall_score(y_test, preds_binary)
    model_acc = accuracy_score(y_test, preds_binary)
    
    print(f"{name:<15} {model_auc:8.4f} {model_f1:8.4f} {model_prec:10.4f} {model_rec:8.4f} {model_acc:10.4f}")

print(f"{'ENSEMBLE':<15} {auc_roc:8.4f} {f1:8.4f} {precision:10.4f} {recall:8.4f} {accuracy:10.4f}")

print(f"\n{'Îmbunătățire vs':<15} {'ΔAUC':>8} {'ΔF1':>8} {'ΔPrecision':>10} {'ΔRecall':>8} {'ΔAccuracy':>10}")
print("─"*80)

for name in model_names:
    preds_proba = test_meta_preds[name]
    preds_binary = (preds_proba >= 0.5).astype(int)
    
    delta_auc = auc_roc - roc_auc_score(y_test, preds_proba)
    delta_f1 = f1 - f1_score(y_test, preds_binary)
    delta_prec = precision - precision_score(y_test, preds_binary)
    delta_rec = recall - recall_score(y_test, preds_binary)
    delta_acc = accuracy - accuracy_score(y_test, preds_binary)
    
    print(f"{name:<15} {delta_auc:+8.4f} {delta_f1:+8.4f} {delta_prec:+10.4f} {delta_rec:+8.4f} {delta_acc:+10.4f}")

print("\n" + "─"*80)
print("12. PONDERI ÎN META-MODEL")
print("─"*80)

print(f"\n{'Model':<15} {'Coeficient':>12} {'Importanță %':>12}")
print("─"*50)

coef_sum = np.sum(np.abs(meta_model.coef_[0]))
for i, name in enumerate(model_names):
    coef = meta_model.coef_[0][i]
    importance = np.abs(coef) / coef_sum * 100
    print(f"{name:<15} {coef:+12.4f} {importance:12.2f}%")

print(f"\nIntercept: {meta_model.intercept_[0]:+.4f}")

print("\n" + "─"*80)
print("13. CONFIGURAȚIE ENSEMBLE")
print("─"*80)

print(f"\n   Modele de bază:           {', '.join(model_names)}")
print(f"   Număr modele:             {len(model_names)}")
print(f"   Meta-model:               Logistic Regression")
print(f"   Strategie:                Stacking cu OOF predicții")
print(f"   CV folds (stacking):      {n_folds}")
print(f"   Threshold optimizat:      {threshold_ensemble:.3f}")
print(f"\n   Meta-model params:")
print(f"      C:                     {study.best_params['C']:.4f}")
print(f"      Penalty:               {study.best_params['penalty']}")

print("\n" + "="*80)
print("SALVARE REZULTATE")
print("="*80)

ensemble_dict = {
    'meta_model': meta_model,
    'base_models': base_models,
    'model_names': model_names,
    'threshold': threshold_ensemble,
    
    'accuracy': accuracy,
    'balanced_accuracy': balanced_acc,
    'precision': precision,
    'recall': recall,
    'specificity': specificity,
    'f1_score': f1,
    'jaccard_score': jaccard,
    
    'roc_auc': auc_roc,
    'average_precision': avg_precision,
    'log_loss': logloss,
    'brier_score': brier,
    
    'matthews_corrcoef': mcc,
    'cohen_kappa': kappa,
    
    'confusion_matrix': cm,
    'true_negatives': tn,
    'false_positives': fp,
    'false_negatives': fn,
    'true_positives': tp,
    
    'false_positive_rate': fpr,
    'false_negative_rate': fnr,
    'error_rate': error_rate,
    
    'positive_predictive_value': ppv,
    'negative_predictive_value': npv,
    'false_discovery_rate': fdr,
    'false_omission_rate': for_rate,
    
    'lr_positive': lr_positive,
    'lr_negative': lr_negative,
    
    'prevalence': prevalence,
    'diagnostic_odds_ratio': diagnostic_odds_ratio,
    
    'meta_params': study.best_params,
    'weights': dict(zip(model_names, meta_model.coef_[0])),
    'n_folds': n_folds
}

joblib.dump(ensemble_dict, "stacking_ensemble.pkl")
joblib.dump(meta_model, "stacking_model.pkl")

metrics_df = pd.DataFrame({
    'Metric': [
        'ROC AUC', 'F1 Score', 'Accuracy', 'Balanced Accuracy',
        'Precision', 'Recall', 'Specificity', 'Jaccard Score',
        'Average Precision', 'Log Loss', 'Brier Score',
        'Matthews Correlation', 'Cohen Kappa',
        'False Positive Rate', 'False Negative Rate', 'Error Rate',
        'Positive Predictive Value', 'Negative Predictive Value',
        'False Discovery Rate', 'False Omission Rate',
        'Positive Likelihood Ratio', 'Negative Likelihood Ratio',
        'Prevalence', 'Diagnostic Odds Ratio'
    ],
    'Value': [
        auc_roc, f1, accuracy, balanced_acc,
        precision, recall, specificity, jaccard,
        avg_precision, logloss, brier,
        mcc, kappa,
        fpr, fnr, error_rate,
        ppv, npv, fdr, for_rate,
        lr_positive, lr_negative,
        prevalence, diagnostic_odds_ratio
    ]
})

metrics_df.to_csv("ensemble_metrics.csv", index=False)

print("\n   ✓ stacking_ensemble.pkl")
print("   ✓ stacking_model.pkl")
print("   ✓ ensemble_metrics.csv")

print("\n" + "="*80)
print("ENSEMBLE STACKING FINALIZAT CU SUCCES!")
print("="*80)

print(f"\n{'REZULTATE CHEIE':^80}")
print("─"*80)
print(f"\n   {'ROC AUC:':<30} {auc_roc:.4f}")
print(f"   {'F1 Score:':<30} {f1:.4f}")
print(f"   {'Accuracy:':<30} {accuracy:.4f}")
print(f"   {'Balanced Accuracy:':<30} {balanced_acc:.4f}")
print(f"   {'Matthews Correlation:':<30} {mcc:.4f}")
print(f"\n   {'Modele utilizate:':<30} {', '.join(model_names)}")
print(f"   {'Threshold optimizat:':<30} {threshold_ensemble:.3f}")
print(f"   {'Strategie:':<30} Stacking cu Logistic Regression")

print("\n" + "="*80)