from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent  # models/

MLFLOW_TRACKING_URI = (BASE_DIR / "mlruns")
EXPERIMENT_NAME = "youtube-viral"

def setup_mlflow():
    print(BASE_DIR)
    (BASE_DIR / "mlruns").mkdir(exist_ok=True)  # ensure folder exists
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
def run_experiment(run_name, model, X_tr, y_tr, X_ev, y_ev, params=None, tags=None, skip_fit = False):

    if not skip_fit:
        model.fit(X_tr, y_tr)
    if hasattr(model, 'best_params_'):
        best_params = model.best_params_
        best_score = model.best_score_
        print(f"Best Params: {best_params}, Best CV Score: {best_score:.4f}")
    else:
        best_params = params or {}
        best_score = None
    y_pred = model.predict(X_ev)
    # predict probability if supported
    y_prob = model.predict_proba(X_ev)[:, 1] if hasattr(model, 'predict_proba') else None

    metrics = {
        'accuracy' : accuracy_score(y_ev, y_pred),
        'f1'       : f1_score(y_ev, y_pred, zero_division=0, pos_label=1, average='macro'),
        'precision': precision_score(y_ev, y_pred, zero_division=0, pos_label=1),
        'recall'   : recall_score(y_ev, y_pred, zero_division=0, pos_label=1),
        'roc_auc'  : roc_auc_score(y_ev, y_prob) if y_prob is not None else None,
    }
    tn, fp, fn, tp = confusion_matrix(y_ev, y_pred).ravel()
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    metrics['Wasted_Resources_False_Positive'] = fp / (fp + tp) if (fp + tp) > 0 else None ## costly for marketers
    metrics['Missed_Trending_Videos'] = fn / (fn + tp) if (fn + tp) > 0 else None ## Wasted opportunity

    print(f'  \n{run_name}')
    for k, v in metrics.items():
        if v is not None:
            print(f'  {k}: {v:.4f}')
    print()
    print(classification_report(y_ev, y_pred))

    fig, ax = plt.subplots(figsize=(4, 3))
    ConfusionMatrixDisplay(confusion_matrix(y_ev, y_pred)).plot(ax=ax, colorbar=False)
    ax.set_title(run_name, fontsize=10)
    plt.tight_layout()
    plt.show()

    # track experiments
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(tags or {})
        mlflow.log_params(best_params or {})
        if best_score is not None:
            mlflow.log_metric('best_cv_score', best_score)
        mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
        mlflow.sklearn.log_model(model, 'model')
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ConfusionMatrixDisplay(confusion_matrix(y_ev, y_pred)).plot(ax=ax2, colorbar=False)
        fig2.savefig('_cm_tmp.png', dpi=100, bbox_inches='tight') # save confusion matrix plot
        mlflow.log_artifact('_cm_tmp.png', artifact_path='plots') # save in mlflow
        plt.close(fig2)

    return metrics, model
