import os
import pandas as pd

RESULTS_PATH = "reports/experiment_results.csv"

def save_experiment_result(model_name, results):
    os.makedirs("reports", exist_ok=True)
    
    row={
        "model": model_name,
        "accuracy": results["accuracy"],
        "precision": results["precision"],
        "recall": results["recall"],
        "f1": results["f1"],
        "roc_auc": results["roc_auc"],
        "pr_auc": results["pr_auc"],
        "training_time": results["training_time"],
        "prediction_time": results["prediction_time"],
    }
    
    new_result=pd.DataFrame([row])
    
    if os.path.exists(RESULTS_PATH):
        existing_results=pd.read_csv(RESULTS_PATH)
        
        # Replace an existing result for the same model
        existing_results = existing_results[
            existing_results["model"] != model_name
        ]
        
        results_df = pd.concat(
            [existing_results, new_result],
            ignore_index=True,
        )
        
    else:
        results_df=new_result
        
    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )
    
    print(f"\nExperiment results saved to: {RESULTS_PATH}")
    
    
def load_experiment_results():
    if not os.path.exists(RESULTS_PATH):
        return pd.DataFrame()
    
    return pd.read_csv(RESULTS_PATH)


        