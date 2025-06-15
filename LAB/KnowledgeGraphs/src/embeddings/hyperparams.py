import os
import json

def load_best_hyperparams(model_dir):
    results_path = os.path.join(model_dir, 'results.json')
    
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"No results.json found in {model_dir}")

    with open(results_path, 'r') as f:
        results = json.load(f)

    best_params = results.get('best_parameters', {})
    return best_params

def main():
    base_dir = "models"
    transh_dir = os.path.join(base_dir, "transh_model")

    try:
        best_hyperparams = load_best_hyperparams(transh_dir)
        print("\nBest Hyperparameters for TransH:")
        for key, value in best_hyperparams.items():
            print(f"  {key}: {value}")
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    main()
