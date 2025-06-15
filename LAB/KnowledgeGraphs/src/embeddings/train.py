import os
import json
import logging
import pandas as pd
import torch
import itertools

from src.embeddings.helpers import (
    execute_training,  # formerly train_model
    perform_cv,        # formerly cross_validate_model
    default_hyperparameters  # formerly get_default_hyperparams
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def train_and_optimize_kge_models(triples_path, output_dir, model_names=None):
    """
    Train and optimize multiple Knowledge Graph Embedding models with hyperparameter search.

    Args:
        triples_path (str): Path to the TSV file containing triples.
        output_dir (str): Directory to save models and results.
        model_names (list): List of model names to train (default: all supported models).
    """

    os.makedirs(output_dir, exist_ok=True)

    param_grid = {
        'embedding_dim': [256, 64, 128],
        'learning_rate': [0.001],
        'num_negatives': [3,1,5],
        'batch_size': [64],
        'early_stopping': [True],
        'patience': [5],
        'relative_delta': [0.002],
        'num_epochs': [10]
    }
    
    if model_names is None:
        model_names = ['DistMult', 'ComplEx']

    results = {}

    for model_name in model_names:
        logger.info(f"Training {model_name}...")

        default_params = default_hyperparameters(model_name)
        best_score = float('-inf')
        best_params = None

        param_combinations = [
            dict(zip(param_grid.keys(), values))
            for values in itertools.product(*param_grid.values())
        ]

        for i, params in enumerate(param_combinations, 1):
            current_params = default_params.copy()
            current_params.update(params)

            logger.info(f"[{i}/{len(param_combinations)}] Testing parameters: {current_params}")

            metrics = perform_cv(
                path_to_triples=triples_path,
                model_type=model_name,
                parameters=current_params,
                splits=2
            )

            logger.info("Metrics:")
            for metric_name, metric_value in metrics.items():
                logger.info(f"  {metric_name}: {metric_value:.4f}")

            score = metrics.get('inverse_harmonic_mean_rank', 0)
            if score > best_score:
                best_score = score
                best_params = current_params

        logger.info(f"Training final {model_name} model with best parameters...")
        model, entity_to_id, relation_to_id, test_triples, final_metrics = execute_training(
            path_to_triples=triples_path,
            model_type=model_name,
            parameters=best_params
        )

        model_dir = os.path.join(output_dir, f"{model_name.lower()}_model")
        os.makedirs(model_dir, exist_ok=True)

        torch.save(model, os.path.join(model_dir, "trained_model.pkl"))

        entity_df = pd.DataFrame({
            'id': list(entity_to_id.values()),
            'entity': list(entity_to_id.keys())
        })
        entity_df.to_csv(os.path.join(model_dir, "entity_to_id.tsv.gz"), sep='\t', index=False, compression='gzip')

        relation_df = pd.DataFrame({
            'id': list(relation_to_id.values()),
            'relation': list(relation_to_id.keys())
        })
        relation_df.to_csv(os.path.join(model_dir, "relation_to_id.tsv.gz"), sep='\t', index=False, compression='gzip')

        results[model_name] = {
            'best_parameters': best_params,
            'metrics': final_metrics
        }

        with open(os.path.join(model_dir, 'results.json'), 'w') as f:
            json.dump(results[model_name], f, indent=2)

    # Build summary dataframe with variance and embedding dimension
    summary_data = []
    for model_name, result in results.items():
        metrics = result['metrics']
        best_params = result['best_parameters']
        summary_data.append({
            'Model': model_name,
            'MRR': metrics.get('inverse_harmonic_mean_rank', 0),
            'Hits@1': metrics.get('hits_at_1', 0),
            'Hits@5': metrics.get('hits_at_5', 0),
            'Hits@10': metrics.get('hits_at_10', 0),
            'Variance': metrics.get('variance', 0),
            'Embedding_Dim': best_params.get('embedding_dim', 0)
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)

    logger.info("\nModel Comparison Summary:")
    logger.info("\n" + summary_df.to_string(index=False))
    
def summarize_existing_models(output_dir: str):
    """
    Load results from all trained models in the output directory and generate a summary table.
    
    Args:
        output_dir (str): Directory where model folders with 'results.json' are stored.
    """
    summary_data = []

    for model_folder in os.listdir(output_dir):
        model_dir = os.path.join(output_dir, model_folder)
        results_path = os.path.join(model_dir, 'results.json')

        if not os.path.isfile(results_path):
            logger.warning(f"No results.json found in {model_dir}, skipping.")
            continue

        with open(results_path, 'r') as f:
            result = json.load(f)

        metrics = result.get('metrics', {})
        best_params = result.get('best_parameters', {})

        summary_data.append({
            'Model': model_folder.replace('_model', '').capitalize(),
            'MRR': metrics.get('inverse_harmonic_mean_rank', 0),
            'Hits@1': metrics.get('hits_at_1', 0),
            'Hits@5': metrics.get('hits_at_5', 0),
            'Hits@10': metrics.get('hits_at_10', 0),
            'Variance': metrics.get('variance', 0),
            'Embedding_Dim': best_params.get('embedding_dim', 0)
        })

    if not summary_data:
        logger.info("No model results found to summarize.")
        return

    summary_df = pd.DataFrame(summary_data)
    summary_df.sort_values(by='MRR', ascending=False, inplace=True)
    
    summary_path = os.path.join(output_dir, 'model_comparison.csv')
    summary_df.to_csv(summary_path, index=False)

    logger.info("\nSummary of Existing Models:")
    logger.info("\n" + summary_df.to_string(index=False))



def main():
    triples_path = "abox_export.tsv"
    output_dir = "models"
    #train_and_optimize_kge_models(triples_path, output_dir)
    summarize_existing_models(output_dir)
    
if __name__ == "__main__":
    main()
