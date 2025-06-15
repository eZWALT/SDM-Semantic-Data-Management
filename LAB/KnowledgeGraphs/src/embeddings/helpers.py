import torch
import pandas as pd
import os
import logging
import warnings
import numpy as np
from sklearn.model_selection import KFold
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.losses import MarginRankingLoss
import hashlib

# Suppress PyKEEN and torch logs
warnings.filterwarnings("ignore")
logging.getLogger("pykeen").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

def resolve_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def restore_model(model_directory):
    model = torch.load(os.path.join(model_directory, "trained_model.pkl"), weights_only=False)
    entity_frame = pd.read_csv(os.path.join(model_directory, "entity_to_id.tsv.gz"), sep='\t', compression='gzip')
    relation_frame = pd.read_csv(os.path.join(model_directory, "relation_to_id.tsv.gz"), sep='\t', compression='gzip')
    return model, dict(zip(entity_frame['entity'], entity_frame['id'])), dict(zip(relation_frame['relation'], relation_frame['id']))

def summarize_metrics(results_dict, embed_dim):
    realistic = results_dict.get('both', {}).get('realistic', {})
    hits_values = [
        realistic.get('hits_at_1', 0.0),
        realistic.get('hits_at_5', 0.0),
        realistic.get('hits_at_10', 0.0)
    ]
    realistic['variance'] = realistic.get("variance", 0.0)
    realistic['inverse_harmonic_mean_rank'] = realistic.get("inverse_harmonic_mean_rank", 0.0)
    realistic['embedding_dim'] = embed_dim
    return realistic

def execute_training(path_to_triples, model_type, parameters, seed=42):
    triples_df = pd.read_csv(path_to_triples, sep='\t', header=None, names=['head', 'relation', 'tail'])
    factory = TriplesFactory.from_labeled_triples(triples_df.values, create_inverse_triples=True)
    train_tf, test_tf = factory.split()

    loss_params = {'loss': MarginRankingLoss(margin=parameters.get('margin', 1.0))}
    if parameters.get('model_kwargs'):
        loss_params.update(parameters['model_kwargs'])

    training_params = {
        'num_epochs': parameters.get('num_epochs', 10),
        'batch_size': parameters.get('batch_size', 64)
    }

    result = pipeline(
        training=train_tf,
        testing=test_tf,
        model=model_type,
        model_kwargs=loss_params,
        training_kwargs=training_params,
        optimizer='Adam',
        optimizer_kwargs={'lr': parameters.get('learning_rate', 0.01)},
        negative_sampler='basic',
        negative_sampler_kwargs={'num_negs_per_pos': parameters.get('num_negatives', 3)},
        device=resolve_device(),
        random_seed=seed,
        use_tqdm=False,
        evaluator='rankbased',
        evaluator_kwargs={'filtered': True, 'metrics': ['inverse_harmonic_mean_rank', 'hits_at_k']},
        evaluation_kwargs={'use_tqdm': False}
    )

    return result.model, factory.entity_to_id, factory.relation_to_id, test_tf, summarize_metrics(result.metric_results.to_dict(), parameters.get('embedding_dim', 0))

def perform_cv(path_to_triples, model_type, parameters, splits=3, seed=42):
    df = pd.read_csv(path_to_triples, sep='\t', header=None, names=['head', 'relation', 'tail'])
    factory = TriplesFactory.from_labeled_triples(df.values, create_inverse_triples=True)
    kf = KFold(n_splits=splits, shuffle=True, random_state=seed)

    scores = {'inverse_harmonic_mean_rank': [], 'hits_at_1': [], 'hits_at_5': [], 'hits_at_10': []}

    for idx, (train_idx, test_idx) in enumerate(kf.split(factory.triples)):
        print(f"\nRunning fold {idx + 1} of {splits}...")

        train_tf = TriplesFactory.from_labeled_triples(
            triples=factory.triples[train_idx],
            entity_to_id=factory.entity_to_id,
            relation_to_id=factory.relation_to_id,
            create_inverse_triples=True
        )
        test_tf = TriplesFactory.from_labeled_triples(
            triples=factory.triples[test_idx],
            entity_to_id=factory.entity_to_id,
            relation_to_id=factory.relation_to_id,
            create_inverse_triples=True
        )

        result = pipeline(
            training=train_tf,
            testing=test_tf,
            model=model_type,
            model_kwargs=parameters.get('model_kwargs', {}),
            training_kwargs=parameters.get('training_kwargs', {}),
            optimizer='Adam',
            optimizer_kwargs={'lr': parameters.get('learning_rate', 0.001)},
            device=resolve_device(),
            random_seed=seed + idx,
            evaluator='rankbased',
            evaluator_kwargs={'filtered': True, 'metrics': ['inverse_harmonic_mean_rank', 'hits_at_k']},
            evaluation_kwargs={'use_tqdm': False},
            use_tqdm=False
        )

        metrics = summarize_metrics(result.metric_results.to_dict(), parameters.get('embedding_dim', 0))

        for k in ['inverse_harmonic_mean_rank', 'hits_at_1', 'hits_at_5', 'hits_at_10']:
            scores[k].append(metrics.get(k, 0.0))

    avg_scores = {k: float(np.mean(v)) for k, v in scores.items()}
    return avg_scores

def default_hyperparameters(model_type):
    shared = {
        'embedding_dim': 128,
        'num_epochs': 10,
        'batch_size': 64,
        'num_negatives': 3,
        'learning_rate': 0.01,
        'early_stopping': True,
        'patience': 5,
        'relative_delta': 0.002
    }

    mods = {
        'TransH': {'scoring_fct_norm': 2},
        'RotatE': {'scoring_fct_norm': 2},
        'DistMult': {'regularizer_weight': 0.1},
        'ComplEx': {'regularizer_weight': 0.1}
    }

    return {**shared, **mods.get(model_type, {})}
