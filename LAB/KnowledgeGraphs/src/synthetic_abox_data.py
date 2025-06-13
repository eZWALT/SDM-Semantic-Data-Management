# This script generates synthetic proceedings data from the given paper and edition edges.
# We generate a CSV treating proceedings as nodes.
# We generate a CSV for edges between editions and proceedings,
# and another for edges between proceedings and papers.

import pandas as pd
import numpy as np
import random
from pathlib import Path

input_dir = Path('./abox_data_raw')
output_dir = Path('./abox_data_raw')

# 1. Proceedings nodes and edges

# Load paper-edition edges
paper_edition = pd.read_csv(input_dir / 'edges_paper_published_in_edition.csv')

# Unique editions and papers
unique_editions = sorted(paper_edition['edition_id'].unique())
unique_papers = paper_edition['paper_id'].unique()

# Map each edition to a unique proceedings_id
edition_to_proceedings = {
    edition: f"proceedings_{str(i+1).zfill(3)}"
    for i, edition in enumerate(unique_editions)
}

# nodes_proceedings.csv
proceedings_df = pd.DataFrame(
    {'proceedings_id': list(edition_to_proceedings.values())})
proceedings_df.to_csv(output_dir / 'nodes_proceedings.csv', index=False)

# edges_edition_has_proceedings.csv
edition_proceedings_edges = pd.DataFrame([
    {'edition_id': edition, 'proceedings_id': proceedings}
    for edition, proceedings in edition_to_proceedings.items()
])
edition_proceedings_edges.to_csv(
    output_dir / 'edges_edition_has_proceedings.csv', index=False)

# edges_proceedings_includes_paper.csv
proceedings_paper_edges = paper_edition.copy()
proceedings_paper_edges['proceedings_id'] = proceedings_paper_edges['edition_id'].map(
    edition_to_proceedings)
proceedings_paper_edges = proceedings_paper_edges[[
    'proceedings_id', 'paper_id']]
proceedings_paper_edges.to_csv(
    output_dir / 'edges_proceedings_includes_paper.csv', index=False)

# 2. Add 30 extra editions to nodes_edition.csv (for missing Workshop events)
editions_path = input_dir / 'nodes_edition.csv'
editions_df = pd.read_csv(editions_path)

years = editions_df['year'].dropna().unique()
locations = editions_df['location'].dropna().unique()

new_editions = []
for i in range(1, 31):
    new_editions.append({
        'edition_id': f'Edition{i}',
        'year': random.choice(years),
        'location': random.choice(locations)
    })

for col in editions_df.columns:
    if col not in ['edition_id', 'year', 'location']:
        for entry in new_editions:
            entry[col] = random.choice(editions_df[col].dropna(
            ).unique()) if not editions_df[col].dropna().empty else ''

new_editions_df = pd.DataFrame(new_editions)
editions_df = pd.concat([editions_df, new_editions_df], ignore_index=True)
editions_df.to_csv(output_dir / 'nodes_edition.csv', index=False)

# 3. nodes_workshop.csv (we create 50 workshops)
conferences_path = input_dir / 'nodes_conference.csv'
conferences_df = pd.read_csv(conferences_path)
themes = conferences_df['theme'].dropna().unique()

workshops = []
for i in range(1, 51):
    workshops.append({
        'workshop_id': f'Workshop{i}',
        'theme': random.choice(themes)
    })
workshops_df = pd.DataFrame(workshops)
workshops_df.to_csv(output_dir / 'nodes_workshop.csv', index=False)

# 4. edges_edition_held_for_workshop.csv
# We link each workshop to a new edition (Edition1 to Edition30), all editions used at least once, rest can repeat

edition_ids = [f'Edition{i}' for i in range(1, 31)]
workshop_ids = [f'Workshop{i}' for i in range(1, 51)]

# Ensure all editions are used at least once
assigned_editions = edition_ids.copy()
random.shuffle(assigned_editions)

edges = []
for i, workshop_id in enumerate(workshop_ids):
    if i < len(assigned_editions):
        edition_id = assigned_editions[i]
    else:
        edition_id = random.choice(edition_ids)
    edges.append({'edition_id': edition_id, 'workshop_id': workshop_id})

edges_df = pd.DataFrame(edges)
edges_df.to_csv(
    output_dir / 'edges_edition_held_for_workshop.csv', index=False)

# 5. edges_paper_has_review.csv

# Load reviewer-paper edges
reviews_path = input_dir / 'edges_author_reviews_paper.csv'
reviews_df = pd.read_csv(reviews_path)
unique_paper_ids = reviews_df['paper_id'].unique()

paper_review_edges = []
review_ids = []
for idx, paper_id in enumerate(unique_paper_ids, 1):
    review_id = f'review{str(idx).zfill(3)}'
    paper_review_edges.append({'paper_id': paper_id, 'review_id': review_id})
    review_ids.append(review_id)
paper_review_edges_df = pd.DataFrame(paper_review_edges)
paper_review_edges_df.to_csv(
    output_dir / 'edges_paper_has_review.csv', index=False)

# 6. nodes_review.csv

# Simple review text generator


def generate_review():
    samples = [
        "Well-structured and insightful paper.",
        "Interesting approach, but needs more experiments.",
        "Clear methodology and strong results.",
        "The paper lacks sufficient background discussion.",
        "Excellent contribution to the field.",
        "Some sections require clarification.",
        "Good writing and relevant references.",
        "The results are promising and well presented.",
        "The paper could benefit from more real-world examples.",
        "Solid work, but the novelty is limited."
    ]
    # Randomly pick or combine up to 40 words
    text = random.choice(samples)
    if random.random() < 0.3:
        text += " " + random.choice(samples)
    return text[:250]  # Ensure not too long


nodes_review = []
for review_id in review_ids:
    nodes_review.append({
        'review_id': review_id,
        'review': generate_review()
    })
nodes_review_df = pd.DataFrame(nodes_review)
nodes_review_df.to_csv(output_dir / 'nodes_review.csv', index=False)

print("All synthetic and extended CSVs created successfully.")

# ------------------------------------------------------------------------------------------
# SYNTHETIC CSV GENERATION SUMMARY
#
# This script extends our existing data from task 1 to be able comply with our knowledge graph TBOX. We generate several synthetic CSVs
# to cover aspects not present in the original data. The process is as follows:
#
# 1. Proceedings Nodes and Edges:
#    - nodes_proceedings.csv: For each unique edition (from edges_paper_published_in_edition.csv),
#      a unique proceedings node is created (proceedings_001, proceedings_002, ...).
#    - edges_edition_has_proceedings.csv: Each edition is linked to its corresponding proceedings node.
#    - edges_proceedings_includes_paper.csv: Each paper is linked to the proceedings of its edition,
#      ensuring all papers from the same edition are linked to the same proceedings.
#
# 2. Synthetic Editions:
#    - nodes_edition.csv: 30 new editions (Edition1 to Edition30) are appended, each with a random
#      year and location sampled from existing values. Any extra columns are filled with random or empty values.
#    - These editions will be linked to synthetic workshops created later.
#
# 3. Synthetic Workshops:
#    - nodes_workshop.csv: 50 workshops (Workshop1 to Workshop50) are created, each assigned a random
#      theme from the existing themes in nodes_conference.csv.
#
# 4. Workshop-Edition Relationships:
#    - edges_edition_held_for_workshop.csv: Each workshop is linked to an edition among the newly
#      created ones (Edition1 to Edition30). All new editions are used at least once; the rest are assigned randomly.
#
# 5. Paper-Review Relationships:
#    - edges_paper_has_review.csv: For each unique paper_id in edges_author_reviews_paper.csv,
#      a new review_id (review001, review002, ...) is generated and linked.
#
# 6. Synthetic Reviews:
#    - nodes_review.csv: For each review_id, a short, plausible review text is generated automatically.
#      The text is coherent and limited to a reasonable length (max ~40 words).
#
# ------------------------------------------------------------------------------------------
