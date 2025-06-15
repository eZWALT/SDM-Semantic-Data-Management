import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import umap
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from adjustText import adjust_text

import torch
import pandas as pd

@st.cache_data(show_spinner=True)
def load_kg_data_and_embeddings(model_dir):
    model = torch.load(f"{model_dir}/trained_model.pkl", map_location=torch.device('cpu'))

    ent_df = pd.read_csv(f"{model_dir}/entity_to_id.tsv.gz", sep='\t', compression='gzip')
    rel_df = pd.read_csv(f"{model_dir}/relation_to_id.tsv.gz", sep='\t', compression='gzip')

    entity_to_id = dict(zip(ent_df.iloc[:, 1], ent_df.iloc[:, 0]))
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    relation_to_id = dict(zip(rel_df.iloc[:, 1], rel_df.iloc[:, 0]))

    triples_df = pd.read_csv("abox_export.tsv", sep='\t', header=None, names=["head", "relation", "tail"])
    all_triples = triples_df.values

    paper_ids_set = set()
    author_ids_set = set()

    for h, r, t in all_triples:
        if r == "<http://SDM.org/research/hasAuthor>":
            paper_ids_set.add(h)
            author_ids_set.add(t)
        elif r == "<http://SDM.org/research/cites>":
            paper_ids_set.add(h)
            paper_ids_set.add(t)

    paper_idxs = [entity_to_id[e] for e in paper_ids_set if e in entity_to_id]
    author_idxs = [entity_to_id[e] for e in author_ids_set if e in entity_to_id]

    with torch.no_grad():
        embeddings_tensor = model.entity_representations[0]().cpu().numpy()

    paper_embeddings = embeddings_tensor[paper_idxs]
    author_embeddings = embeddings_tensor[author_idxs]

    paper_ids = [id_to_entity[i] for i in paper_idxs]
    author_ids = [id_to_entity[i] for i in author_idxs]

    paper_titles = {
        uri: uri.split("/")[-1].replace(">", "").replace("_", " ")
        for uri in paper_ids
    }
    author_names = {
        uri: uri.split("/")[-1].replace(">", "").replace("_", " ")
        for uri in author_ids
    }

    paper_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
    author_to_idx = {aid: i for i, aid in enumerate(author_ids)}

    # 🔷 Perform clustering
    paper_embeddings_scaled = StandardScaler().fit_transform(paper_embeddings)
    author_embeddings_scaled = StandardScaler().fit_transform(author_embeddings)

    k_paper = compute_optimal_clusters(paper_embeddings_scaled, max_k=8)
    k_author = compute_optimal_clusters(author_embeddings_scaled, max_k=8)

    kmeans_paper = KMeans(n_clusters=k_paper, random_state=42).fit(paper_embeddings_scaled)
    kmeans_author = KMeans(n_clusters=k_author, random_state=42).fit(author_embeddings_scaled)

    clusters_papers = kmeans_paper.labels_
    clusters_authors = kmeans_author.labels_

    return (
        model,
        paper_ids, author_ids,
        paper_embeddings, author_embeddings,
        clusters_papers, clusters_authors,
        paper_to_idx, author_to_idx,
        paper_titles, author_names
    )

def compute_optimal_clusters(X, max_k=10):
    scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append((k, score))
    best_k, _ = max(scores, key=lambda x: x[1])
    return best_k

def recommend_items(query, embeddings, id_to_idx, top_k=10):
    if query not in id_to_idx:
        return [], []
    idx = id_to_idx[query]
    query_vec = embeddings[idx].reshape(1, -1)
    sims = cosine_similarity(query_vec, embeddings).flatten()
    sims[idx] = -1  # exclude self
    top_indices = sims.argsort()[-top_k:][::-1]
    return top_indices, sims[top_indices]

def get_reducer(name="umap"):
    if name == "umap":
        return umap.UMAP(random_state=42)
    elif name == "pca":
        return PCA(n_components=2)
    elif name == "tsne":
        return TSNE(random_state=42)
    else:
        raise ValueError(f"Unknown reducer: {name}")

def plot_clusters(embeddings, clusters, title, reducer_name="umap", labels=None, show_labels=False):
    reducer = get_reducer(reducer_name)
    reduced = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.set(style="whitegrid")

    sns.scatterplot(
        x=reduced[:, 0],
        y=reduced[:, 1],
        hue=clusters,
        palette="tab10",
        s=100,
        alpha=0.85,
        edgecolor="k",
        ax=ax
    )

    ax.set_title(f"{title} ({reducer_name.upper()} Projection)", fontsize=16)
    ax.set_xlabel(f"{reducer_name.upper()} Dimension 1")
    ax.set_ylabel(f"{reducer_name.upper()} Dimension 2")
    ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')

    if show_labels and labels is not None:
        texts = []
        for (x, y, label) in zip(reduced[:, 0], reduced[:, 1], labels):
            texts.append(ax.text(x, y, label, fontsize=7.5, alpha=0.8))
        try:
            adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))
        except Exception:
            pass

    st.pyplot(fig)
    plt.clf()

def paper_recommender_ui(paper_ids, embeddings_papers, paper_to_idx, paper_titles):
    st.subheader("📄 Paper Recommender")
    col1, col2 = st.columns([1, 5])
    with col1:
        selected_paper = st.selectbox("Choose Paper ID", options=paper_ids, index=0)
    with col2:
        st.success(paper_titles[selected_paper])

    top_idx, scores = recommend_items(selected_paper, embeddings_papers, paper_to_idx)

    st.markdown("### Recommendations")
    for i, (idx, score) in enumerate(zip(top_idx, scores), start=1):
        pid = paper_ids[idx]
        title = paper_titles[pid]
        similarity_color = f"rgba({int(255 - score*255)}, {int(score*255)}, 150, 0.2)"
        with st.container():
            st.markdown(f"""
            <div style="background-color:{similarity_color}; padding: 12px; border-radius: 10px; margin-bottom: 10px;">
                <b>🔗 {i}. <code>{pid}</code></b>  
                <div style='margin-top: 5px;'>{title}</div>
                <span style='font-size: 0.9em;'>💡 Similarity Score: <b>{score:.3f}</b></span>
            </div>
            """, unsafe_allow_html=True)

def author_recommender_ui(author_ids, embeddings_authors, author_to_idx, author_names):
    st.subheader("👩‍🔬 Author Recommender")
    col1, col2 = st.columns([1, 5])
    with col1:
        selected_author = st.selectbox("Choose Author ID", options=author_ids, index=0)
    with col2:
        st.info(author_names[selected_author])

    top_idx, scores = recommend_items(selected_author, embeddings_authors, author_to_idx)

    st.markdown("### Recommendations")
    for i, (idx, score) in enumerate(zip(top_idx, scores), start=1):
        aid = author_ids[idx]
        name = author_names[aid]
        similarity_color = f"rgba({int(255 - score*255)}, {int(score*255)}, 200, 0.15)"
        with st.container():
            st.markdown(f"""
            <div style="background-color:{similarity_color}; padding: 12px; border-radius: 10px; margin-bottom: 10px;">
                <b>📚 {i}. <code>{aid}</code></b>  
                <div style='margin-top: 5px;'>{name}</div>
                <span style='font-size: 0.9em;'>💡 Similarity Score: <b>{score:.3f}</b></span>
            </div>
            """, unsafe_allow_html=True)

def clustering_ui(
    embeddings_authors,
    clusters_authors,
    embeddings_papers,
    clusters_papers,
    author_names,
    paper_titles,
    reducer_name="umap"
):
    st.header("Clustering Visualizations")

    st.subheader("👥 Authors Clustering")
    st.caption(f"Clustered into {len(set(clusters_authors))} groups")
    show_labels_auth = st.checkbox("Show Author Labels", key="auth_labels")
    author_labels = [author_names[aid] for aid in sorted(author_names)]
    plot_clusters(embeddings_authors, clusters_authors, "Clusters of Authors", reducer_name, labels=author_labels, show_labels=show_labels_auth)

    st.subheader("📚 Papers Clustering")
    st.caption(f"Clustered into {len(set(clusters_papers))} groups")
    show_labels_paper = st.checkbox("Show Paper Labels", key="paper_labels")
    paper_labels = [paper_titles[pid] for pid in sorted(paper_titles)]
    plot_clusters(embeddings_papers, clusters_papers, "Clusters of Papers", reducer_name, labels=paper_labels, show_labels=show_labels_paper)

def main():
    st.set_page_config(page_title="KGE Explorer", layout="wide")
    st.title("🎓 Knowledge Graph Embeddings Explorer")

    with st.expander("ℹ️ About this app", expanded=True):
        st.markdown("""
        Welcome to the **Knowledge Graph Embeddings Explorer**!  
        This interactive app lets you:
        - Explore **recommendations** for papers and authors using embedding similarity.
        - Visualize **clustering** of entities based on their semantic relationships.
        """)

    model_dir = "models/transh_model"

    (
        model,
        paper_ids, author_ids,
        embeddings_papers, embeddings_authors,
        clusters_papers, clusters_authors,
        paper_to_idx, author_to_idx,
        paper_titles, author_names
    ) = load_kg_data_and_embeddings(model_dir)

    with st.sidebar:
        st.header("🛠️ Settings")
        task = st.radio(
            "Choose Task",
            ("Recommend Papers", "Recommend Authors", "Show Clustering"),
            index=2
        )

        reducer_choice = st.selectbox(
            "Dimensionality Reduction",
            options=["umap", "pca", "tsne"],
            index=0
        )

    if task == "Recommend Papers":
        paper_recommender_ui(paper_ids, embeddings_papers, paper_to_idx, paper_titles)

    elif task == "Recommend Authors":
        author_recommender_ui(author_ids, embeddings_authors, author_to_idx, author_names)

    elif task == "Show Clustering":
        clustering_ui(
            embeddings_authors,
            clusters_authors,
            embeddings_papers,
            clusters_papers,
            author_names,
            paper_titles,
            reducer_name=reducer_choice
        )

if __name__ == "__main__":
    main()
