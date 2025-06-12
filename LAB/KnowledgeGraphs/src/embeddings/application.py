from SPARQLWrapper import SPARQLWrapper, JSON
import streamlit as st
from streamlit_extras.badges import badge 
import numpy as np
import seaborn as sns
from adjustText import adjust_text
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import umap
from sklearn.manifold import TSNE

# ===----------------------------------------------------------------------===#
# Embeddings Applications (And dashboard)                                     #
#                                                                             #
# Section C.4                                                                 #                                                                          
# Author: Walter J.T.V                                                        #
# ===----------------------------------------------------------------------===#

###################################### DUMMY LOADER (Replace with real KG data)

def load_dummy_data():
    np.random.seed(42)
    paper_ids = [f"P{str(i).zfill(3)}" for i in range(1, 51)]
    author_ids = [f"A{str(i).zfill(3)}" for i in range(1, 21)]

    embeddings_papers = np.random.rand(len(paper_ids), 100)
    embeddings_authors = np.random.rand(len(author_ids), 100)

    clusters_papers = np.random.randint(0, 5, size=len(paper_ids))
    clusters_authors = np.random.randint(0, 3, size=len(author_ids))

    paper_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
    author_to_idx = {aid: i for i, aid in enumerate(author_ids)}

    paper_titles = {pid: f"Paper Title {pid}" for pid in paper_ids}
    author_names = {aid: f"Author {aid}" for aid in author_ids}

    return (paper_ids, author_ids,
            embeddings_papers, embeddings_authors,
            clusters_papers, clusters_authors,
            paper_to_idx, author_to_idx,
            paper_titles, author_names)

###################################### RECOMMENDATION

def recommend_items(query, embeddings, id_to_idx, top_k=5):
    if query not in id_to_idx:
        return [], []
    idx = id_to_idx[query]
    query_vec = embeddings[idx].reshape(1, -1)
    sims = cosine_similarity(query_vec, embeddings).flatten()
    sims[idx] = -1  # exclude self
    top_indices = sims.argsort()[-top_k:][::-1]
    return top_indices, sims[top_indices]

###################################### REDUCTION METHODS

def get_reducer(name="umap"):
    if name == "umap":
        return umap.UMAP(random_state=42)
    elif name == "pca":
        return PCA(n_components=2)
    elif name == "tsne":
        return TSNE(random_state=42)
    else:
        raise ValueError(f"Unknown reducer: {name}")

###################################### CLUSTERING VISUALIZATION

def plot_clusters(embeddings, clusters, title, reducer_name="umap", labels=None, show_labels=False):
    reducer = get_reducer(reducer_name)
    reduced = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.set(style="whitegrid")

    scatter = sns.scatterplot(
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
            pass  # gracefully skip if adjust_text fails

    st.pyplot(fig)
    plt.clf()

###################################### RECOMMENDER SYSTEMS (UI)

def paper_recommender_ui(paper_ids, embeddings_papers, paper_to_idx, paper_titles):
    st.subheader("📄 Paper Recommender")

    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown(f"**Search for a Paper**")
        selected_paper = st.selectbox("Choose Paper ID", options=paper_ids, index=0)
    with col2:
        st.markdown("**Title**")
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
        st.markdown(f"**Search for an Author**")
        selected_author = st.selectbox("Choose Author ID", options=author_ids, index=0)
    with col2:
        st.markdown("**Name**")
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

###################################### CLUSTERING UI

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
    with st.expander("ℹ️ About this plot", expanded=False):
        st.markdown(f"""
        This visualization shows how **authors** are grouped based on their vector embeddings from the knowledge graph.  
        Similar authors are placed closer together after **{reducer_name.upper()}** dimensionality reduction.
        """)
    show_labels_auth = st.checkbox("Show Author Labels", key="auth_labels")
    author_labels = [author_names[aid] for aid in sorted(author_names)]
    plot_clusters(embeddings_authors, clusters_authors, "Clusters of Authors", reducer_name, labels=author_labels, show_labels=show_labels_auth)

    st.subheader("📚 Papers Clustering")
    with st.expander("ℹ️ About this plot", expanded=False):
        st.markdown(f"""
        This shows how **papers** are clustered.  
        The embedding vectors may reflect semantic similarity, citation networks, or topic relationships, projected using **{reducer_name.upper()}**.
        """)
    show_labels_paper = st.checkbox("Show Paper Labels", key="paper_labels")
    paper_labels = [paper_titles[pid] for pid in sorted(paper_titles)]
    plot_clusters(embeddings_papers, clusters_papers, "Clusters of Papers", reducer_name, labels=paper_labels, show_labels=show_labels_paper)

###################################### MAIN APP

def main():
    st.set_page_config(page_title="KGE Explorer", layout="wide")
    st.title("🎓 Knowledge Graph Embeddings Explorer")

    with st.expander("ℹ️ About this app", expanded=True):
        st.markdown("""
        Welcome to the **Knowledge Graph Embeddings Explorer**!  
        This interactive app lets you:
        - Explore **recommendations** for papers and authors using embedding similarity.
        - Visualize **clustering** of entities based on their semantic relationships.
        
        Built for navigating complex knowledge graph representations with ease!
        """)

    # Load dummy embeddings and metadata
    (paper_ids, author_ids,
     embeddings_papers, embeddings_authors,
     clusters_papers, clusters_authors,
     paper_to_idx, author_to_idx,
     paper_titles, author_names) = load_dummy_data()

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
            index=1
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

