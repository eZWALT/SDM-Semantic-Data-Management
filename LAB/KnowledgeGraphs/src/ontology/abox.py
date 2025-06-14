# ===----------------------------------------------------------------------===#
# Ontology ABOX (instances)                                                   #
# Section B.2                                                                 #
# ===----------------------------------------------------------------------===#

import pandas as pd
import re
from pathlib import Path
from rdflib import Graph, Namespace, RDF, RDFS, Literal, XSD, URIRef
from collections import defaultdict

# === Initialize RDF Graph and Namespaces ===
g = Graph()
DATA_DIR = Path("./abox_data_raw")
OUT_PATH = "GroupIvanWalter-B2-MartinezTroiani.rdf"

URL = Namespace("http://SDM.org/research/")
g.bind("res", URL)
g.bind("rdf", RDF)
g.bind("rdfs", RDFS)

# === URI Sanitization Utilities ===


def clean_string_for_uri(value: str) -> str:
    """Remove punctuation, normalize whitespace and sanitize for use in URI."""
    value = re.sub(r'[^\w\s-]', '', value)
    return re.sub(r'\s+', '_', value.strip())


# === Duplicate Handling ===
author_name_counts = defaultdict(int)
# To be able to use conference names as URIs
conf_name_counts = defaultdict(int)
# To be able to use journal names as URIs
journal_name_counts = defaultdict(int)


def get_unique_author_uri(name: str) -> URIRef:
    """Create unique URI for Author based on name, with suffix if needed."""
    base = clean_string_for_uri(name)
    count = author_name_counts[base]
    uri = URL[base] if count == 0 else URL[f"{base}_{count}"]
    author_name_counts[base] += 1
    return uri


def get_unique_conference_uri(name: str) -> URIRef:
    """Create unique URI for Conference based on name, with suffix if needed."""
    base = clean_string_for_uri(name)
    count = conf_name_counts[base]
    uri = URL[base] if count == 0 else URL[f"{base}_{count}"]
    conf_name_counts[base] += 1
    return uri


def get_unique_journal_uri(name: str) -> URIRef:
    """Create unique URI for Journal based on name, with suffix if needed."""
    base = clean_string_for_uri(name)
    count = journal_name_counts[base]
    uri = URL[base] if count == 0 else URL[f"{base}_{count}"]
    journal_name_counts[base] += 1
    return uri


def uri(cls, raw_value):
    """Generate URIs for individuals based on class-specific logic."""
    if cls == "Author":
        return get_unique_author_uri(str(raw_value))
    elif cls == "Paper":
        return URL[clean_string_for_uri(str(raw_value))]
    elif cls == "Conference":
        return get_unique_conference_uri(str(raw_value))
    elif cls == "Journal":
        return get_unique_journal_uri(str(raw_value))
    else:
        # For all other entities (Review, Volume, Proceedings, etc.), use ID directly
        return URL[clean_string_for_uri(str(raw_value))]


# === 1. Authors and Reviewers ===
authors = pd.read_csv(DATA_DIR / "nodes_author.csv")
author_id_to_uri = {}
for _, row in authors.iterrows():
    author_uri = uri("Author", row["name"])
    author_id_to_uri[str(row["author_id"]).strip()] = author_uri
    g.add((author_uri, RDF.type, URL.Author))

# === 2. Papers and Associated Topics ===
papers = pd.read_csv(DATA_DIR / "nodes_paper.csv")
paper_title_to_uri = {}

for _, row in papers.iterrows():
    paper_uri = uri("Paper", row["title"])
    paper_title_to_uri[row["paper_id"]] = paper_uri
    g.add((paper_uri, RDF.type, URL.Paper))

    if pd.notnull(row["abstract"]):
        g.add((paper_uri, URL.hasAbstract, Literal(row["abstract"])))

# Process fields_of_study as Topic URI
if pd.notnull(row["fields_of_study"]):
    topics_raw = eval(row["fields_of_study"])
    topic_label = "_and_".join(clean_string_for_uri(t) for t in topics_raw)
else:
    topic_label = "NoSpecifiedTopic"

topic_uri = URL[topic_label]
g.add((paper_uri, URL.aboutTopic, topic_uri))
g.add((topic_uri, RDF.type, URL.Topic))

if pd.notnull(row["keywords"]):
    g.add((topic_uri, URL.hasKeywords, Literal(row["keywords"].strip())))

# === 3. Reviews and their Text ===
reviews = pd.read_csv(DATA_DIR / "nodes_review.csv")
for _, row in reviews.iterrows():
    review_uri = uri("Review", row["review_id"])
    g.add((review_uri, RDF.type, URL.Review))
    g.add((review_uri, URL.hasContent, Literal(row["review"])))

# === 4. Journals ===
journals = pd.read_csv(DATA_DIR / "nodes_journal.csv")
journal_id_to_uri = {}
for _, row in journals.iterrows():
    journal_uri = uri("Journal", row["name"])
    journal_id_to_uri[row["journal_id"]] = journal_uri
    g.add((journal_uri, RDF.type, URL.Journal))
    # g.add((journal_uri, URL.name, Literal(row["name"])))
    # g.add((journal_uri, URL.issn, Literal(row["issn"])))


# === 5. Volumes ===
volumes = pd.read_csv(DATA_DIR / "nodes_volume.csv")
for _, row in volumes.iterrows():
    volume_uri = uri("Volume", row["volume_id"])
    g.add((volume_uri, RDF.type, URL.Volume))
# Convert volume number safely, default to 0 if invalid (e.g., NA, PP, etc.)
raw_number = str(row["number"]).strip()
try:
    number = int(raw_number)
except (ValueError, TypeError):
    number = 0  # Default for malformed or missing volume number

g.add((volume_uri, URL.hasNumber, Literal(number, datatype=XSD.integer)))

g.add((volume_uri, URL.volumeYear, Literal(
    int(row["year"]), datatype=XSD.gYear)))

# === 6. Conferences (by name) and Workshops ===
confs = pd.read_csv(DATA_DIR / "nodes_conference.csv")
conf_id_to_uri = {}
for _, row in confs.iterrows():
    conf_uri = get_unique_conference_uri(row["name"])
    conf_id_to_uri[row["conference_id"]] = conf_uri
    g.add((conf_uri, RDF.type, URL.Conference))
    # g.add((conf_uri, URL.name, Literal(row["name"])))

workshops = pd.read_csv(DATA_DIR / "nodes_workshop.csv")
for _, row in workshops.iterrows():
    ws_uri = uri("Workshop", row["workshop_id"])
    g.add((ws_uri, RDF.type, URL.Workshop))
    # g.add((ws_uri, URL.theme, Literal(row["theme"])))

# === 7. Editions with City ===
editions = pd.read_csv(DATA_DIR / "nodes_edition.csv")
for _, row in editions.iterrows():
    edition_uri = uri("Edition", row["edition_id"])
    g.add((edition_uri, RDF.type, URL.Edition))
    g.add((edition_uri, URL.heldOn, Literal(
        f"{row['year']}-01-01", datatype=XSD.date)))
    city = row["location"].split(",")[0].strip()
    city_uri = URL[clean_string_for_uri(city)]
    g.add((city_uri, RDF.type, URL.City))
    g.add((edition_uri, URL.heldIn, city_uri))

# === 8. Proceedings ===
proceedings = pd.read_csv(DATA_DIR / "nodes_proceedings.csv")
for _, row in proceedings.iterrows():
    g.add(
        (uri("Proceedings", row["proceedings_id"]), RDF.type, URL.Proceedings))

# === 9. Edges (Relationships) ===


def add_edge(csv, subj_cls, pred, obj_cls, subj_col, obj_col, use_title_for_paper=False):
    df = pd.read_csv(DATA_DIR / csv)
    for _, row in df.iterrows():
        # === Resolve subject URI ===
        if use_title_for_paper and subj_cls == "Paper":
            subj_uri = paper_title_to_uri[row[subj_col]]
        elif subj_cls == "Conference":
            subj_uri = conf_id_to_uri[row[subj_col]]
        elif subj_cls == "Author":
            subj_uri = author_id_to_uri.get(str(row[subj_col]).strip())
        elif subj_cls == "Journal":
            subj_uri = journal_id_to_uri[row[subj_col]]
        else:
            subj_uri = uri(subj_cls, row[subj_col])

        # === Resolve object URI ===
        if use_title_for_paper and obj_cls == "Paper":
            obj_uri = paper_title_to_uri[row[obj_col]]
        elif obj_cls == "Conference":
            obj_uri = conf_id_to_uri[row[obj_col]]
        elif obj_cls == "Author":
            obj_uri = author_id_to_uri.get(str(row[obj_col]).strip())
        elif obj_cls == "Journal":
            obj_uri = journal_id_to_uri[row[obj_col]]
        else:
            obj_uri = uri(obj_cls, row[obj_col])

        # === Add triple ===
        g.add((subj_uri, URL[pred], obj_uri))


# Authorship
add_edge("edges_author_writes_paper.csv", "Paper", "hasAuthor",
         "Author", "paper_id", "author_id", use_title_for_paper=True)
add_edge("edges_author_corresponds_paper.csv", "Paper", "hasCorrespondingAuthor",
         "Author", "paper_id", "author_id", use_title_for_paper=True)

# Reviews
add_edge("edges_paper_has_review.csv", "Paper", "hasReview",
         "Review", "paper_id", "review_id", use_title_for_paper=True)

# Join to properly link Review -> performedBy -> Reviewer
reviewer_map = pd.read_csv(DATA_DIR / "edges_author_reviews_paper.csv")
paper_to_review = pd.read_csv(DATA_DIR / "edges_paper_has_review.csv")

# Join to get (review_id, reviewer_id)
review_performance = pd.merge(paper_to_review, reviewer_map, on="paper_id")

for _, row in review_performance.iterrows():
    review_uri = uri("Review", row["review_id"])
    reviewer_uri = author_id_to_uri.get(str(row["reviewer_id"]).strip())
    g.add((review_uri, URL["performedBy"], reviewer_uri))


# Publication
add_edge("edges_paper_published_in_volume.csv", "Paper", "publishedIn",
         "Volume", "paper_id", "volume_id", use_title_for_paper=True)
add_edge("edges_volume_issued_by_journal.csv", "Journal",
         "hasVolume", "Volume", "journal_id", "volume_id")

# Events
add_edge("edges_edition_held_for_conference.csv", "Conference",
         "hasEdition", "Edition", "conference_id", "edition_id")
add_edge("edges_edition_held_for_workshop.csv", "Workshop",
         "hasEdition", "Edition", "workshop_id", "edition_id")

# Proceedings
add_edge("edges_edition_has_proceedings.csv", "Edition",
         "hasProceedings", "Proceedings", "edition_id", "proceedings_id")
add_edge("edges_proceedings_includes_paper.csv", "Proceedings", "includesPaper",
         "Paper", "proceedings_id", "paper_id", use_title_for_paper=True)

# Citations
add_edge("edges_paper_cites_paper.csv", "Paper", "cites", "Paper",
         "citing_paper_id", "cited_paper_id", use_title_for_paper=True)

# === Serialize Graph ===
g.serialize(destination=OUT_PATH, format="xml")
print(f"ABOX successfully created at: {OUT_PATH}")

# ------------------------------------------------------------
# ABOX DESIGN NOTES AND MODELING STRATEGY
# ------------------------------------------------------------

# URI GENERATION AND DISAMBIGUATION
# ------------------------------------------------------------
# - URIs for Authors, Conferences, Journals, and Papers are based on their names or titles.
#   This improves readability and semantic clarity in the knowledge graph.
# - All string inputs are sanitized to remove punctuation and normalize whitespace using
#   clean_string_for_uri(). Spaces become underscores, special characters are removed.
#
# - Deduplication logic is applied: if multiple entities share the same cleaned name,
#   a numeric suffix (_1, _2, ...) is appended to ensure each URI is unique.
#   For example: "John Smith" becomes John_Smith, John_Smith_1, John_Smith_2.
#
# - Paper URIs are derived directly from their sanitized titles.
# - Topic URIs are built by concatenating all fields of study for a paper using "_and_".
#   These URIs are shared across papers that reference the same topic(s), such as:
#   http://SDM.org/research/Medicine_and_AI
#   The same topic may be linked to multiple papers, and the associated keywords can differ.
#   This is supported in RDF, as literals are not considered defining identifiers for resources.
#
# - If no fields_of_study are present, the fallback topic URI "NoSpecifiedTopic" is used.
# - City URIs are constructed from the first part of the location field (before comma),
#   using the raw city name (e.g., http://SDM.org/research/London).
#   No deduplication suffix is applied, so cities with the same name share a single URI.

# TOPIC AND KEYWORDS MODELING
# ------------------------------------------------------------
# - Topics are dynamically generated based on the fields_of_study of each paper.
# - Keywords are linked to the Topic (not directly to the Paper) using :hasKeywords.
# - Each :hasKeywords triple contains the entire keyword string as found in the paper CSV.
#   This means multiple papers sharing the same topic may contribute different keyword strings
#   to the same Topic resource.
# - This design reflects the idea that keywords describe the topic collectively, not exclusively.

# EDGE MODELING
# ------------------------------------------------------------
# - Edges (relationships) are added by reading CSV files and matching subjects/objects
#   to the correct URIs based on class and CSV column mappings.
# - Special cases:
#   * Papers use title-based URIs, tracked with paper_title_to_uri.
#   * Conferences use name-based URIs, mapped from their CSV ID via conf_id_to_uri.
#   * Journals use name-based URIs, mapped similarly using journal_id_to_uri.
# - Relationship direction strictly follows the TBOX definition.
#   For example, :hasAuthor has domain :Paper and range :Author,
#   so triples are created as: <Paper> :hasAuthor <Author>.

# RDF TYPE MINIMIZATION
# ------------------------------------------------------------
# - rdf:type is only added for individuals where it is not otherwise inferable from properties.
# - This keeps the RDF lean and relies on RDFS reasoning to infer class membership
#   where appropriate.

# SERIALIZATION
# ------------------------------------------------------------
# - The RDF graph is serialized in RDF/XML format.
# - Output is written to:
#   "GroupIvanWalter-B2-MartinezTroiani.rdf"

# ------------------------------------------------------------
