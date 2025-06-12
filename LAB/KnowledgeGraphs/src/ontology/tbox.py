# ===----------------------------------------------------------------------===#
# Ontology TBOX (Schema)                                                      #
#                                                                             #
# Section B.1                                                                 #
# Author: ???                                                                 #
# ===----------------------------------------------------------------------===#

from rdflib import Graph, Namespace, RDF, RDFS, XSD

# Initialize graph
g = Graph()

# Define Namespaces
URL = Namespace("http://SDM.org/research/")
g.bind("res", URL)  # res is short for research, which is the general domain.
g.bind("rdfs", RDFS)

# --- Classes ---
classes = [
    "Author", "Paper", "Conference", "Workshop", "Journal",
    "Edition", "Proceedings", "Volume", "Review", "City", "Topic"
]

for cls in classes:
    g.add((URL[cls], RDF.type, RDFS.Class))

# --- Properties (Domain/Range) ---
# Format: (property, domain, range)
properties = [
    # Authorship
    ("writes", "Author", "Paper"),
    ("hasCorrespondingAuthor", "Paper", "Author"),

    # Publication
    ("includesPaper", "Proceedings", "Paper"),
    ("hasProceedings", "Edition", "Proceedings"),
    ("includedInVolume", "Paper", "Volume"),
    ("hasVolume", "Journal", "Volume"),

    # Events
    ("hasConferenceEdition", "Conference", "Edition"),
    ("hasWorkshopEdition", "Workshop", "Edition"),
    ("heldIn", "Edition", "City"),
    ("heldOn", "Edition", XSD.date),

    # Citations
    ("cites", "Paper", "Paper"),

    # Topics / Keywords
    ("aboutTopic", "Paper", "Topic"),
    ("hasKeywords", "Topic", XSD.string),

    # Abstract
    ("hasAbstract", "Paper", XSD.string),  # Literal

    # Review process
    ("assignedTo", "Review", "Paper"),
    ("hasReviewer", "Paper", "Author"),
    ("performsReview", "Author", "Review"),

    # Volume Year
    ("volumeYear", "Volume", XSD.gYear)
]

# Add properties
for prop, domain, range_ in properties:
    g.add((URL[prop], RDF.type, RDF.Property))
    g.add((URL[prop], RDFS.domain, URL[domain]
          if isinstance(domain, str) else domain))
    if range_:
        g.add((URL[prop], RDFS.range, URL[range_]
              if isinstance(range_, str) else range_))
    else:
        g.add((URL[prop], RDFS.range, RDFS.Literal))

# --- Serialize to RDF/XML (RDFS-compliant file) ---
g.serialize(destination="GroupIvanWalter-B1-MartinezTroiani.rdfs", format="xml")

# ------------------------------------------------------------
# TBOX PROPERTY EXPLANATIONS
# ------------------------------------------------------------
# :writes                  An Author writes a Paper.
# :hasCorrespondingAuthor  A Paper has one Author as its corresponding author.
# :includedInVolume        A Paper is included in a specific Volume.
# :hasVolume               A Journal includes one or more Volumes.
# :volumeYear              A Volume is associated with a publication year (xsd:gYear).
# :cites                   A Paper cites another Paper.
# :aboutTopic              A Paper is about a specific Topic.
# :hasKeywords             A Topic is described by one or more Keywords (xsd:string).
# :hasAbstract             A Paper has a textual abstract (Literal).
# :performsReview          An Author performs (writes) a Review.
# :assignedTo              A Review is assigned to a specific Paper.
# :hasReviewer             A Paper has one or more Reviewers (Authors). Shortcut property derived from performsReview + assignedTo
# :hasConferenceEdition    A Conference has Editions.
# :hasWorkshopEdition      A Workshop has Editions.
# :heldIn                  An Edition is held in a City.
# :heldOn                  An Edition is held on a specific date (xsd:date).
# :hasProceedings          An Edition results in a Proceedings collection.
# :includesPaper           A Proceedings includes all Papers presented at the Edition.

# ------------------------------------------------------------
# UNEXPRESSIBLE CONSTRAINTS IN RDFS (documented but not enforced)
# ------------------------------------------------------------
# - A Paper must have exactly one corresponding author.
# - A Paper should have exactly three reviewers assigned.
# - An Author cannot review their own Paper (i.e., reviewer ≠ author).
# - A Journal can have multiple volumes per year (cardinality not enforced).
# - Keywords could be modeled as Literals instead of a class; here modeled as a class for semantic richness.
# - Workshops and Conferences are structurally similar but not related hierarchically (no subclass relation).

# These constraints would require OWL constructs such as:
# - owl:FunctionalProperty (for unique corresponding author)
# - owl:QualifiedCardinality or owl:cardinality (for enforcing review counts)
# - owl:AllDifferent, SWRL rules, or SHACL constraints (for reviewer ≠ author)
# - owl:UnionOf or equivalent reasoning features (for generalizing conference/workshop structure)

# In this TBOX, these constraints are documented but not enforced.
# They should be handled at the application level or with OWL/SHACL validation where appropriate.
