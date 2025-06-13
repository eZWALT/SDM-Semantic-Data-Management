# ===----------------------------------------------------------------------===#
# Ontology TBOX (Schema)                                                      #
#                                                                             #
# Section B.1                                                                 #
# ===----------------------------------------------------------------------===#

from rdflib import Graph, Namespace, RDF, RDFS, XSD

# Initialize graph
g = Graph()

# Define Namespaces
URL = Namespace("http://SDM.org/research/")
g.bind("res", URL)
g.bind("rdfs", RDFS)

# --- Classes ---
classes = [
    "Author", "Reviewer", "Paper", "Conference", "Workshop", "Journal",
    "Edition", "Proceedings", "Volume", "Review", "City", "Topic", "Event"
]

for cls in classes:
    g.add((URL[cls], RDF.type, RDFS.Class))

# Subclasses
g.add((URL["Conference"], RDFS.subClassOf, URL["Event"]))
g.add((URL["Workshop"], RDFS.subClassOf, URL["Event"]))
g.add((URL["Reviewer"], RDFS.subClassOf, URL["Author"]))

# Subproperty
g.add((URL["hasCorrespondingAuthor"], RDFS.subPropertyOf, URL["hasAuthor"]))

# --- Properties (Domain/Range) ---
properties = [
    # Authorship
    # ("writes", "Author", "Paper"),
    ("hasAuthor", "Paper", "Author"),
    ("hasCorrespondingAuthor", "Paper", "Author"),

    # Publication
    ("includesPaper", "Proceedings", "Paper"),
    ("hasProceedings", "Edition", "Proceedings"),
    ("publishedIn", "Paper", "Volume"),
    ("hasVolume", "Journal", "Volume"),
    ("hasNumber", "Volume", XSD.integer),

    # Events
    ("hasEdition", "Event", "Edition"),
    ("heldIn", "Edition", "City"),
    ("heldOn", "Edition", XSD.date),

    # Citations
    ("cites", "Paper", "Paper"),

    # Topics / Keywords
    ("aboutTopic", "Paper", "Topic"),
    ("hasKeywords", "Topic", XSD.string),

    # Abstract
    ("hasAbstract", "Paper", XSD.string),

    # Review process
    ("hasReview", "Paper", "Review"),
    ("performedBy", "Review", "Reviewer"),
    ("hasContent", "Review", XSD.string),

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

# Serialize to RDF/XML
g.serialize(destination="GroupIvanWalter-B1-MartinezTroiani.rdfs", format="xml")

# ------------------------------------------------------------
# TBOX PROPERTY EXPLANATIONS
# ------------------------------------------------------------
# :hasAuthor              A Paper is written by one or more Authors.
# :hasCorrespondingAuthor A Paper has one Author as its corresponding author (subproperty of hasAuthor).
# :includesPaper          A Proceedings includes all Papers presented at the Edition.
# :hasProceedings         An Edition results in a Proceedings collection.
# :publishedIn            A Paper is published in a specific Volume.
# :hasVolume              A Journal includes one or more Volumes.
# :hasNumber              A Volume has a numeric identifier (e.g., Volume 2).
# :volumeYear             A Volume is associated with a publication year (xsd:gYear).
# :hasEdition             A Conference or Workshop (Event) has one or more Editions.
# :heldIn                 An Edition is held in a City.
# :heldOn                 An Edition is held on a specific date (xsd:date).
# :cites                  A Paper cites another Paper.
# :aboutTopic             A Paper is about a specific Topic.
# :hasKeywords            A Topic is described by one or more Keywords (xsd:string).
# :hasAbstract            A Paper has a textual abstract (Literal).
# :hasReview              A Paper has a Review assigned to it.
# :performedBy            A Review is performed by a Reviewer.

# ------------------------------------------------------------
# UNEXPRESSIBLE CONSTRAINTS IN RDFS (documented but not enforced)
# ------------------------------------------------------------
# - A Paper must have exactly one corresponding author.
# - A Paper should have exactly three reviews.
# - An Author cannot review their own Paper (i.e., reviewer ≠ author).
# - A Journal can have multiple volumes per year (cardinality not enforced).
# - Keywords could be modeled as separate class; here modeled as strings for simplicity.
# - Reviewer is defined as a subclass of Author.
# - Conference and Workshop are subclasses of Event; Editions apply generically to Events.

# These constraints would require OWL constructs such as:
# - owl:FunctionalProperty (for unique corresponding author)
# - owl:QualifiedCardinality or owl:cardinality (for enforcing review counts)
# - owl:AllDifferent, SWRL rules, or SHACL constraints (for reviewer ≠ author)
# - owl:UnionOf or equivalent reasoning features (for generalizing conference/workshop structure)

# In this TBOX, these constraints are documented but not enforced.
# They should be handled at the application level or with OWL/SHACL validation where appropriate.

# Removed :writes to avoid redundancy with :hasAuthor. Semantically inverse, but not expressible in RDFS.
