from SPARQLWrapper import SPARQLWrapper, JSON, TSV
import csv

# ===----------------------------------------------------------------------===#
# Ontology ETL                                                                #
#                                                                             #
# Section C.1                                                                 #                                                                          
# Author: Walter J.T.V                                                        #
# ===----------------------------------------------------------------------===#

import argparse
from SPARQLWrapper import SPARQLWrapper, TSV, JSON
import csv
import os

DEFAULT_ENDPOINT = "http://localhost:7200/repositories/SDM2"
DEFAULT_QUERY = "queries/default.rq"
DEFAULT_FORMAT = "tsv"
DEFAULT_OUTPUT = "data/graph_export.tsv"


def load_query(query_file: str = None) -> str:
    if query_file and os.path.exists(query_file):
        with open(query_file, "r", encoding="utf-8") as f:
            return f.read()
    return DEFAULT_QUERY


def export_results(endpoint: str, query: str, output_path: str, output_format: str):
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query)

    if output_format.lower() == "tsv":
        sparql.setReturnFormat(TSV)
        results = sparql.query().convert()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(results.decode("utf-8"))

    elif output_format.lower() == "csv":
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        with open(output_path, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            headers = results["head"]["vars"]
            writer.writerow(headers)
            for result in results["results"]["bindings"]:
                row = [result.get(var, {}).get("value", "") for var in headers]
                writer.writerow(row)

    else:
        raise ValueError("Unsupported format. Use 'tsv' or 'csv'.")

    # I swear to god this emoji was not placed by any LLM model, let me cook
    print(f"✅ Exported results to `{output_path}` as {output_format.upper()}.")


def main():
    parser = argparse.ArgumentParser(description="Export data from a SPARQL endpoint.")
    parser.add_argument("--endpoint", type=str, default=DEFAULT_ENDPOINT, help="SPARQL endpoint URL")
    parser.add_argument("--query-file", type=str, default=DEFAULT_QUERY, help="Path to a SPARQL query file")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output file path")
    parser.add_argument("--format", type=str, default=DEFAULT_FORMAT, choices=["tsv", "csv"], help="Output format")

    args = parser.parse_args()
    query = load_query(args.query_file)
    export_results(args.endpoint, query, args.output, args.format)


if __name__ == "__main__":
    main()

# python export_graph.py \
#   --endpoint http://localhost:7200/repositories/SDM2 \
#   --query-file queries/default.rq \
#   --output data/graph_dump.tsv \
#   --format tsv