from SPARQLWrapper import SPARQLWrapper, JSON


# ===----------------------------------------------------------------------===#
# Embeddings Exploration                                                      #
#                                                                             #
# Section C.2                                                                 #                                                                          
# Author: Walter J.T.V                                                        #
# ===----------------------------------------------------------------------===#

# Set up the GraphDB endpoint
sparql = SPARQLWrapper("http://localhost:7200/repositories/SDM2")

    
    
if __name__ == "__main__":
    sparql.setQuery("""
        SELECT ?s ?p ?o
        WHERE {
            ?s ?p ?o .
        }
        LIMIT 100
    """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    # Extract and print
    for result in results["results"]["bindings"]:
        print(result["s"]["value"], result["p"]["value"], result["o"]["value"])