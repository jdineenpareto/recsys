"""
Ontology API Package

Provides unified access to multiple biomedical and skills ontologies:
- Gene Ontology (GO)
- ChEBI (Chemical Entities)
- ESCO (European Skills/Competences)
- Cell Ontology (CL)
- MeSH (Medical Subject Headings)

Each API provides:
1. Node retrieval: get_node(node_id)
2. Neighbor navigation: get_neighbors(node_id, relation_types, direction)
3. RAG-based term matching: match_term_to_nodes(query_term, top_k, min_similarity)
4. Batch matching: batch_match_terms(terms, top_k, min_similarity)
"""

from .base_ontology import (
    BaseOntologyAPI,
    OntologyNode,
    OntologyRelation,
    TermMatch
)
from .go_api import GeneOntologyAPI
from .chebi_api import ChEBIAPI
from .esco_api import ESCOAPI
from .cell_ontology_api import CellOntologyAPI
from .mesh_api import MeSHAPI
from .ontology_manager import OntologyManager

__all__ = [
    # Base classes
    'BaseOntologyAPI',
    'OntologyNode',
    'OntologyRelation',
    'TermMatch',

    # Individual APIs
    'GeneOntologyAPI',
    'ChEBIAPI',
    'ESCOAPI',
    'CellOntologyAPI',
    'MeSHAPI',

    # Unified manager
    'OntologyManager'
]

__version__ = '0.1.0'
