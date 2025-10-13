"""
Unified Ontology Manager

Provides a single interface to access all loaded ontologies and perform
cross-ontology term matching for comprehensive coverage.
"""

from typing import List, Dict, Optional, Any
from .base_ontology import OntologyNode, TermMatch
from .go_api import GeneOntologyAPI
from .chebi_api import ChEBIAPI
from .esco_api import ESCOAPI
from .cell_ontology_api import CellOntologyAPI
from .mesh_api import MeSHAPI


class OntologyManager:
    """
    Unified manager for all ontology APIs

    Provides:
    - Lazy loading of ontologies
    - Cross-ontology term matching
    - Unified search across all ontologies
    """

    def __init__(self):
        """Initialize ontology manager (ontologies loaded on demand)"""
        self._apis: Dict[str, Any] = {}
        self._loaded: Dict[str, bool] = {
            'go': False,
            'chebi': False,
            'esco': False,
            'cell': False,
            'mesh': False
        }

    def load_go(self, use_basic: bool = True) -> GeneOntologyAPI:
        """
        Load Gene Ontology

        Args:
            use_basic: Use go-basic.obo (smaller, recommended) vs full go.obo
        """
        if not self._loaded['go']:
            print("\n" + "="*60)
            print("Loading Gene Ontology...")
            print("="*60)
            self._apis['go'] = GeneOntologyAPI(use_basic=use_basic)
            self._loaded['go'] = True
        return self._apis['go']

    def load_chebi(self) -> ChEBIAPI:
        """Load ChEBI chemical ontology"""
        if not self._loaded['chebi']:
            print("\n" + "="*60)
            print("Loading ChEBI...")
            print("="*60)
            self._apis['chebi'] = ChEBIAPI()
            self._loaded['chebi'] = True
        return self._apis['chebi']

    def load_esco(self) -> ESCOAPI:
        """Load ESCO skills/competences taxonomy"""
        if not self._loaded['esco']:
            print("\n" + "="*60)
            print("Loading ESCO...")
            print("="*60)
            self._apis['esco'] = ESCOAPI()
            self._loaded['esco'] = True
        return self._apis['esco']

    def load_cell_ontology(self) -> CellOntologyAPI:
        """Load Cell Ontology"""
        if not self._loaded['cell']:
            print("\n" + "="*60)
            print("Loading Cell Ontology...")
            print("="*60)
            self._apis['cell'] = CellOntologyAPI()
            self._loaded['cell'] = True
        return self._apis['cell']

    def load_mesh(self, use_rdf: bool = False) -> MeSHAPI:
        """
        Load MeSH medical terminology

        Args:
            use_rdf: Use RDF format (True) vs XML format (False, recommended)
        """
        if not self._loaded['mesh']:
            print("\n" + "="*60)
            print("Loading MeSH...")
            print("="*60)
            self._apis['mesh'] = MeSHAPI(use_rdf=use_rdf)
            self._loaded['mesh'] = True
        return self._apis['mesh']

    def load_all(self,
                 go_basic: bool = True,
                 mesh_rdf: bool = False,
                 skip_large: bool = False) -> None:
        """
        Load all ontologies

        Args:
            go_basic: Use GO basic version (smaller)
            mesh_rdf: Use MeSH RDF format (vs XML)
            skip_large: Skip large ontologies (ChEBI ~1GB)
        """
        print("\n" + "="*60)
        print("Loading All Ontologies")
        print("="*60)

        self.load_go(use_basic=go_basic)
        self.load_esco()
        self.load_cell_ontology()
        self.load_mesh(use_rdf=mesh_rdf)

        if not skip_large:
            self.load_chebi()
        else:
            print("\nSkipping ChEBI (large ontology)")

        print("\n" + "="*60)
        print("All ontologies loaded!")
        print("="*60)

    def get_api(self, ontology_name: str):
        """
        Get a specific ontology API

        Args:
            ontology_name: 'go', 'chebi', 'esco', 'cell', or 'mesh'
        """
        if ontology_name not in self._loaded:
            raise ValueError(f"Unknown ontology: {ontology_name}")

        if not self._loaded[ontology_name]:
            # Auto-load on demand
            load_methods = {
                'go': self.load_go,
                'chebi': self.load_chebi,
                'esco': self.load_esco,
                'cell': self.load_cell_ontology,
                'mesh': self.load_mesh
            }
            load_methods[ontology_name]()

        return self._apis[ontology_name]

    def match_term_across_ontologies(self,
                                     query_term: str,
                                     ontologies: Optional[List[str]] = None,
                                     top_k_per_ontology: int = 5,
                                     min_similarity: float = 0.3) -> Dict[str, List[TermMatch]]:
        """
        Match a term across multiple ontologies

        Args:
            query_term: Term to match
            ontologies: List of ontology names (None = all loaded)
            top_k_per_ontology: Top matches per ontology
            min_similarity: Minimum similarity threshold

        Returns:
            Dictionary mapping ontology name to list of matches
        """
        if ontologies is None:
            ontologies = [name for name, loaded in self._loaded.items() if loaded]

        results = {}
        for ontology_name in ontologies:
            api = self.get_api(ontology_name)
            matches = api.match_term_to_nodes(
                query_term,
                top_k=top_k_per_ontology,
                min_similarity=min_similarity
            )
            if matches:
                results[ontology_name] = matches

        return results

    def batch_match_across_ontologies(self,
                                      terms: List[str],
                                      ontologies: Optional[List[str]] = None,
                                      top_k_per_ontology: int = 3,
                                      min_similarity: float = 0.3) -> Dict[str, Dict[str, List[TermMatch]]]:
        """
        Match multiple terms across ontologies

        Args:
            terms: List of terms to match
            ontologies: List of ontology names (None = all loaded)
            top_k_per_ontology: Top matches per ontology per term
            min_similarity: Minimum similarity threshold

        Returns:
            Nested dictionary: term -> ontology -> matches
        """
        results = {}
        for term in terms:
            results[term] = self.match_term_across_ontologies(
                term,
                ontologies,
                top_k_per_ontology,
                min_similarity
            )
        return results

    def search_all_ontologies(self,
                             name: str,
                             exact: bool = False,
                             ontologies: Optional[List[str]] = None) -> Dict[str, List[OntologyNode]]:
        """
        Search by name across all ontologies

        Args:
            name: Name to search for
            exact: Exact match vs partial match
            ontologies: List of ontology names (None = all loaded)

        Returns:
            Dictionary mapping ontology name to list of matching nodes
        """
        if ontologies is None:
            ontologies = [name for name, loaded in self._loaded.items() if loaded]

        results = {}
        for ontology_name in ontologies:
            api = self.get_api(ontology_name)
            matches = api.search_by_name(name, exact=exact)
            if matches:
                results[ontology_name] = matches

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all loaded ontologies"""
        stats = {
            'loaded_ontologies': [
                name for name, loaded in self._loaded.items() if loaded
            ],
            'ontology_stats': {}
        }

        for name, loaded in self._loaded.items():
            if loaded:
                api = self._apis[name]
                stats['ontology_stats'][name] = api.get_stats()

        return stats

    def get_best_match_across_ontologies(self,
                                         query_term: str,
                                         ontologies: Optional[List[str]] = None,
                                         min_similarity: float = 0.3) -> Optional[TermMatch]:
        """
        Get the single best match across all ontologies

        Args:
            query_term: Term to match
            ontologies: List of ontology names (None = all loaded)
            min_similarity: Minimum similarity threshold

        Returns:
            Best TermMatch or None
        """
        all_matches = self.match_term_across_ontologies(
            query_term,
            ontologies,
            top_k_per_ontology=1,
            min_similarity=min_similarity
        )

        best_match = None
        best_score = 0.0

        for ontology_name, matches in all_matches.items():
            if matches and matches[0].similarity_score > best_score:
                best_score = matches[0].similarity_score
                best_match = matches[0]

        return best_match
