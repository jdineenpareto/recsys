"""
Cell Ontology (CL) API

Provides access to cell types with:
- Cell type retrieval and relationship navigation
- RAG-based cell type matching
- Cell lineage and development queries
"""

from pathlib import Path
from .go_api import GeneOntologyAPI  # Reuse OBO parser


class CellOntologyAPI(GeneOntologyAPI):
    """
    API for Cell Ontology (CL)

    Inherits from GeneOntologyAPI since both use OBO format
    """

    def __init__(self, ontology_path: str = None):
        """
        Initialize Cell Ontology API

        Args:
            ontology_path: Path to CL OBO file (optional)
        """
        if ontology_path is None:
            base_dir = Path(__file__).parent.parent / "cell_ontology"
            ontology_path = str(base_dir / "cl.obo")

        # Initialize with Cell Ontology path
        # Use parent's parent __init__ to avoid GO-specific initialization
        from .base_ontology import BaseOntologyAPI
        BaseOntologyAPI.__init__(self, ontology_path)

        # Cell Ontology-specific structures
        self._cell_types: dict[str, dict] = {}
        self._relations: list = []

        # Replace GO namespaces with CL-specific categories
        self._namespace_map: dict[str, list[str]] = {
            'cell': [],
            'native_cell': [],
            'stuff': []
        }

        # Use the parent attribute name for compatibility
        self._terms = self._cell_types

        # Load on initialization
        self.load_ontology()

    def load_ontology(self) -> None:
        """Load Cell Ontology OBO file"""
        print(f"Loading Cell Ontology from {self.ontology_path}...")

        with open(self.ontology_path, 'r', encoding='utf-8') as f:
            self._parse_obo(f)

        print(f"Loaded {len(self._cell_types)} cell types")
        for namespace, terms in self._namespace_map.items():
            if terms:
                print(f"  {namespace}: {len(terms)}")

    def get_cell_lineage(self, node_id: str) -> list:
        """
        Get the developmental lineage of a cell type (all ancestors)

        Args:
            node_id: Cell type ID

        Returns:
            List of ancestor cell types in order from specific to general
        """
        return self.get_ancestors(node_id, relation_types=['is_a', 'develops_from'])

    def get_derived_cell_types(self, node_id: str) -> list:
        """
        Get cell types that develop from this cell type

        Args:
            node_id: Cell type ID

        Returns:
            List of derived cell types
        """
        neighbors = self.get_neighbors(
            node_id,
            relation_types=['develops_from'],
            direction='incoming'
        )
        return [node for node, _ in neighbors]

    def get_cell_parts(self, node_id: str) -> list:
        """
        Get parts/components of a cell type

        Args:
            node_id: Cell type ID

        Returns:
            List of cell parts
        """
        neighbors = self.get_neighbors(
            node_id,
            relation_types=['has_part'],
            direction='outgoing'
        )
        return [node for node, _ in neighbors]
