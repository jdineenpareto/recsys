"""
ChEBI (Chemical Entities of Biological Interest) API

Provides access to chemical compounds with:
- Chemical entity retrieval and relationship navigation
- RAG-based chemical name/synonym matching
- ChEBI-specific queries (molecular formulas, charges, roles, etc.)
"""

import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from .base_ontology import (
    BaseOntologyAPI,
    OntologyNode,
    OntologyRelation,
    TermMatch
)


class ChEBIAPI(BaseOntologyAPI):
    """API for ChEBI chemical ontology"""

    def __init__(self, ontology_path: str = None):
        """
        Initialize ChEBI API

        Args:
            ontology_path: Path to ChEBI OBO file (optional)
        """
        if ontology_path is None:
            base_dir = Path(__file__).parent.parent / "chebi"
            ontology_path = str(base_dir / "chebi.obo")

        super().__init__(ontology_path)

        # ChEBI-specific structures
        self._compounds: Dict[str, Dict] = {}
        self._relations: List[OntologyRelation] = []
        self._formula_index: Dict[str, List[str]] = {}  # formula -> compound IDs

        # Load on initialization
        self.load_ontology()

    def load_ontology(self) -> None:
        """Load ChEBI OBO file"""
        print(f"Loading ChEBI from {self.ontology_path}...")
        print("Note: ChEBI is large (~250MB), this may take a minute...")

        with open(self.ontology_path, 'r', encoding='utf-8') as f:
            self._parse_obo(f)

        print(f"Loaded {len(self._compounds)} ChEBI compounds")
        print(f"  Formulas indexed: {len(self._formula_index)}")

    def _parse_obo(self, file_handle) -> None:
        """Parse ChEBI OBO format file"""
        current_compound = None

        for line in file_handle:
            line = line.strip()

            if line == "[Term]":
                if current_compound:
                    self._add_compound(current_compound)
                current_compound = {}
            elif line.startswith("id:"):
                if current_compound is not None:
                    current_compound['id'] = line.split("id:", 1)[1].strip()
            elif line.startswith("name:"):
                if current_compound is not None:
                    current_compound['name'] = line.split("name:", 1)[1].strip()
            elif line.startswith("def:"):
                if current_compound is not None:
                    match = re.search(r'"([^"]*)"', line)
                    if match:
                        current_compound['definition'] = match.group(1)
            elif line.startswith("synonym:"):
                if current_compound is not None:
                    match = re.search(r'"([^"]*)"', line)
                    if match:
                        if 'synonyms' not in current_compound:
                            current_compound['synonyms'] = []
                        current_compound['synonyms'].append(match.group(1))
            elif line.startswith("property_value: http://purl.obolibrary.org/obo/chebi/formula"):
                if current_compound is not None:
                    # Extract formula
                    match = re.search(r'"([^"]*)"', line)
                    if match:
                        current_compound['formula'] = match.group(1)
            elif line.startswith("property_value: http://purl.obolibrary.org/obo/chebi/charge"):
                if current_compound is not None:
                    # Extract charge
                    match = re.search(r'"([^"]*)"', line)
                    if match:
                        current_compound['charge'] = match.group(1)
            elif line.startswith("property_value: http://purl.obolibrary.org/obo/chebi/mass"):
                if current_compound is not None:
                    # Extract mass
                    match = re.search(r'"([^"]*)"', line)
                    if match:
                        current_compound['mass'] = match.group(1)
            elif line.startswith("property_value: http://purl.obolibrary.org/obo/chebi/inchi "):
                if current_compound is not None:
                    # Extract InChI
                    match = re.search(r'"([^"]*)"', line)
                    if match:
                        current_compound['inchi'] = match.group(1)
            elif line.startswith("property_value: http://purl.obolibrary.org/obo/chebi/smiles"):
                if current_compound is not None:
                    # Extract SMILES
                    match = re.search(r'"([^"]*)"', line)
                    if match:
                        current_compound['smiles'] = match.group(1)
            elif line.startswith("is_a:"):
                if current_compound is not None:
                    target_id = line.split("is_a:", 1)[1].strip().split()[0]
                    if 'is_a' not in current_compound:
                        current_compound['is_a'] = []
                    current_compound['is_a'].append(target_id)
            elif line.startswith("relationship:"):
                if current_compound is not None:
                    parts = line.split("relationship:", 1)[1].strip().split()
                    if len(parts) >= 2:
                        rel_type = parts[0]
                        target_id = parts[1]
                        if 'relationships' not in current_compound:
                            current_compound['relationships'] = []
                        current_compound['relationships'].append((rel_type, target_id))
            elif line.startswith("is_obsolete:"):
                if current_compound is not None:
                    current_compound['obsolete'] = True

        # Add last compound
        if current_compound:
            self._add_compound(current_compound)

    def _add_compound(self, compound_data: Dict) -> None:
        """Add a parsed compound to internal structures"""
        # Skip obsolete entries
        if compound_data.get('obsolete', False):
            return

        compound_id = compound_data.get('id')
        if not compound_id:
            return

        # Store compound
        self._compounds[compound_id] = compound_data

        # Index by formula
        formula = compound_data.get('formula')
        if formula:
            if formula not in self._formula_index:
                self._formula_index[formula] = []
            self._formula_index[formula].append(compound_id)

        # Create relations
        if 'is_a' in compound_data:
            for parent_id in compound_data['is_a']:
                self._relations.append(OntologyRelation(
                    source_id=compound_id,
                    target_id=parent_id,
                    relation_type='is_a'
                ))

        if 'relationships' in compound_data:
            for rel_type, target_id in compound_data['relationships']:
                self._relations.append(OntologyRelation(
                    source_id=compound_id,
                    target_id=target_id,
                    relation_type=rel_type
                ))

    def get_node(self, node_id: str) -> Optional[OntologyNode]:
        """Get a ChEBI compound by ID"""
        if node_id in self._node_cache:
            return self._node_cache[node_id]

        compound_data = self._compounds.get(node_id)
        if not compound_data:
            return None

        # Build metadata
        metadata = {}
        for key in ['formula', 'charge', 'mass', 'inchi', 'smiles']:
            if key in compound_data:
                metadata[key] = compound_data[key]

        node = OntologyNode(
            id=node_id,
            name=compound_data.get('name', ''),
            definition=compound_data.get('definition'),
            synonyms=compound_data.get('synonyms', []),
            metadata=metadata
        )

        self._node_cache[node_id] = node
        return node

    def get_neighbors(self,
                     node_id: str,
                     relation_types: Optional[List[str]] = None,
                     direction: str = 'both') -> List[Tuple[OntologyNode, str]]:
        """Get neighboring chemical entities"""
        neighbors = []

        for relation in self._relations:
            include_relation = False

            if direction in ('outgoing', 'both') and relation.source_id == node_id:
                target_id = relation.target_id
                include_relation = True
            elif direction in ('incoming', 'both') and relation.target_id == node_id:
                target_id = relation.source_id
                include_relation = True
            else:
                continue

            if relation_types and relation.relation_type not in relation_types:
                continue

            if include_relation:
                neighbor_node = self.get_node(target_id)
                if neighbor_node:
                    neighbors.append((neighbor_node, relation.relation_type))

        return neighbors

    def search_by_name(self, name: str, exact: bool = False) -> List[OntologyNode]:
        """Search for compounds by name"""
        results = []
        name_lower = name.lower()

        for compound_id, compound_data in self._compounds.items():
            compound_name = compound_data.get('name', '').lower()

            if exact:
                if compound_name == name_lower:
                    node = self.get_node(compound_id)
                    if node:
                        results.append(node)
            else:
                if name_lower in compound_name:
                    node = self.get_node(compound_id)
                    if node:
                        results.append(node)

        return results

    def get_all_nodes(self) -> List[OntologyNode]:
        """Get all ChEBI compounds"""
        if self._all_nodes is None:
            self._all_nodes = [
                self.get_node(compound_id)
                for compound_id in self._compounds.keys()
            ]
            self._all_nodes = [n for n in self._all_nodes if n is not None]

        return self._all_nodes

    def search_by_formula(self, formula: str) -> List[OntologyNode]:
        """
        Search for compounds by molecular formula

        Args:
            formula: Molecular formula (e.g., "H2O", "C6H12O6")
        """
        compound_ids = self._formula_index.get(formula, [])
        return [self.get_node(cid) for cid in compound_ids if self.get_node(cid)]

    def get_compound_roles(self, node_id: str) -> List[OntologyNode]:
        """
        Get the roles/applications of a compound

        Args:
            node_id: ChEBI compound ID

        Returns:
            List of role terms (e.g., "drug", "metabolite", "toxin")
        """
        # Roles are typically indicated by 'has_role' relationships
        neighbors = self.get_neighbors(
            node_id,
            relation_types=['has_role'],
            direction='outgoing'
        )
        return [node for node, _ in neighbors]

    def get_parent_compounds(self, node_id: str) -> List[OntologyNode]:
        """Get parent (more general) chemical entities"""
        return [node for node, _ in self.get_neighbors(node_id, ['is_a'], 'outgoing')]

    def get_child_compounds(self, node_id: str) -> List[OntologyNode]:
        """Get child (more specific) chemical entities"""
        return [node for node, _ in self.get_neighbors(node_id, ['is_a'], 'incoming')]
