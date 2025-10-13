"""
Gene Ontology (GO) API

Provides access to Gene Ontology terms with:
- Term retrieval and relationship navigation (is_a, part_of, regulates, etc.)
- RAG-based term matching
- GO-specific queries (biological process, molecular function, cellular component)
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


class GeneOntologyAPI(BaseOntologyAPI):
    """API for Gene Ontology (GO)"""

    def __init__(self, ontology_path: str = None, use_basic: bool = True):
        """
        Initialize GO API

        Args:
            ontology_path: Path to GO OBO file (optional)
            use_basic: Use go-basic.obo (True) or full go.obo (False)
        """
        if ontology_path is None:
            base_dir = Path(__file__).parent.parent / "gene_ontology"
            filename = "go-basic.obo" if use_basic else "go.obo"
            ontology_path = str(base_dir / filename)

        super().__init__(ontology_path)

        # GO-specific structures
        self._terms: Dict[str, Dict] = {}
        self._relations: List[OntologyRelation] = []
        self._namespace_map: Dict[str, List[str]] = {
            'biological_process': [],
            'molecular_function': [],
            'cellular_component': []
        }

        # Load on initialization
        self.load_ontology()

    def load_ontology(self) -> None:
        """Load GO OBO file"""
        print(f"Loading Gene Ontology from {self.ontology_path}...")

        with open(self.ontology_path, 'r', encoding='utf-8') as f:
            self._parse_obo(f)

        print(f"Loaded {len(self._terms)} GO terms")
        print(f"  Biological Process: {len(self._namespace_map['biological_process'])}")
        print(f"  Molecular Function: {len(self._namespace_map['molecular_function'])}")
        print(f"  Cellular Component: {len(self._namespace_map['cellular_component'])}")

    def _parse_obo(self, file_handle) -> None:
        """Parse OBO format file"""
        current_term = None

        for line in file_handle:
            line = line.strip()

            if line == "[Term]":
                if current_term:
                    self._add_term(current_term)
                current_term = {}
            elif line.startswith("id:"):
                if current_term is not None:
                    current_term['id'] = line.split("id:", 1)[1].strip()
            elif line.startswith("name:"):
                if current_term is not None:
                    current_term['name'] = line.split("name:", 1)[1].strip()
            elif line.startswith("namespace:"):
                if current_term is not None:
                    current_term['namespace'] = line.split("namespace:", 1)[1].strip()
            elif line.startswith("def:"):
                if current_term is not None:
                    # Extract definition text (in quotes)
                    match = re.search(r'"([^"]*)"', line)
                    if match:
                        current_term['definition'] = match.group(1)
            elif line.startswith("synonym:"):
                if current_term is not None:
                    # Extract synonym text (in quotes)
                    match = re.search(r'"([^"]*)"', line)
                    if match:
                        if 'synonyms' not in current_term:
                            current_term['synonyms'] = []
                        current_term['synonyms'].append(match.group(1))
            elif line.startswith("is_a:"):
                if current_term is not None:
                    target_id = line.split("is_a:", 1)[1].strip().split()[0]
                    if 'is_a' not in current_term:
                        current_term['is_a'] = []
                    current_term['is_a'].append(target_id)
            elif line.startswith("relationship:"):
                if current_term is not None:
                    parts = line.split("relationship:", 1)[1].strip().split()
                    if len(parts) >= 2:
                        rel_type = parts[0]
                        target_id = parts[1]
                        if 'relationships' not in current_term:
                            current_term['relationships'] = []
                        current_term['relationships'].append((rel_type, target_id))
            elif line.startswith("is_obsolete:"):
                if current_term is not None:
                    current_term['obsolete'] = True

        # Add last term
        if current_term:
            self._add_term(current_term)

    def _add_term(self, term_data: Dict) -> None:
        """Add a parsed term to internal structures"""
        # Skip obsolete terms
        if term_data.get('obsolete', False):
            return

        term_id = term_data.get('id')
        if not term_id:
            return

        # Store term
        self._terms[term_id] = term_data

        # Index by namespace
        namespace = term_data.get('namespace', 'unknown')
        if namespace in self._namespace_map:
            self._namespace_map[namespace].append(term_id)

        # Create relations
        if 'is_a' in term_data:
            for parent_id in term_data['is_a']:
                self._relations.append(OntologyRelation(
                    source_id=term_id,
                    target_id=parent_id,
                    relation_type='is_a'
                ))

        if 'relationships' in term_data:
            for rel_type, target_id in term_data['relationships']:
                self._relations.append(OntologyRelation(
                    source_id=term_id,
                    target_id=target_id,
                    relation_type=rel_type
                ))

    def get_node(self, node_id: str) -> Optional[OntologyNode]:
        """Get a GO term by ID"""
        if node_id in self._node_cache:
            return self._node_cache[node_id]

        term_data = self._terms.get(node_id)
        if not term_data:
            return None

        node = OntologyNode(
            id=node_id,
            name=term_data.get('name', ''),
            definition=term_data.get('definition'),
            synonyms=term_data.get('synonyms', []),
            metadata={
                'namespace': term_data.get('namespace', 'unknown')
            }
        )

        self._node_cache[node_id] = node
        return node

    def get_neighbors(self,
                     node_id: str,
                     relation_types: Optional[List[str]] = None,
                     direction: str = 'both') -> List[Tuple[OntologyNode, str]]:
        """Get neighboring GO terms"""
        neighbors = []

        for relation in self._relations:
            include_relation = False

            # Check direction
            if direction in ('outgoing', 'both') and relation.source_id == node_id:
                target_id = relation.target_id
                include_relation = True
            elif direction in ('incoming', 'both') and relation.target_id == node_id:
                target_id = relation.source_id
                include_relation = True
            else:
                continue

            # Check relation type filter
            if relation_types and relation.relation_type not in relation_types:
                continue

            if include_relation:
                neighbor_node = self.get_node(target_id)
                if neighbor_node:
                    neighbors.append((neighbor_node, relation.relation_type))

        return neighbors

    def search_by_name(self, name: str, exact: bool = False) -> List[OntologyNode]:
        """Search for GO terms by name"""
        results = []
        name_lower = name.lower()

        for term_id, term_data in self._terms.items():
            term_name = term_data.get('name', '').lower()

            if exact:
                if term_name == name_lower:
                    node = self.get_node(term_id)
                    if node:
                        results.append(node)
            else:
                if name_lower in term_name:
                    node = self.get_node(term_id)
                    if node:
                        results.append(node)

        return results

    def get_all_nodes(self) -> List[OntologyNode]:
        """Get all GO terms"""
        if self._all_nodes is None:
            self._all_nodes = [
                self.get_node(term_id)
                for term_id in self._terms.keys()
            ]
            self._all_nodes = [n for n in self._all_nodes if n is not None]

        return self._all_nodes

    def get_terms_by_namespace(self, namespace: str) -> List[OntologyNode]:
        """
        Get all terms in a specific GO namespace

        Args:
            namespace: 'biological_process', 'molecular_function', or 'cellular_component'
        """
        if namespace not in self._namespace_map:
            return []

        return [
            self.get_node(term_id)
            for term_id in self._namespace_map[namespace]
            if self.get_node(term_id) is not None
        ]

    def get_ancestors(self, node_id: str, relation_types: Optional[List[str]] = None) -> List[OntologyNode]:
        """
        Get all ancestor terms (parents, grandparents, etc.)

        Args:
            node_id: GO term ID
            relation_types: Limit to specific relations (default: ['is_a'])
        """
        if relation_types is None:
            relation_types = ['is_a']

        ancestors = []
        visited = set()
        queue = [node_id]

        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)

            # Get parents
            for neighbor, rel_type in self.get_neighbors(current_id, relation_types, 'outgoing'):
                if neighbor.id not in visited:
                    ancestors.append(neighbor)
                    queue.append(neighbor.id)

        return ancestors

    def get_descendants(self, node_id: str, relation_types: Optional[List[str]] = None) -> List[OntologyNode]:
        """
        Get all descendant terms (children, grandchildren, etc.)

        Args:
            node_id: GO term ID
            relation_types: Limit to specific relations (default: ['is_a'])
        """
        if relation_types is None:
            relation_types = ['is_a']

        descendants = []
        visited = set()
        queue = [node_id]

        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)

            # Get children
            for neighbor, rel_type in self.get_neighbors(current_id, relation_types, 'incoming'):
                if neighbor.id not in visited:
                    descendants.append(neighbor)
                    queue.append(neighbor.id)

        return descendants
