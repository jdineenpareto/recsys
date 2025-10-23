"""
MeSH (Medical Subject Headings) API

Provides access to medical terms with:
- Medical term retrieval and relationship navigation
- RAG-based medical terminology matching
- MeSH tree structure navigation
"""

import xml.etree.ElementTree as ET
import gzip
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from .base_ontology import (
    BaseOntologyAPI,
    OntologyNode,
    OntologyRelation,
    TermMatch
)


class MeSHAPI(BaseOntologyAPI):
    """API for MeSH medical terminology"""

    def __init__(self, ontology_path: str = None, use_rdf: bool = False):
        """
        Initialize MeSH API

        Args:
            ontology_path: Path to MeSH file (optional)
            use_rdf: Use RDF N-Triples format (True) or XML descriptors (False)
        """
        if ontology_path is None:
            base_dir = Path(__file__).parent.parent / "mesh"
            if use_rdf:
                ontology_path = str(base_dir / "mesh.nt.gz")
            else:
                ontology_path = str(base_dir / "desc2025.xml")

        super().__init__(ontology_path)

        # MeSH-specific structures
        self._descriptors: Dict[str, Dict] = {}
        self._relations: List[OntologyRelation] = []
        self._tree_numbers: Dict[str, str] = {}  # tree_number -> descriptor_id
        self._use_rdf = use_rdf

        # Load on initialization
        self.load_ontology()

    def load_ontology(self) -> None:
        """Load MeSH file"""
        print(f"Loading MeSH from {self.ontology_path}...")

        if self._use_rdf:
            self._load_rdf()
        else:
            self._load_xml()

        print(f"Loaded {len(self._descriptors)} MeSH descriptors")

    def _load_xml(self) -> None:
        """Load MeSH XML descriptors"""
        try:
            tree = ET.parse(self.ontology_path)
            root = tree.getroot()

            for descriptor in root.findall('.//DescriptorRecord'):
                self._parse_descriptor_xml(descriptor)

        except ET.ParseError as e:
            print(f"Warning: Could not parse XML file: {e}")
            self._create_minimal_structure()

    def _parse_descriptor_xml(self, descriptor_elem) -> None:
        """Parse a MeSH descriptor from XML"""
        descriptor_data = {}

        # Get Descriptor UI (unique identifier)
        ui_elem = descriptor_elem.find('DescriptorUI')
        if ui_elem is not None:
            descriptor_data['id'] = ui_elem.text

        # Get Descriptor Name
        name_elem = descriptor_elem.find('DescriptorName/String')
        if name_elem is not None:
            descriptor_data['name'] = name_elem.text

        # Get tree numbers (hierarchical classification)
        tree_number_list = descriptor_elem.find('TreeNumberList')
        if tree_number_list is not None:
            tree_numbers = [
                tn.text for tn in tree_number_list.findall('TreeNumber')
                if tn.text
            ]
            if tree_numbers:
                descriptor_data['tree_numbers'] = tree_numbers
                # Index tree numbers
                for tn in tree_numbers:
                    self._tree_numbers[tn] = descriptor_data['id']

        # Get concepts (for definitions and synonyms)
        concept_list = descriptor_elem.find('ConceptList')
        if concept_list is not None:
            concepts = concept_list.findall('Concept')
            if concepts:
                # Use preferred concept
                for concept in concepts:
                    is_preferred = concept.get('PreferredConceptYN') == 'Y'
                    if is_preferred or not descriptor_data.get('definition'):
                        # Get scope note (definition)
                        scope_note = concept.find('ScopeNote')
                        if scope_note is not None:
                            descriptor_data['definition'] = scope_note.text

                        # Get terms (synonyms)
                        term_list = concept.find('TermList')
                        if term_list is not None:
                            terms = term_list.findall('Term/String')
                            if terms:
                                if 'synonyms' not in descriptor_data:
                                    descriptor_data['synonyms'] = []
                                descriptor_data['synonyms'].extend([
                                    t.text for t in terms if t.text
                                ])

        self._add_descriptor(descriptor_data)

    def _load_rdf(self) -> None:
        """Load MeSH RDF N-Triples (compressed)"""
        print("Note: RDF loading is basic. Use XML for full functionality.")

        try:
            with gzip.open(self.ontology_path, 'rt', encoding='utf-8') as f:
                # Parse N-Triples format (simple line-by-line)
                current_subject = None
                current_descriptor = {}

                for line_num, line in enumerate(f):
                    if line_num > 100000:  # Limit for performance
                        break

                    line = line.strip()
                    if not line:
                        continue

                    # Parse triple: <subject> <predicate> <object> .
                    parts = line.split(' ', 2)
                    if len(parts) < 3:
                        continue

                    subject = parts[0].strip('<>')
                    predicate = parts[1].strip('<>')
                    object_part = parts[2].rsplit(' .', 1)[0].strip()

                    # Extract descriptor info
                    if 'rdfs#label' in predicate:
                        # Extract label text
                        label = object_part.strip('"').split('"')[0]
                        if subject != current_subject:
                            if current_descriptor:
                                self._add_descriptor(current_descriptor)
                            current_subject = subject
                            current_descriptor = {
                                'id': subject,
                                'name': label
                            }

        except Exception as e:
            print(f"Warning: Could not load RDF file: {e}")
            self._create_minimal_structure()

    def _create_minimal_structure(self) -> None:
        """Create minimal MeSH structure for demonstration"""
        sample_descriptors = [
            {
                'id': 'D000001',
                'name': 'Calcimycin',
                'definition': 'An ionophorous, polyether antibiotic from Streptomyces chartreusensis.',
                'synonyms': ['A-23187', 'Antibiotic A23187']
            },
            {
                'id': 'D000002',
                'name': 'Temefos',
                'definition': 'An organothiophosphate cholinesterase inhibitor.',
                'synonyms': ['Abate', 'Temephos']
            }
        ]

        for descriptor_data in sample_descriptors:
            self._add_descriptor(descriptor_data)

    def _add_descriptor(self, descriptor_data: Dict) -> None:
        """Add a parsed descriptor to internal structures"""
        descriptor_id = descriptor_data.get('id')
        if not descriptor_id or not descriptor_data.get('name'):
            return

        # Store descriptor
        self._descriptors[descriptor_id] = descriptor_data

        # Create hierarchical relations from tree numbers
        if 'tree_numbers' in descriptor_data:
            for tree_num in descriptor_data['tree_numbers']:
                # Parent is tree number with last segment removed
                if '.' in tree_num:
                    parent_tree = '.'.join(tree_num.split('.')[:-1])
                    if parent_tree in self._tree_numbers:
                        parent_id = self._tree_numbers[parent_tree]
                        self._relations.append(OntologyRelation(
                            source_id=descriptor_id,
                            target_id=parent_id,
                            relation_type='broader'
                        ))

    def get_node(self, node_id: str) -> Optional[OntologyNode]:
        """Get a MeSH descriptor by ID"""
        if node_id in self._node_cache:
            return self._node_cache[node_id]

        descriptor_data = self._descriptors.get(node_id)
        if not descriptor_data:
            return None

        metadata = {}
        if 'tree_numbers' in descriptor_data:
            metadata['tree_numbers'] = descriptor_data['tree_numbers']

        node = OntologyNode(
            id=node_id,
            name=descriptor_data.get('name', ''),
            definition=descriptor_data.get('definition'),
            synonyms=descriptor_data.get('synonyms', []),
            metadata=metadata
        )

        self._node_cache[node_id] = node
        return node

    def get_neighbors(self,
                     node_id: str,
                     relation_types: Optional[List[str]] = None,
                     direction: str = 'both') -> List[Tuple[OntologyNode, str]]:
        """Get neighboring MeSH descriptors"""
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
        """Search for MeSH descriptors by name"""
        results = []
        name_lower = name.lower()

        for descriptor_id, descriptor_data in self._descriptors.items():
            descriptor_name = descriptor_data.get('name', '').lower()

            if exact:
                if descriptor_name == name_lower:
                    node = self.get_node(descriptor_id)
                    if node:
                        results.append(node)
            else:
                if name_lower in descriptor_name:
                    node = self.get_node(descriptor_id)
                    if node:
                        results.append(node)

        return results

    def get_all_nodes(self) -> List[OntologyNode]:
        """Get all MeSH descriptors"""
        if self._all_nodes is None:
            self._all_nodes = [
                self.get_node(descriptor_id)
                for descriptor_id in self._descriptors.keys()
            ]
            self._all_nodes = [n for n in self._all_nodes if n is not None]

        return self._all_nodes

    def get_descriptor_by_tree_number(self, tree_number: str) -> Optional[OntologyNode]:
        """
        Get MeSH descriptor by tree number

        Args:
            tree_number: MeSH tree number (e.g., "C01.539")
        """
        descriptor_id = self._tree_numbers.get(tree_number)
        if descriptor_id:
            return self.get_node(descriptor_id)
        return None

    def get_broader_terms(self, node_id: str) -> List[OntologyNode]:
        """Get more general (parent) terms"""
        return [node for node, _ in self.get_neighbors(node_id, ['broader'], 'outgoing')]

    def get_narrower_terms(self, node_id: str) -> List[OntologyNode]:
        """Get more specific (child) terms"""
        return [node for node, _ in self.get_neighbors(node_id, ['broader'], 'incoming')]
