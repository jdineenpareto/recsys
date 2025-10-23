"""
ESCO (European Skills, Competences, Qualifications and Occupations) API

Provides access to ESCO taxonomy with:
- Skills, occupations, and qualifications retrieval
- Hierarchical navigation (broader/narrower concepts)
- RAG-based skill matching for job requirements
"""

import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from rdflib import Graph, Namespace, RDF, SKOS
from .base_ontology import (
    BaseOntologyAPI,
    OntologyNode,
    OntologyRelation,
    TermMatch
)


class ESCOAPI(BaseOntologyAPI):
    """API for ESCO taxonomy"""

    def __init__(self, ontology_path: str = None):
        """
        Initialize ESCO API

        Args:
            ontology_path: Path to ESCO RDF file (optional)
        """
        if ontology_path is None:
            base_dir = Path(__file__).parent.parent / "esco"
            ontology_path = str(base_dir / "esco-v1.2.0.ttl")

        super().__init__(ontology_path)

        # ESCO-specific structures
        self._concepts: Dict[str, Dict] = {}
        self._relations: List[OntologyRelation] = []
        self._concept_types: Dict[str, List[str]] = {
            'skill': [],
            'occupation': [],
            'qualification': [],
            'concept': []
        }

        # Load on initialization
        self.load_ontology()

    def load_ontology(self) -> None:
        """Load ESCO RDF file (supports both .ttl and .rdf formats with pickle caching)"""
        import pickle
        import os

        # Check for pickle cache
        cache_path = self.ontology_path + '.cache.pkl'

        if os.path.exists(cache_path):
            # Load from cache
            cache_mtime = os.path.getmtime(cache_path)
            source_mtime = os.path.getmtime(self.ontology_path)

            if cache_mtime > source_mtime:
                print(f"Loading ESCO from cache ({cache_path})...")
                try:
                    with open(cache_path, 'rb') as f:
                        cache_data = pickle.load(f)
                        self._concepts = cache_data['concepts']
                        self._relations = cache_data['relations']
                        self._concept_types = cache_data['concept_types']
                    print(f"Loaded {len(self._concepts)} ESCO concepts from cache (instant)")
                    for ctype, concepts in self._concept_types.items():
                        if concepts:
                            print(f"  {ctype.capitalize()}: {len(concepts)}")
                    return
                except Exception as e:
                    print(f"Cache load failed: {e}, parsing from source...")

        # Parse from source
        print(f"Loading ESCO from {self.ontology_path}...")
        print("(This will take 1-2 minutes on first load, then cache for instant loading)")

        # Detect file format
        if self.ontology_path.endswith('.ttl'):
            self._parse_turtle()
        else:
            # Fall back to XML parser for .rdf files
            with open(self.ontology_path, 'r', encoding='utf-8') as f:
                self._parse_rdf(f)

        print(f"Loaded {len(self._concepts)} ESCO concepts")
        for ctype, concepts in self._concept_types.items():
            if concepts:
                print(f"  {ctype.capitalize()}: {len(concepts)}")

        # Save to cache
        print(f"Saving cache to {cache_path}...")
        try:
            cache_data = {
                'concepts': self._concepts,
                'relations': self._relations,
                'concept_types': self._concept_types
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("Cache saved successfully!")
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

    def _parse_turtle(self) -> None:
        """Parse ESCO Turtle (.ttl) format using rdflib"""
        print("Parsing Turtle format (this may take 1-2 minutes for large files)...")
        import time
        start_time = time.time()

        # Create RDF graph
        g = Graph()
        print("  Loading file into memory...")
        g.parse(self.ontology_path, format='turtle')
        print(f"  Loaded in {time.time() - start_time:.1f}s")

        # Define namespaces
        SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

        # Pre-build lookup dictionaries for faster access
        print("  Building concept index...")
        concepts = list(g.subjects(RDF.type, SKOS.Concept))
        print(f"  Found {len(concepts)} concepts")

        # Batch process concepts
        print("  Extracting concept data...")
        concept_count = 0
        for concept_uri in concepts:
            self._parse_concept_from_graph(g, concept_uri, SKOS)
            concept_count += 1

            # Progress indicator
            if concept_count % 10000 == 0:
                elapsed = time.time() - start_time
                rate = concept_count / elapsed
                remaining = (len(concepts) - concept_count) / rate
                print(f"  Progress: {concept_count}/{len(concepts)} ({concept_count*100//len(concepts)}%) - ETA: {remaining:.0f}s")

        elapsed = time.time() - start_time
        print(f"Finished parsing {concept_count} concepts in {elapsed:.1f}s ({concept_count/elapsed:.0f} concepts/sec)")

    def _parse_concept_from_graph(self, graph: Graph, concept_uri, SKOS) -> None:
        """Parse a single SKOS concept from RDF graph"""
        concept_data = {'id': str(concept_uri)}

        # Get preferred label (prefLabel)
        pref_labels = list(graph.objects(concept_uri, SKOS.prefLabel))
        if pref_labels:
            # Prefer English label, or first available
            for label in pref_labels:
                if label.language == 'en' or label.language is None:
                    concept_data['name'] = str(label)
                    break
            if 'name' not in concept_data and pref_labels:
                concept_data['name'] = str(pref_labels[0])

        # Get definition
        definitions = list(graph.objects(concept_uri, SKOS.definition))
        if definitions:
            for definition in definitions:
                if hasattr(definition, 'language'):
                    if definition.language == 'en' or definition.language is None:
                        concept_data['definition'] = str(definition)
                        break
                else:
                    # URIRef or other non-literal
                    concept_data['definition'] = str(definition)
                    break

        # Get alternative labels (synonyms)
        alt_labels = [str(label) for label in graph.objects(concept_uri, SKOS.altLabel)
                     if hasattr(label, 'language') and (label.language == 'en' or label.language is None)]
        if alt_labels:
            concept_data['synonyms'] = alt_labels

        # Get broader concepts (parent relationships)
        broader_concepts = [str(broader) for broader in graph.objects(concept_uri, SKOS.broader)]
        if broader_concepts:
            concept_data['broader'] = broader_concepts

        # Get narrower concepts (child relationships)
        narrower_concepts = [str(narrower) for narrower in graph.objects(concept_uri, SKOS.narrower)]
        if narrower_concepts:
            concept_data['narrower'] = narrower_concepts

        # Determine concept type from URI
        concept_type = self._determine_concept_type_from_uri(str(concept_uri))
        concept_data['concept_type'] = concept_type

        self._add_concept(concept_data)

    def _determine_concept_type_from_uri(self, uri: str) -> str:
        """Determine if concept is a skill, occupation, qualification, or generic concept from URI"""
        if '/skill/' in uri:
            return 'skill'
        elif '/occupation/' in uri:
            return 'occupation'
        elif '/qualification/' in uri:
            return 'qualification'
        return 'concept'

    def _parse_rdf(self, file_handle) -> None:
        """Parse ESCO RDF/XML format"""
        try:
            tree = ET.parse(file_handle)
            root = tree.getroot()

            # Define namespaces
            namespaces = {
                'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
                'skos': 'http://www.w3.org/2004/02/skos/core#',
                'dc': 'http://purl.org/dc/terms/'
            }

            # Parse SKOS concepts
            for concept in root.findall('.//skos:Concept', namespaces):
                self._parse_concept(concept, namespaces)

        except ET.ParseError as e:
            print(f"Warning: Could not parse RDF file: {e}")
            print("Creating minimal ESCO API structure...")
            # Create a minimal structure for demonstration
            self._create_minimal_structure()

    def _parse_concept(self, concept_elem, namespaces: Dict[str, str]) -> None:
        """Parse a single SKOS concept"""
        # Get concept URI
        about = concept_elem.get(f'{{{namespaces["rdf"]}}}about')
        if not about:
            return

        concept_data = {'id': about}

        # Get preferred label
        pref_label = concept_elem.find('skos:prefLabel', namespaces)
        if pref_label is not None:
            concept_data['name'] = pref_label.text

        # Get definition
        definition = concept_elem.find('skos:definition', namespaces)
        if definition is not None:
            concept_data['definition'] = definition.text

        # Get alternative labels (synonyms)
        alt_labels = concept_elem.findall('skos:altLabel', namespaces)
        if alt_labels:
            concept_data['synonyms'] = [label.text for label in alt_labels if label.text]

        # Get broader concepts (parent relationships)
        broader = concept_elem.findall('skos:broader', namespaces)
        if broader:
            concept_data['broader'] = [
                b.get(f'{{{namespaces["rdf"]}}}resource')
                for b in broader
                if b.get(f'{{{namespaces["rdf"]}}}resource')
            ]

        # Get narrower concepts (child relationships)
        narrower = concept_elem.findall('skos:narrower', namespaces)
        if narrower:
            concept_data['narrower'] = [
                n.get(f'{{{namespaces["rdf"]}}}resource')
                for n in narrower
                if n.get(f'{{{namespaces["rdf"]}}}resource')
            ]

        # Determine concept type from URI or rdf:type
        concept_type = self._determine_concept_type(about, concept_elem, namespaces)
        concept_data['concept_type'] = concept_type

        self._add_concept(concept_data)

    def _determine_concept_type(self, uri: str, concept_elem, namespaces: Dict[str, str]) -> str:
        """Determine if concept is a skill, occupation, qualification, or generic concept"""
        # Check rdf:type
        rdf_type = concept_elem.find('rdf:type', namespaces)
        if rdf_type is not None:
            type_uri = rdf_type.get(f'{{{namespaces["rdf"]}}}resource', '')
            if 'Skill' in type_uri:
                return 'skill'
            elif 'Occupation' in type_uri:
                return 'occupation'
            elif 'Qualification' in type_uri:
                return 'qualification'

        # Infer from URI structure
        if '/skill/' in uri:
            return 'skill'
        elif '/occupation/' in uri:
            return 'occupation'
        elif '/qualification/' in uri:
            return 'qualification'

        return 'concept'

    def _create_minimal_structure(self) -> None:
        """Create a minimal ESCO structure for demonstration purposes"""
        # This is a fallback if the RDF parsing fails
        sample_concepts = [
            {
                'id': 'http://data.europa.eu/esco/skill/S1',
                'name': 'Communication Skills',
                'definition': 'Ability to convey information effectively',
                'concept_type': 'skill',
                'synonyms': ['Communication abilities', 'Verbal communication']
            },
            {
                'id': 'http://data.europa.eu/esco/skill/S2',
                'name': 'Technical Skills',
                'definition': 'Competence in technical domains',
                'concept_type': 'skill',
                'synonyms': ['Technical competence', 'Technical abilities']
            }
        ]

        for concept_data in sample_concepts:
            self._add_concept(concept_data)

    def _add_concept(self, concept_data: Dict) -> None:
        """Add a parsed concept to internal structures"""
        concept_id = concept_data.get('id')
        if not concept_id:
            return

        # Store concept
        self._concepts[concept_id] = concept_data

        # Index by type
        concept_type = concept_data.get('concept_type', 'concept')
        if concept_type in self._concept_types:
            self._concept_types[concept_type].append(concept_id)

        # Create broader relationships
        if 'broader' in concept_data:
            for parent_id in concept_data['broader']:
                self._relations.append(OntologyRelation(
                    source_id=concept_id,
                    target_id=parent_id,
                    relation_type='broader'
                ))

        # Create narrower relationships
        if 'narrower' in concept_data:
            for child_id in concept_data['narrower']:
                self._relations.append(OntologyRelation(
                    source_id=concept_id,
                    target_id=child_id,
                    relation_type='narrower'
                ))

    def get_node(self, node_id: str) -> Optional[OntologyNode]:
        """Get an ESCO concept by ID"""
        if node_id in self._node_cache:
            return self._node_cache[node_id]

        concept_data = self._concepts.get(node_id)
        if not concept_data:
            return None

        node = OntologyNode(
            id=node_id,
            name=concept_data.get('name', ''),
            definition=concept_data.get('definition'),
            synonyms=concept_data.get('synonyms', []),
            metadata={
                'concept_type': concept_data.get('concept_type', 'concept')
            }
        )

        self._node_cache[node_id] = node
        return node

    def get_neighbors(self,
                     node_id: str,
                     relation_types: Optional[List[str]] = None,
                     direction: str = 'both') -> List[Tuple[OntologyNode, str]]:
        """Get neighboring ESCO concepts"""
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
        """Search for ESCO concepts by name"""
        results = []
        name_lower = name.lower()

        for concept_id, concept_data in self._concepts.items():
            concept_name = concept_data.get('name', '').lower()

            if exact:
                if concept_name == name_lower:
                    node = self.get_node(concept_id)
                    if node:
                        results.append(node)
            else:
                if name_lower in concept_name:
                    node = self.get_node(concept_id)
                    if node:
                        results.append(node)

        return results

    def get_all_nodes(self) -> List[OntologyNode]:
        """Get all ESCO concepts"""
        if self._all_nodes is None:
            self._all_nodes = [
                self.get_node(concept_id)
                for concept_id in self._concepts.keys()
            ]
            self._all_nodes = [n for n in self._all_nodes if n is not None]

        return self._all_nodes

    def get_skills(self) -> List[OntologyNode]:
        """Get all skill concepts"""
        return [
            self.get_node(concept_id)
            for concept_id in self._concept_types['skill']
            if self.get_node(concept_id) is not None
        ]

    def get_occupations(self) -> List[OntologyNode]:
        """Get all occupation concepts"""
        return [
            self.get_node(concept_id)
            for concept_id in self._concept_types['occupation']
            if self.get_node(concept_id) is not None
        ]

    def get_qualifications(self) -> List[OntologyNode]:
        """Get all qualification concepts"""
        return [
            self.get_node(concept_id)
            for concept_id in self._concept_types['qualification']
            if self.get_node(concept_id) is not None
        ]

    def get_broader_concepts(self, node_id: str) -> List[OntologyNode]:
        """Get more general (parent) concepts"""
        return [node for node, _ in self.get_neighbors(node_id, ['broader'], 'outgoing')]

    def get_narrower_concepts(self, node_id: str) -> List[OntologyNode]:
        """Get more specific (child) concepts"""
        return [node for node, _ in self.get_neighbors(node_id, ['narrower'], 'outgoing')]
