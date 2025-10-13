"""
Base Ontology API Interface

Provides common functionality for all ontology APIs:
- Node retrieval and neighbor traversal
- RAG-based term matching using semantic embeddings
- Caching for performance optimization
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


@dataclass
class OntologyNode:
    """Represents a node in an ontology"""
    id: str
    name: str
    definition: Optional[str] = None
    synonyms: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.synonyms is None:
            self.synonyms = []
        if self.metadata is None:
            self.metadata = {}

    def get_searchable_text(self) -> str:
        """Get combined text for semantic search"""
        parts = [self.name]
        if self.definition:
            parts.append(self.definition)
        if self.synonyms:
            parts.extend(self.synonyms)
        return " | ".join(parts)


@dataclass
class OntologyRelation:
    """Represents a relationship between nodes"""
    source_id: str
    target_id: str
    relation_type: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TermMatch:
    """Represents a match between a query term and ontology node"""
    node: OntologyNode
    similarity_score: float
    match_type: str  # 'exact', 'synonym', 'semantic'
    matched_text: str


class BaseOntologyAPI(ABC):
    """Base class for all ontology APIs"""

    def __init__(self, ontology_path: str, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.ontology_path = ontology_path
        self.embedding_model_name = embedding_model
        self.embedding_model: Optional[SentenceTransformer] = None

        # Caches
        self._node_cache: Dict[str, OntologyNode] = {}
        self._neighbor_cache: Dict[str, List[str]] = {}
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._all_nodes: Optional[List[OntologyNode]] = None
        self._embeddings_computed = False

        # FAISS index for efficient vector search
        self._faiss_index: Optional[faiss.IndexFlatIP] = None  # Inner product for normalized vectors
        self._faiss_id_map: Optional[List[str]] = None  # Maps FAISS index to node IDs

    @abstractmethod
    def load_ontology(self) -> None:
        """Load the ontology file and populate internal structures"""
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[OntologyNode]:
        """Get a node by its ID"""
        pass

    @abstractmethod
    def get_neighbors(self, node_id: str,
                     relation_types: Optional[List[str]] = None,
                     direction: str = 'both') -> List[Tuple[OntologyNode, str]]:
        """
        Get neighboring nodes

        Args:
            node_id: ID of the source node
            relation_types: Filter by specific relation types (None = all)
            direction: 'outgoing', 'incoming', or 'both'

        Returns:
            List of (neighbor_node, relation_type) tuples
        """
        pass

    @abstractmethod
    def search_by_name(self, name: str, exact: bool = False) -> List[OntologyNode]:
        """Search for nodes by name (exact or partial match)"""
        pass

    @abstractmethod
    def get_all_nodes(self) -> List[OntologyNode]:
        """Get all nodes in the ontology"""
        pass

    def _ensure_embedding_model_loaded(self) -> None:
        """Lazily load the embedding model"""
        if self.embedding_model is None:
            print(f"Loading embedding model: {self.embedding_model_name}...")
            self.embedding_model = SentenceTransformer(self.embedding_model_name, device='cpu')

    def _compute_embeddings(self, force_recompute: bool = False) -> None:
        """Compute embeddings for all nodes and build FAISS index"""
        if self._embeddings_computed and not force_recompute:
            return

        self._ensure_embedding_model_loaded()
        nodes = self.get_all_nodes()

        print(f"Computing embeddings for {len(nodes)} nodes...")
        texts = [node.get_searchable_text() for node in nodes]
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,
            device='cpu'
        )

        # Store embeddings in cache
        for node, embedding in zip(nodes, embeddings):
            self._embedding_cache[node.id] = embedding

        # Build FAISS index for efficient similarity search
        print("Building FAISS index...")
        embedding_dim = embeddings.shape[1]
        self._faiss_index = faiss.IndexFlatIP(embedding_dim)  # Inner product for normalized vectors
        self._faiss_index.add(embeddings.astype('float32'))
        self._faiss_id_map = [node.id for node in nodes]

        self._embeddings_computed = True
        print(f"Embeddings computed and FAISS index built! ({len(nodes)} vectors)")

    def match_term_to_nodes(self,
                           query_term: str,
                           top_k: int = 10,
                           min_similarity: float = 0.3) -> List[TermMatch]:
        """
        Match a query term to ontology nodes using RAG

        Args:
            query_term: Term to match
            top_k: Number of top matches to return
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of TermMatch objects sorted by similarity
        """
        # First try exact and synonym matches
        exact_matches = self._find_exact_matches(query_term)
        if exact_matches:
            return exact_matches[:top_k]

        # Fall back to FAISS-based semantic search
        self._compute_embeddings()
        self._ensure_embedding_model_loaded()

        # Compute query embedding
        query_embedding = self.embedding_model.encode(
            [query_term],
            normalize_embeddings=True,
            device='cpu'
        )[0].astype('float32').reshape(1, -1)

        # Use FAISS for efficient nearest neighbor search
        # Search for more candidates than top_k to apply min_similarity filter
        search_k = min(top_k * 10, self._faiss_index.ntotal)
        similarities, indices = self._faiss_index.search(query_embedding, search_k)

        # Convert FAISS results to TermMatch objects
        matches = []
        for similarity, idx in zip(similarities[0], indices[0]):
            # Map from [-1, 1] to [0, 1] (inner product of normalized vectors = cosine similarity)
            similarity_normalized = (similarity + 1) / 2

            if similarity_normalized >= min_similarity:
                node_id = self._faiss_id_map[idx]
                node = self.get_node(node_id)
                if node:
                    matches.append(TermMatch(
                        node=node,
                        similarity_score=float(similarity_normalized),
                        match_type='semantic',
                        matched_text=node.name
                    ))

            if len(matches) >= top_k:
                break

        return matches

    def _find_exact_matches(self, query_term: str) -> List[TermMatch]:
        """Find exact and synonym matches"""
        query_lower = query_term.lower().strip()
        matches = []

        for node in self.get_all_nodes():
            # Check exact name match
            if node.name.lower() == query_lower:
                matches.append(TermMatch(
                    node=node,
                    similarity_score=1.0,
                    match_type='exact',
                    matched_text=node.name
                ))
            # Check synonym matches
            elif node.synonyms:
                for synonym in node.synonyms:
                    if synonym.lower() == query_lower:
                        matches.append(TermMatch(
                            node=node,
                            similarity_score=0.95,
                            match_type='synonym',
                            matched_text=synonym
                        ))
                        break

        return sorted(matches, key=lambda x: x.similarity_score, reverse=True)

    def get_node_context(self,
                        node_id: str,
                        depth: int = 1,
                        relation_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get a node with its neighborhood context

        Args:
            node_id: ID of the central node
            depth: How many hops to traverse (1 = immediate neighbors)
            relation_types: Filter by specific relation types

        Returns:
            Dictionary with node and multi-level neighbors
        """
        node = self.get_node(node_id)
        if not node:
            return {}

        context = {
            'node': node,
            'neighbors': {}
        }

        # BFS to get neighbors at each depth level
        visited = {node_id}
        current_level = [node_id]

        for level in range(1, depth + 1):
            next_level = []
            level_neighbors = []

            for current_id in current_level:
                neighbors = self.get_neighbors(
                    current_id,
                    relation_types=relation_types
                )

                for neighbor_node, relation_type in neighbors:
                    if neighbor_node.id not in visited:
                        visited.add(neighbor_node.id)
                        next_level.append(neighbor_node.id)
                        level_neighbors.append({
                            'node': neighbor_node,
                            'relation': relation_type,
                            'distance': level
                        })

            if level_neighbors:
                context['neighbors'][f'level_{level}'] = level_neighbors

            current_level = next_level
            if not current_level:
                break

        return context

    def batch_match_terms(self,
                         terms: List[str],
                         top_k: int = 5,
                         min_similarity: float = 0.3) -> Dict[str, List[TermMatch]]:
        """
        Match multiple terms to ontology nodes

        Args:
            terms: List of terms to match
            top_k: Number of top matches per term
            min_similarity: Minimum similarity threshold

        Returns:
            Dictionary mapping terms to their matches
        """
        # Ensure embeddings are computed once
        self._compute_embeddings()

        results = {}
        for term in terms:
            results[term] = self.match_term_to_nodes(term, top_k, min_similarity)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded ontology"""
        nodes = self.get_all_nodes()

        return {
            'total_nodes': len(nodes),
            'nodes_with_definitions': sum(1 for n in nodes if n.definition),
            'nodes_with_synonyms': sum(1 for n in nodes if n.synonyms),
            'total_synonyms': sum(len(n.synonyms) for n in nodes),
            'embeddings_computed': self._embeddings_computed,
            'cache_size': len(self._node_cache)
        }
