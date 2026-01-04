"""
Hybrid Retriever for RAG Course Generation System

Combines vector similarity search with knowledge graph traversal
for multi-hop reasoning in vocational course retrieval.
"""

from dataclasses import dataclass
from typing import List, Set, Dict, Optional, Tuple, TYPE_CHECKING
import pickle
from pathlib import Path

if TYPE_CHECKING:
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document
    from ..graph.graph_builder import VocationalGraphBuilder
    from ..graph.schema import RelationType, ExperienceLevel, SkillComplexity
    from .config import RetrieverConfig
    from .entity_linker import EntityLinker
    from .graph_scorer import GraphScorer

from .graph_scorer import ScoringMethod


@dataclass
class RetrievalResult:
    """
    Result from hybrid retrieval.

    Attributes:
        document: The retrieved document
        fused_score: Final fused score (vector + graph)
        vector_score: Raw vector similarity score
        graph_score: Graph relevance score
        graph_paths: Graph paths from query to this result
    """
    document: "Document"
    fused_score: float
    vector_score: float
    graph_score: float
    graph_paths: List[List[str]] = None

    def __post_init__(self):
        if self.graph_paths is None:
            self.graph_paths = []


class HybridRetriever:
    """
    Graph-enhanced hybrid retriever for RAG system.

    Combines:
    1. Vector similarity search (ChromaDB)
    2. Knowledge graph traversal (NetworkX)
    3. Fusion scoring (configurable weights)

    Example:
        retriever = HybridRetriever(
            vector_store=chroma_db,
            graph_builder=graph,
            config=RetrieverConfig(vector_weight=0.6, graph_weight=0.4)
        )
        results = retriever.retrieve("网络安全工程师技能要求", top_k=5)
    """

    def __init__(
        self,
        vector_store: "Chroma",
        graph_builder: "VocationalGraphBuilder",
        config: Optional["RetrieverConfig"] = None,
    ):
        """
        Initialize hybrid retriever.

        Args:
            vector_store: ChromaDB vector store
            graph_builder: Knowledge graph builder
            config: Retriever configuration (uses defaults if None)
        """
        from .config import RetrieverConfig
        from .entity_linker import EntityLinker
        from .graph_scorer import GraphScorer

        self.vector_store = vector_store
        self.graph = graph_builder
        self.config = config or RetrieverConfig()

        # Initialize components
        self.entity_linker = EntityLinker(graph_builder, self.config)
        self.graph_scorer = GraphScorer(graph_builder, self.config)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        method: ScoringMethod = ScoringMethod.PAGERANK_PATH,
        filters: Optional[Dict] = None,
    ) -> List[RetrievalResult]:
        """
        Perform hybrid retrieval combining vector and graph search.

        Args:
            query: User query
            top_k: Number of final results to return
            method: Graph scoring method
            filters: Optional filters (experience_level, skill_complexity, etc.)

        Returns:
            List of ranked RetrievalResult objects
        """
        # Step 1: Vector search (recall)
        vector_results = self._vector_search(query)

        if not vector_results:
            return []

        # Step 2: Entity linking
        chunk_entity_ids = self.entity_linker.link_from_results(vector_results)

        # Step 3: Get all source node IDs
        source_node_ids = set()
        for entity_ids in chunk_entity_ids:
            source_node_ids.update(entity_ids)

        # Step 4: Graph expansion
        expanded_node_ids = self._expand_graph(source_node_ids)

        # Step 5: Graph scoring
        graph_scores = self.graph_scorer.score(
            list(expanded_node_ids),
            source_node_ids,
            method=method,
        )

        # Step 6: Score fusion
        fused_results = self._fuse_scores(
            vector_results,
            chunk_entity_ids,
            graph_scores,
        )

        # Step 7: Apply filters if specified
        if filters:
            fused_results = self._apply_filters(fused_results, filters)

        # Step 8: Re-rank and return top-K
        fused_results.sort(key=lambda r: r.fused_score, reverse=True)
        return fused_results[:top_k]

    def _vector_search(
        self,
        query: str,
    ) -> List[Tuple["Document", float]]:
        """
        Perform vector similarity search.

        Args:
            query: Search query

        Returns:
            List of (document, score) tuples
        """
        top_k = self.config.vector_top_k

        # Use similarity search with scores
        results = self.vector_store.similarity_search_with_score(query, k=top_k)

        # Convert ChromaDB distance to similarity (if needed)
        # ChromaDB uses L2 distance by default, lower is better
        # Convert to similarity: higher is better
        converted_results = []
        for doc, score in results:
            # Assuming L2 distance, convert to similarity
            # Use simple normalization for now
            similarity = 1.0 / (1.0 + score)
            converted_results.append((doc, similarity))

        return converted_results

    def _expand_graph(
        self,
        seed_node_ids: Set[str],
    ) -> Set[str]:
        """
        Expand graph from seed nodes.

        Args:
            seed_node_ids: Starting node IDs

        Returns:
            Set of expanded node IDs
        """
        if not seed_node_ids:
            return set()

        # Get relations for expansion
        all_relations = ["composed_of", "maps_to", "operates"]
        relations = self.config.get_expansion_relations(all_relations)

        expanded = set(seed_node_ids)

        for relation in relations:
            from ..graph.schema import RelationType
            relation_type = RelationType(relation)

            # Expand by this relation type
            expanded_ids = self.graph.expand_by_relation(
                node_ids=list(expanded),
                relation_type=relation_type,
                max_hops=self.config.graph_max_hops,
                max_nodes=self.config.graph_max_nodes,
            )
            expanded.update(expanded_ids)

        return expanded

    def _fuse_scores(
        self,
        vector_results: List[Tuple["Document", float]],
        chunk_entity_ids: List[Set[str]],
        graph_scores: Dict[str, float],
    ) -> List[RetrievalResult]:
        """
        Fuse vector and graph scores.

        fused_score = α × vector_score + β × graph_score

        Args:
            vector_results: Results from vector search
            chunk_entity_ids: Entity IDs for each chunk
            graph_scores: Graph scores for each node

        Returns:
            List of RetrievalResult objects
        """
        fused_results = []

        for (doc, vector_score), entity_ids in zip(vector_results, chunk_entity_ids):
            # Get graph score for this chunk
            if entity_ids:
                # Use max graph score among entities
                graph_score = max(
                    graph_scores.get(eid, 0.0) for eid in entity_ids
                )
            else:
                graph_score = 0.0

            # Normalize vector score to [0, 1]
            # (already normalized in _vector_search)

            # Calculate fused score
            fused_score = (
                self.config.vector_weight * vector_score +
                self.config.graph_weight * graph_score
            )

            # Get graph paths for visualization
            paths = self._get_graph_paths(entity_ids)

            result = RetrievalResult(
                document=doc,
                fused_score=fused_score,
                vector_score=vector_score,
                graph_score=graph_score,
                graph_paths=paths,
            )
            fused_results.append(result)

        return fused_results

    def _get_graph_paths(
        self,
        entity_ids: Set[str],
    ) -> List[List[str]]:
        """
        Get graph paths for a set of entities.

        Args:
            entity_ids: Entity IDs to find paths for

        Returns:
            List of node ID paths
        """
        paths = []

        for entity_id in entity_ids:
            if entity_id in self.graph.graph:
                # Get shortest paths to StandardTask nodes
                from ..graph.schema import EntityType, RelationType

                # Find StandardTask nodes
                standard_tasks = self.graph.get_nodes_by_type(EntityType.STANDARD_TASK)

                for task in standard_tasks:
                    task_id = task["id"]
                    path = self.graph.find_shortest_path(
                        entity_id,
                        task_id,
                        [RelationType.MAPS_TO],
                    )
                    if path:
                        paths.append(path)

        return paths

    def _apply_filters(
        self,
        results: List[RetrievalResult],
        filters: Dict,
    ) -> List[RetrievalResult]:
        """
        Apply filters to results.

        Args:
            results: List of retrieval results
            filters: Filter criteria

        Returns:
            Filtered results
        """
        filtered = []

        for result in results:
            doc = result.document
            metadata = doc.metadata

            # Check experience level filter
            if "experience_level" in filters:
                if metadata.get("experience_level") != filters["experience_level"]:
                    continue

            # Check complexity filter
            if "complexity" in filters:
                if metadata.get("complexity") != filters["complexity"]:
                    continue

            # Check entity type filter
            if "entity_type" in filters:
                if metadata.get("entity_type") != filters["entity_type"]:
                    continue

            filtered.append(result)

        return filtered

    def retrieve_by_experience(
        self,
        query: str,
        experience_level: str,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Retrieve with experience level filter.

        Args:
            query: Search query
            experience_level: Experience level to filter by
            top_k: Number of results

        Returns:
            Filtered retrieval results
        """
        return self.retrieve(
            query,
            top_k=top_k,
            filters={"experience_level": experience_level},
        )

    def retrieve_by_complexity(
        self,
        query: str,
        complexity: str,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Retrieve with skill complexity filter.

        Args:
            query: Search query
            complexity: Skill complexity to filter by
            top_k: Number of results

        Returns:
            Filtered retrieval results
        """
        return self.retrieve(
            query,
            top_k=top_k,
            filters={"complexity": complexity},
        )

    @staticmethod
    def load_graph(graph_path: str) -> "VocationalGraphBuilder":
        """
        Load graph from file.

        Args:
            graph_path: Path to graph pickle file

        Returns:
            Loaded VocationalGraphBuilder
        """
        from ..graph.graph_builder import VocationalGraphBuilder

        builder = VocationalGraphBuilder()
        builder.load(graph_path)
        return builder

    @staticmethod
    def load_vector_store(
        persist_dir: str,
        embedding_function,
    ) -> "Chroma":
        """
        Load vector store from directory.

        Args:
            persist_dir: Path to ChromaDB directory
            embedding_function: Embedding function

        Returns:
            Loaded Chroma vector store
        """
        from langchain_community.vectorstores import Chroma

        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_function,
        )
