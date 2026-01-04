"""
Graph Scorer for Hybrid Retrieval

Calculates relevance scores based on knowledge graph structure using
PageRank and path proximity algorithms.
"""

from enum import Enum
from typing import Dict, List, Set, Optional, TYPE_CHECKING
import networkx as nx

if TYPE_CHECKING:
    from ..graph.graph_builder import VocationalGraphBuilder
    from .config import RetrieverConfig


class ScoringMethod(str, Enum):
    """Scoring methods for graph-based relevance."""

    PAGERANK_PATH = "pagerank_path"  # PageRank × path proximity
    PAGERANK_ONLY = "pagerank_only"  # PageRank importance only
    PATH_ONLY = "path_only"  # Path proximity only
    DEGREE_CENTRALITY = "degree_centrality"  # Degree centrality


class GraphScorer:
    """
    Calculate relevance scores based on graph structure.

    Uses a combination of:
    - Node importance (PageRank)
    - Path proximity (shorter paths = higher scores)
    - Edge weights (from relation types)
    """

    def __init__(
        self,
        graph_builder: "VocationalGraphBuilder",
        config: Optional["RetrieverConfig"] = None,
    ):
        """
        Initialize graph scorer.

        Args:
            graph_builder: The graph builder containing the knowledge graph
            config: Retriever configuration
        """
        self.graph = graph_builder
        self.config = config
        self._pagerank_cache: Optional[Dict[str, float]] = None

    def _get_pagerank(self) -> Dict[str, float]:
        """
        Get or compute PageRank scores for all nodes.

        Returns:
            Dict mapping node_id to PageRank score
        """
        if self._pagerank_cache is None:
            alpha = 0.85  # Standard damping factor
            self._pagerank_cache = self.graph.compute_pagerank(alpha=alpha)
        return self._pagerank_cache

    def score(
        self,
        node_ids: List[str],
        source_node_ids: Set[str],
        method: ScoringMethod = ScoringMethod.PAGERANK_PATH,
    ) -> Dict[str, float]:
        """
        Calculate graph-based relevance scores for nodes.

        Args:
            node_ids: Node IDs to score
            source_node_ids: Source nodes (e.g., from vector retrieval)
            method: Scoring method to use

        Returns:
            Dict mapping node_id to normalized score [0, 1]
        """
        if not node_ids:
            return {}

        if method == ScoringMethod.PAGERANK_PATH:
            scores = self._score_pagerank_path(node_ids, source_node_ids)
        elif method == ScoringMethod.PAGERANK_ONLY:
            scores = self._score_pagerank_only(node_ids)
        elif method == ScoringMethod.PATH_ONLY:
            scores = self._score_path_only(node_ids, source_node_ids)
        elif method == ScoringMethod.DEGREE_CENTRALITY:
            scores = self._score_degree_centrality(node_ids)
        else:
            raise ValueError(f"Unknown scoring method: {method}")

        # Normalize scores to [0, 1]
        return self._normalize_scores(scores)

    def _score_pagerank_path(
        self, node_ids: List[str], source_node_ids: Set[str]
    ) -> Dict[str, float]:
        """
        Score using PageRank × path proximity.

        score[node] = pagerank[node] × Σ(weight × decay^hops)

        Args:
            node_ids: Nodes to score
            source_node_ids: Source nodes for proximity calculation

        Returns:
            Dict mapping node_id to raw score
        """
        pagerank = self._get_pagerank()
        path_decay = self.config.path_decay if self.config else 0.5
        scores = {}

        for node_id in node_ids:
            if node_id not in pagerank:
                scores[node_id] = 0.0
                continue

            # Base importance from PageRank
            base_score = pagerank[node_id]

            # Calculate proximity to source nodes
            proximity = self._calculate_proximity(node_id, source_node_ids, path_decay)

            # Combine: importance × proximity
            scores[node_id] = base_score * proximity

        return scores

    def _score_pagerank_only(self, node_ids: List[str]) -> Dict[str, float]:
        """
        Score using PageRank importance only.

        Args:
            node_ids: Nodes to score

        Returns:
            Dict mapping node_id to PageRank score
        """
        pagerank = self._get_pagerank()
        return {node_id: pagerank.get(node_id, 0.0) for node_id in node_ids}

    def _score_path_only(
        self, node_ids: List[str], source_node_ids: Set[str]
    ) -> Dict[str, float]:
        """
        Score using path proximity only.

        Args:
            node_ids: Nodes to score
            source_node_ids: Source nodes for proximity calculation

        Returns:
            Dict mapping node_id to proximity score
        """
        path_decay = self.config.path_decay if self.config else 0.5
        return {
            node_id: self._calculate_proximity(node_id, source_node_ids, path_decay)
            for node_id in node_ids
        }

    def _score_degree_centrality(self, node_ids: List[str]) -> Dict[str, float]:
        """
        Score using degree centrality.

        Args:
            node_ids: Nodes to score

        Returns:
            Dict mapping node_id to degree centrality score
        """
        centrality = nx.degree_centrality(self.graph.graph)
        return {node_id: centrality.get(node_id, 0.0) for node_id in node_ids}

    def _calculate_proximity(
        self, node_id: str, source_node_ids: Set[str], decay: float
    ) -> float:
        """
        Calculate proximity score to source nodes.

        proximity = Σ(edge_weight × decay^hops) over all paths from sources

        Args:
            node_id: Target node
            source_node_ids: Source node IDs
            decay: Decay factor per hop

        Returns:
            Proximity score (higher is better/more relevant)
        """
        if not source_node_ids or node_id in source_node_ids:
            return 1.0

        total_proximity = 0.0

        for source_id in source_node_ids:
            if source_id not in self.graph.graph:
                continue

            try:
                # Find shortest path
                path = self.graph.find_shortest_path(source_id, node_id)
                if path is None:
                    continue

                hops = len(path) - 1
                if hops == 0:
                    # Same node
                    total_proximity += 1.0
                else:
                    # Calculate path score with decay
                    path_score = decay**hops
                    total_proximity += path_score

            except (nx.NetworkXNoPath, ValueError):
                # No path exists
                continue

        return min(total_proximity, 1.0)

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize scores to [0, 1] range using min-max normalization.

        Args:
            scores: Raw scores

        Returns:
            Normalized scores
        """
        if not scores:
            return {}

        values = list(scores.values())
        min_score = min(values)
        max_score = max(values)

        if max_score == min_score:
            # All scores are the same
            return {k: 1.0 for k in scores}

        range_score = max_score - min_score
        return {
            k: (v - min_score) / range_score for k, v in scores.items()
        }

    def score_chunks_by_entities(
        self,
        chunk_entity_ids: List[Set[str]],
        source_node_ids: Set[str],
        method: ScoringMethod = ScoringMethod.PAGERANK_PATH,
    ) -> List[float]:
        """
        Score chunks based on their associated entity IDs.

        Args:
            chunk_entity_ids: List of entity ID sets for each chunk
            source_node_ids: Source nodes for proximity calculation
            method: Scoring method

        Returns:
            List of scores (one per chunk)
        """
        all_node_ids = set()
        for entity_ids in chunk_entity_ids:
            all_node_ids.update(entity_ids)

        # Score all unique nodes
        node_scores = self.score(list(all_node_ids), source_node_ids, method)

        # Aggregate scores per chunk (max score)
        chunk_scores = []
        for entity_ids in chunk_entity_ids:
            if not entity_ids:
                chunk_scores.append(0.0)
            else:
                # Use maximum score among entities
                chunk_scores.append(
                    max(node_scores.get(eid, 0.0) for eid in entity_ids)
                )

        return chunk_scores
