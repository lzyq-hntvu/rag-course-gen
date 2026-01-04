"""
Configuration for Hybrid Retriever

Defines the RetrieverConfig dataclass for configuring graph-enhanced retrieval.
"""

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..graph.schema import RelationType


@dataclass
class RetrieverConfig:
    """
    Configuration for hybrid retrieval combining vector and graph search.

    Attributes:
        vector_weight: Weight for vector similarity score (default: 0.6)
        graph_weight: Weight for graph relevance score (default: 0.4)
        vector_top_k: Number of results to retrieve from vector search (default: 50)
        graph_max_hops: Maximum hops for graph expansion (default: 2)
        graph_max_nodes: Maximum nodes to expand in graph (default: 100)
        expansion_relations: Relation types for graph expansion (None = all)
        use_pagerank: Whether to use PageRank for node importance (default: True)
        path_decay: Decay factor per hop for path scoring (default: 0.5)
        vector_score_min: Minimum vector score threshold (default: 0.0)
        enable_fuzzy_match: Enable fuzzy matching for entity linking (default: True)
        fuzzy_match_threshold: Threshold for fuzzy matching (default: 0.85)
    """

    # Fusion weights (should sum to 1.0)
    vector_weight: float = 0.6
    graph_weight: float = 0.4

    # Vector search settings
    vector_top_k: int = 50

    # Graph expansion settings
    graph_max_hops: int = 2
    graph_max_nodes: int = 100
    expansion_relations: Optional[List[str]] = None  # None means all relations

    # Scoring settings
    use_pagerank: bool = True
    path_decay: float = 0.5

    # Filtering settings
    vector_score_min: float = 0.0

    # Entity linking settings
    enable_fuzzy_match: bool = True
    fuzzy_match_threshold: float = 0.85

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate fusion weights
        if not 0 <= self.vector_weight <= 1:
            raise ValueError(f"vector_weight must be in [0, 1], got {self.vector_weight}")
        if not 0 <= self.graph_weight <= 1:
            raise ValueError(f"graph_weight must be in [0, 1], got {self.graph_weight}")
        if abs(self.vector_weight + self.graph_weight - 1.0) > 0.01:
            raise ValueError(
                f"vector_weight + graph_weight should equal 1.0, "
                f"got {self.vector_weight + self.graph_weight}"
            )

        # Validate other parameters
        if self.vector_top_k < 1:
            raise ValueError(f"vector_top_k must be >= 1, got {self.vector_top_k}")
        if self.graph_max_hops < 1:
            raise ValueError(f"graph_max_hops must be >= 1, got {self.graph_max_hops}")
        if not 0 < self.path_decay <= 1:
            raise ValueError(f"path_decay must be in (0, 1], got {self.path_decay}")

    def get_expansion_relations(
        self, all_relations: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get the list of relations to use for graph expansion.

        Args:
            all_relations: All available relation types (from schema)

        Returns:
            List of relation types to use for expansion
        """
        if self.expansion_relations is not None:
            return self.expansion_relations

        # Default: use all relations if specified
        if all_relations is not None:
            return all_relations

        # Fallback to all 3 relation types
        return ["composed_of", "maps_to", "operates"]
