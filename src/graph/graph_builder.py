"""
职教能力图谱构建器

基于NetworkX构建内存图谱，支持：
1. 图谱构建和更新
2. 图谱查询和遍历
3. 图谱可视化
4. 图谱序列化
"""

import json
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pickle

from .schema import (
    VocationalGraph,
    GraphNode,
    GraphEdge,
    EntityType,
    RelationType,
    StandardTask,
    RealProject,
    ActionSkill,
    Tool,
    RELATION_WEIGHTS,
)
from .extractor import EntityExtractor, entities_to_schema


class VocationalGraphBuilder:
    """职教能力图谱构建器（基于NetworkX）"""

    def __init__(self):
        """初始化图谱构建器"""
        self.graph = nx.MultiDiGraph()  # 有向多重图，允许节点间有多条边
        self.entity_index: Dict[str, GraphNode] = {}  # id -> entity

    def add_node(self, node: GraphNode) -> str:
        """
        添加节点到图谱

        Args:
            node: 图谱节点

        Returns:
            str: 节点ID
        """
        # 转换为可序列化的字典
        node_data = {
            "id": node.id,
            "entity_type": node.entity_type.value,
            "name": node.name,
            "description": node.description,
            "source_document": node.source_document,
            **node.metadata,
        }

        # 添加特定类型的额外属性
        if isinstance(node, StandardTask):
            node_data.update({
                "task_code": node.task_code,
                "knowledge_points": node.knowledge_points,
                "skill_points": node.skill_points,
                "career_level": node.career_level,
            })
        elif isinstance(node, RealProject):
            node_data.update({
                "company_name": node.company_name,
                "project_context": node.project_context,
                "difficulty_level": node.difficulty_level,
                "experience_level": node.experience_level.value if node.experience_level else None,
            })
        elif isinstance(node, ActionSkill):
            node_data.update({
                "skill_category": node.skill_category,
                "proficiency_level": node.proficiency_level,
                "prerequisites": node.prerequisites,
                "complexity": node.complexity.value if node.complexity else None,
            })
        elif isinstance(node, Tool):
            node_data.update({
                "tool_category": node.tool_category,
                "vendor": node.vendor,
                "version": node.version,
            })

        self.graph.add_node(node.id, **node_data)
        self.entity_index[node.id] = node
        return node.id

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        weight: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        添加边到图谱

        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            relation_type: 关系类型
            weight: 边权重（如果为None，使用预设权重）
            metadata: 额外的元数据

        Returns:
            str: 边的key（MultiDigraph中用key标识同节点间的多条边）
        """
        if source_id not in self.graph:
            raise ValueError(f"源节点不存在: {source_id}")
        if target_id not in self.graph:
            raise ValueError(f"目标节点不存在: {target_id}")

        # 使用预设权重或自定义权重
        if weight is None:
            weight = RELATION_WEIGHTS.get(relation_type, 0.5)

        edge_data = {
            "relation_type": relation_type.value,
            "weight": weight,
            **(metadata or {}),
        }

        self.graph.add_edge(source_id, target_id, **edge_data)
        return edge_data

    def add_entities_from_extractor(
        self,
        standard_tasks: List[StandardTask],
        real_projects: List[RealProject],
        action_skills: List[ActionSkill],
        tools: List[Tool],
        edges: List[GraphEdge],
    ) -> None:
        """
        从抽取结果批量添加实体和关系

        Args:
            standard_tasks: 典型工作任务列表
            real_projects: 真实项目载体列表
            action_skills: 行动技能列表
            tools: 工具/设备列表
            edges: 关系列表
        """
        # 添加所有节点
        all_entities = [
            *standard_tasks,
            *real_projects,
            *action_skills,
            *tools,
        ]

        for entity in all_entities:
            self.add_node(entity)

        # 添加所有边
        for edge in edges:
            if edge.source_id in self.graph and edge.target_id in self.graph:
                self.add_edge(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    relation_type=edge.relation_type,
                    weight=edge.weight,
                    metadata=edge.metadata,
                )

    def get_nodes_by_type(self, entity_type: EntityType) -> List[Dict[str, Any]]:
        """按类型获取节点"""
        return [
            self.graph.nodes[node_id]
            for node_id in self.graph.nodes()
            if self.graph.nodes[node_id]["entity_type"] == entity_type.value
        ]

    def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        根据ID获取节点

        Args:
            node_id: 节点ID

        Returns:
            Optional[Dict]: 节点数据，如果不存在则返回None
        """
        if node_id in self.graph:
            return self.graph.nodes[node_id]
        return None

    def get_neighbors(
        self,
        node_id: str,
        relation_type: Optional[RelationType] = None,
        direction: str = "out",
    ) -> List[Dict[str, Any]]:
        """
        获取邻居节点

        Args:
            node_id: 节点ID
            relation_type: 关系类型（None表示所有关系）
            direction: 方向（"out"出边，"in"入边，"both"双向）

        Returns:
            List[Dict]: 邻居节点数据列表
        """
        if node_id not in self.graph:
            return []

        neighbors = []
        edges = []

        if direction in ("out", "both"):
            edges.extend(self.graph.out_edges(node_id, data=True))
        if direction in ("in", "both"):
            edges.extend(self.graph.in_edges(node_id, data=True))

        for source, target, data in edges:
            neighbor_id = target if direction == "out" else source
            if direction == "both":
                neighbor_id = target if source == node_id else source

            if relation_type is None or data.get("relation_type") == relation_type.value:
                neighbors.append(self.graph.nodes[neighbor_id])

        return neighbors

    def find_shortest_path(
        self,
        source_id: str,
        target_id: str,
        relation_types: Optional[List[RelationType]] = None,
    ) -> Optional[List[str]]:
        """
        查找两个节点之间的最短路径

        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            relation_types: 允许使用的关系类型（None表示所有关系）

        Returns:
            Optional[List[str]]: 节点ID列表，如果不存在路径则返回None
        """
        if source_id not in self.graph or target_id not in self.graph:
            return None

        # 如果指定了关系类型，创建子图
        subgraph = self.graph
        if relation_types is not None:
            relation_values = [rt.value for rt in relation_types]
            subgraph = self.graph.edge_subgraph([
                (u, v, k)
                for u, v, k, d in self.graph.edges(keys=True, data=True)
                if d.get("relation_type") in relation_values
            ])

        # Check if both nodes are in the subgraph
        if source_id not in subgraph or target_id not in subgraph:
            return None

        try:
            return nx.shortest_path(subgraph, source_id, target_id)
        except nx.NetworkXNoPath:
            return None

    def expand_by_relation(
        self,
        node_ids: List[str],
        relation_type: RelationType,
        max_hops: int = 2,
        max_nodes: int = 100,
    ) -> List[str]:
        """
        从一组节点出发，通过指定关系进行扩展

        Args:
            node_ids: 起始节点ID列表
            relation_type: 扩展使用的关系类型
            max_hops: 最大跳数
            max_nodes: 最大返回节点数

        Returns:
            List[str]: 扩展后的节点ID列表
        """
        expanded = set(node_ids)
        frontier = set(node_ids)

        for hop in range(max_hops):
            if not frontier:
                break
            if len(expanded) >= max_nodes:
                break

            new_frontier = set()
            for node_id in frontier:
                neighbors = self.get_neighbors(node_id, relation_type, direction="out")
                for neighbor in neighbors:
                    if neighbor["id"] not in expanded:
                        new_frontier.add(neighbor["id"])
                        expanded.add(neighbor["id"])
                        if len(expanded) >= max_nodes:
                            return list(expanded)

            frontier = new_frontier

        return list(expanded)

    def compute_pagerank(
        self,
        personalization: Optional[Dict[str, float]] = None,
        alpha: float = 0.85,
    ) -> Dict[str, float]:
        """
        计算PageRank，用于节点重要性排序

        Args:
            personalization: 个性化向量（用于个性化PageRank）
            alpha: 阻尼系数

        Returns:
            Dict[str, float]: 节点ID到PageRank分数的映射
        """
        return nx.pagerank(self.graph, alpha=alpha, personalization=personalization)

    def get_subgraph_by_nodes(self, node_ids: List[str]) -> nx.MultiDiGraph:
        """获取指定节点的子图"""
        return self.graph.subgraph(node_ids)

    def save(self, filepath: str) -> None:
        """
        保存图谱到文件（使用pickle格式）

        Args:
            filepath: 保存路径
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump({
                "graph": self.graph,
                "entity_index": self.entity_index,
            }, f)

    def load(self, filepath: str) -> None:
        """
        从文件加载图谱

        Args:
            filepath: 文件路径
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.graph = data["graph"]
            self.entity_index = data["entity_index"]

    def save_gexf(self, filepath: str) -> None:
        """
        导出为GEXF格式（可用Gephi等工具可视化）

        Args:
            filepath: 保存路径
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # 转换为可序列化的格式
        export_graph = nx.DiGraph()

        for node_id, node_data in self.graph.nodes(data=True):
            # 清理不可序列化的数据
            clean_data = {}
            for k, v in node_data.items():
                if isinstance(v, (str, int, float, bool, list)):
                    clean_data[k] = v
                elif isinstance(v, list):
                    clean_data[k] = ",".join(str(x) for x in v)
            export_graph.add_node(node_id, **clean_data)

        for u, v, data in self.graph.edges(data=True):
            export_graph.add_edge(u, v, **data)

        nx.write_gexf(export_graph, filepath)

    def get_statistics(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        entity_type_counts = {}
        for node in self.graph.nodes(data=True):
            entity_type = node[1]["entity_type"]
            entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1

        relation_type_counts = {}
        for _, _, edge in self.graph.edges(data=True):
            relation_type = edge["relation_type"]
            relation_type_counts[relation_type] = relation_type_counts.get(relation_type, 0) + 1

        # 处理空图谱情况
        is_connected = False
        density = 0.0
        if self.graph.number_of_nodes() > 0:
            try:
                is_connected = nx.is_weakly_connected(self.graph)
                density = nx.density(self.graph)
            except nx.NetworkXPointlessConcept:
                # 空图或单节点图
                pass

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "entity_type_counts": entity_type_counts,
            "relation_type_counts": relation_type_counts,
            "is_connected": is_connected,
            "density": density,
        }


# ============================================================================
# 工具函数
# ============================================================================
def build_graph_from_documents(
    jd_texts: List[str],
    standard_texts: List[str],
    anthropic_api_key: str,
) -> VocationalGraphBuilder:
    """
    从JD和国标文档构建图谱

    Args:
        jd_texts: JD文本列表
        standard_texts: 国标文本列表
        anthropic_api_key: Anthropic API密钥

    Returns:
        VocationalGraphBuilder: 构建好的图谱
    """
    extractor = EntityExtractor(api_key=anthropic_api_key)
    builder = VocationalGraphBuilder()

    # 抽取JD中的实体
    for i, jd_text in enumerate(jd_texts):
        result = extractor.extract(jd_text, source_document=f"jd_{i}", document_type="jd")
        standard_tasks, real_projects, action_skills, tools, edges = entities_to_schema(result)
        builder.add_entities_from_extractor(standard_tasks, real_projects, action_skills, tools, edges)

    # 抽取国标中的实体
    for i, standard_text in enumerate(standard_texts):
        result = extractor.extract(
            standard_text,
            source_document=f"standard_{i}",
            document_type="standard"
        )
        standard_tasks, real_projects, action_skills, tools, edges = entities_to_schema(result)
        builder.add_entities_from_extractor(standard_tasks, real_projects, action_skills, tools, edges)

    return builder
