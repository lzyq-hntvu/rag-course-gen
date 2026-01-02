"""
职教能力图谱数据模型定义

核心思想：建立"企业岗位（JD）"到"国家教学标准（Standard）"的映射

实体层次：
    Real_Project (真实项目载体) --[composed_of]--> Action_Skill (行动技能)
                                                    |
                                                    +--[maps_to]--> Standard_Task (典型工作任务)
                                                    |
                                                    +--[operates]--> Tool (工具/设备)
"""

from enum import Enum
from typing import Optional, List, Any, Dict
from pydantic import BaseModel, Field
import uuid


class EntityType(str, Enum):
    """图谱实体类型"""
    STANDARD_TASK = "standard_task"      # 典型工作任务（来自国标）
    REAL_PROJECT = "real_project"        # 真实项目载体（来自JD或工程案例）
    ACTION_SKILL = "action_skill"        # 行动技能（连接桥梁）
    TOOL = "tool"                        # 工具/设备


class RelationType(str, Enum):
    """图谱关系类型"""
    COMPOSED_OF = "composed_of"          # Real_Project -> Action_Skill
    MAPS_TO = "maps_to"                  # Action_Skill -> Standard_Task (核心推理边)
    OPERATES = "operates"                # Action_Skill -> Tool


class ExperienceLevel(str, Enum):
    """工作经验等级（用于项目分层）"""
    ENTRY = "entry"           # 入门级：应届生、0-1年
    JUNIOR = "junior"         # 初级：1-3年
    MID_LEVEL = "mid_level"   # 中级：3-5年
    SENIOR = "senior"         # 高级：5-8年
    EXPERT = "expert"         # 专家：8年以上


class SkillComplexity(str, Enum):
    """技能复杂度（用于技能分层）"""
    LOW = "low"               # 低复杂度：执行类、基础操作
    MEDIUM = "medium"         # 中复杂度：独立完成、常规开发
    HIGH = "high"             # 高复杂度：架构设计、性能优化、疑难解决


class GraphNode(BaseModel):
    """图谱节点基类"""
    id: str = Field(default_factory=lambda: f"node_{uuid.uuid4().hex[:8]}")
    entity_type: EntityType
    name: str
    description: Optional[str] = None
    source_document: Optional[str] = None     # 来源文档（如JD文件名或国标文件名）
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, GraphNode):
            return False
        return self.id == other.id


class StandardTask(GraphNode):
    """典型工作任务（来自国标，是课程骨架）"""
    entity_type: EntityType = EntityType.STANDARD_TASK
    task_code: Optional[str] = None           # 国标中的任务编码
    knowledge_points: List[str] = Field(default_factory=list)
    skill_points: List[str] = Field(default_factory=list)
    career_level: Optional[str] = None        # 如：中级工、高级工


class RealProject(GraphNode):
    """真实项目载体（来自JD或工程案例，是教学情境）"""
    entity_type: EntityType = EntityType.REAL_PROJECT
    company_name: Optional[str] = None        # 企业名称
    project_context: Optional[str] = None     # 项目背景描述
    difficulty_level: Optional[str] = None    # 难度等级
    experience_level: Optional[ExperienceLevel] = None  # 经验要求等级（分层核心字段）


class ActionSkill(GraphNode):
    """行动技能（连接JD和国标的桥梁）"""
    entity_type: EntityType = EntityType.ACTION_SKILL
    skill_category: Optional[str] = None      # 技能分类（如：网络配置、安全管理）
    proficiency_level: Optional[str] = None   # 熟练度要求（如：掌握、熟悉、了解）
    prerequisites: List[str] = Field(default_factory=list)  # 前置技能
    complexity: Optional[SkillComplexity] = None  # 技能复杂度（分层核心字段）


class Tool(GraphNode):
    """工具/设备"""
    entity_type: EntityType = EntityType.TOOL
    tool_category: Optional[str] = None       # 工具分类（如：网络设备、编程语言）
    vendor: Optional[str] = None              # 厂商（如：华为、H3C）
    version: Optional[str] = None             # 版本号


class GraphEdge(BaseModel):
    """图谱边"""
    id: str = Field(default_factory=lambda: f"edge_{uuid.uuid4().hex[:8]}")
    source_id: str
    target_id: str
    relation_type: RelationType
    weight: float = 1.0                       # 边权重，用于检索时的相关性计算
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, GraphEdge):
            return False
        return self.id == other.id


class VocationalGraph(BaseModel):
    """职教能力图谱"""
    nodes: Dict[str, GraphNode] = Field(default_factory=dict)
    edges: List[GraphEdge] = Field(default_factory=list)

    def add_node(self, node: GraphNode) -> None:
        """添加节点"""
        self.nodes[node.id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        """添加边"""
        self.edges.append(edge)

    def get_nodes_by_type(self, entity_type: EntityType) -> List[GraphNode]:
        """按类型获取节点"""
        return [n for n in self.nodes.values() if n.entity_type == entity_type]

    def get_neighbors(self, node_id: str, relation_type: Optional[RelationType] = None) -> List[GraphNode]:
        """获取邻居节点"""
        neighbors = []
        for edge in self.edges:
            if edge.source_id == node_id:
                if relation_type is None or edge.relation_type == relation_type:
                    if edge.target_id in self.nodes:
                        neighbors.append(self.nodes[edge.target_id])
        return neighbors

    class Config:
        arbitrary_types_allowed = True


# 预定义的边权重配置（用于检索时的相关性计算）
RELATION_WEIGHTS = {
    RelationType.MAPS_TO: 1.0,        # 核心：技能到任务的映射
    RelationType.COMPOSED_OF: 0.8,    # 项目到技能的组成关系
    RelationType.OPERATES: 0.6,       # 技能到工具的操作关系
}


# Schema版本信息
SCHEMA_VERSION = "1.0.0"
SCHEMA_DESCRIPTION = "职教能力图谱：连接企业JD与国家教学标准"
