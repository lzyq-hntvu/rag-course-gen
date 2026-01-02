"""
图谱可视化脚本

用于验证职教能力图谱的构建结果
"""

import os
import sys
import json
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
from typing import Dict, List, Optional

# 配置matplotlib支持中文
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from src.graph import (
    VocationalGraphBuilder,
    EntityExtractor,
    EntityType,
    RelationType,
    entities_to_schema,
)
from src.graph.loader import JobDataLoader, load_job_records, JobRecord


# 节点类型配色方案
ENTITY_TYPE_COLORS = {
    EntityType.STANDARD_TASK.value: "#4CAF50",    # 绿色
    EntityType.REAL_PROJECT.value: "#2196F3",     # 蓝色
    EntityType.ACTION_SKILL.value: "#FF9800",     # 橙色
    EntityType.TOOL.value: "#9C27B0",             # 紫色
}

# 关系类型样式
RELATION_STYLE = {
    RelationType.COMPOSED_OF.value: {"style": "solid", "color": "#2196F3"},
    RelationType.MAPS_TO.value: {"style": "dashed", "color": "#4CAF50"},
    RelationType.OPERATES.value: {"style": "dotted", "color": "#9C27B0"},
}


def load_mock_data() -> tuple[List[Dict], List[str]]:
    """加载Mock数据"""
    mock_dir = project_root / "data" / "mock"

    # 加载JD数据
    jd_file = mock_dir / "sample_jds.json"
    with open(jd_file, "r", encoding="utf-8") as f:
        jds = json.load(f)

    # 加载国标数据
    standard_file = mock_dir / "sample_standard.txt"
    with open(standard_file, "r", encoding="utf-8") as f:
        standard_text = f.read()

    return jds, [standard_text]


def visualize_graph(
    builder: VocationalGraphBuilder,
    output_path: Optional[str] = None,
    max_nodes: int = 100,
    figsize: tuple = (16, 12),
) -> None:
    """
    可视化图谱

    Args:
        builder: 图谱构建器
        output_path: 输出路径（如果为None则显示）
        max_nodes: 最大显示节点数
        figsize: 图像大小
    """
    graph = builder.graph

    # 如果节点太多，只显示高连通度节点
    if graph.number_of_nodes() > max_nodes:
        degrees = dict(graph.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        top_node_ids = [node[0] for node in top_nodes]
        graph = graph.subgraph(top_node_ids)

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 计算布局
    pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)

    # 按实体类型分组绘制节点
    for entity_type in ENTITY_TYPE_COLORS:
        node_ids = [
            n for n, d in graph.nodes(data=True)
            if d.get("entity_type") == entity_type
        ]
        if node_ids:
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=node_ids,
                node_color=ENTITY_TYPE_COLORS[entity_type],
                node_size=300,
                alpha=0.8,
                ax=ax,
            )

    # 绘制边
    for relation_type, style in RELATION_STYLE.items():
        edges = [
            (u, v)
            for u, v, d in graph.edges(data=True)
            if d.get("relation_type") == relation_type
        ]
        if edges:
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=edges,
                edge_color=style["color"],
                style=style["style"],
                alpha=0.5,
                arrowsize=20,
                arrowstyle="->",
                ax=ax,
            )

    # 绘制节点标签（只显示名称）
    labels = nx.get_node_attributes(graph, "name")
    # 简化过长的标签
    labels = {k: v[:10] + "..." if len(v) > 10 else v for k, v in labels.items()}
    nx.draw_networkx_labels(
        graph,
        pos,
        labels,
        font_size=8,
        font_family="sans-serif",
        ax=ax,
    )

    # 添加图例
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", label="典型工作任务",
                   markerfacecolor=ENTITY_TYPE_COLORS[EntityType.STANDARD_TASK.value],
                   markersize=10),
        plt.Line2D([0], [0], marker="o", color="w", label="真实项目载体",
                   markerfacecolor=ENTITY_TYPE_COLORS[EntityType.REAL_PROJECT.value],
                   markersize=10),
        plt.Line2D([0], [0], marker="o", color="w", label="行动技能",
                   markerfacecolor=ENTITY_TYPE_COLORS[EntityType.ACTION_SKILL.value],
                   markersize=10),
        plt.Line2D([0], [0], marker="o", color="w", label="工具/设备",
                   markerfacecolor=ENTITY_TYPE_COLORS[EntityType.TOOL.value],
                   markersize=10),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

    ax.set_title("职教能力图谱", fontsize=16, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"图谱已保存到: {output_path}")
    else:
        plt.show()


def print_graph_statistics(builder: VocationalGraphBuilder) -> None:
    """打印图谱统计信息"""
    stats = builder.get_statistics()

    print("\n" + "=" * 50)
    print("职教能力图谱统计信息")
    print("=" * 50)
    print(f"总节点数: {stats['total_nodes']}")
    print(f"总边数: {stats['total_edges']}")
    print(f"图谱密度: {stats['density']:.4f}")
    print(f"是否连通: {stats['is_connected']}")

    print("\n实体类型分布:")
    for entity_type, count in stats["entity_type_counts"].items():
        print(f"  - {entity_type}: {count}")

    print("\n关系类型分布:")
    for relation_type, count in stats["relation_type_counts"].items():
        print(f"  - {relation_type}: {count}")

    # 显示一些示例实体
    print("\n" + "=" * 50)
    print("示例实体")
    print("=" * 50)

    for entity_type in EntityType:
        nodes = builder.get_nodes_by_type(entity_type)
        if nodes:
            print(f"\n{entity_type.value} (共{len(nodes)}个):")
            for i, node in enumerate(nodes[:3]):  # 只显示前3个
                print(f"  [{i+1}] {node['name']}")


def demo_entity_expansion(builder: VocationalGraphBuilder) -> None:
    """演示图谱扩展功能"""
    print("\n" + "=" * 50)
    print("图谱扩展演示")
    print("=" * 50)

    # 找一个真实的行动技能节点
    action_skills = builder.get_nodes_by_type(EntityType.ACTION_SKILL)
    if action_skills:
        seed_node = action_skills[0]
        print(f"\n种子节点: {seed_node['name']}")

        # 通过maps_to关系扩展
        expanded = builder.expand_by_relation(
            [seed_node["id"]],
            RelationType.MAPS_TO,
            max_hops=2,
        )

        print(f"通过maps_to关系扩展后的节点 ({len(expanded)}个):")
        for node_id in expanded[:5]:
            node = builder.graph.nodes[node_id]
            print(f"  - [{node['entity_type']}] {node['name']}")

    # 演示最短路径查询
    real_projects = builder.get_nodes_by_type(EntityType.REAL_PROJECT)
    standard_tasks = builder.get_nodes_by_type(EntityType.STANDARD_TASK)

    if real_projects and standard_tasks:
        project = real_projects[0]
        task = standard_tasks[0]

        print(f"\n查找路径:")
        print(f"  从: {project['name']} (项目)")
        print(f"  到: {task['name']} (任务)")

        path = builder.find_shortest_path(project["id"], task["id"])
        if path:
            print(f"  路径长度: {len(path)}")
            for i, node_id in enumerate(path):
                node = builder.graph.nodes[node_id]
                print(f"    {i+1}. [{node['entity_type']}] {node['name']}")
        else:
            print("  未找到路径")


def main():
    """主函数"""
    print("职教能力图谱构建与可视化演示")
    print("=" * 50)

    # 1. 加载JD数据（从CSV）
    print("\n[1/5] 加载JD数据...")
    loader = JobDataLoader()
    available_files = loader.list_available_files()
    print(f"  - 可用CSV文件: {len(available_files)}个")

    # 加载优先技术岗位数据（取5条用于演示）
    job_records = load_job_records(count=5, use_priority=True)
    print(f"  - 加载JD数量: {len(job_records)}")
    for i, record in enumerate(job_records):
        print(f"    [{i+1}] {record.position_name} - {record.company}")

    # 2. 初始化LLM抽取器
    print("\n[2/5] 初始化LLM抽取器...")
    try:
        extractor = EntityExtractor()  # 从.env读取配置
        print(f"  - 模型: {extractor.model}")
        print(f"  - API地址: {extractor.base_url}")
    except ValueError as e:
        print(f"  错误: {e}")
        print("  请确保.env文件中配置了OPENAI_API_KEY")
        print("  使用演示数据...")
        demo_without_llm()
        return

    builder = VocationalGraphBuilder()

    # 3. 抽取实体和关系
    print("\n[3/5] 使用LLM抽取实体和关系...")
    for i, record in enumerate(job_records):
        print(f"  - 处理JD [{i+1}/{len(job_records)}]: {record.position_name}")

        try:
            result = extractor.extract(
                record.to_text(),
                source_document=f"{record.source_file}_{i}",
                document_type="jd"
            )
            standard_tasks, real_projects, action_skills, tools, edges = entities_to_schema(result)
            builder.add_entities_from_extractor(standard_tasks, real_projects, action_skills, tools, edges)
            print(f"    抽取: {len(action_skills)}个技能, {len(tools)}个工具")
        except Exception as e:
            print(f"    警告: 抽取失败 - {e}")
            continue

    print(f"  - 抽取完成")

    # 4. 打印统计信息
    print("\n[4/5] 图谱统计信息...")
    print_graph_statistics(builder)

    # 5. 演示图谱功能
    print("\n[5/5] 图谱功能演示...")
    demo_entity_expansion(builder)

    # 6. 可视化
    print("\n生成可视化图像...")
    output_dir = project_root / "data" / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "graph_visualization.png"
    visualize_graph(builder, output_path=str(output_path))

    # 保存图谱
    graph_save_path = output_dir / "graph.pkl"
    builder.save(str(graph_save_path))
    print(f"图谱已保存到: {graph_save_path}")

    print("\n完成!")


def demo_without_llm():
    """不使用LLM的演示模式"""
    print("\n创建一个示例图谱用于演示...")

    builder = VocationalGraphBuilder()

    # 手动添加一些示例节点和边
    from src.graph import StandardTask, RealProject, ActionSkill, Tool, GraphEdge, RelationType

    # 添加节点
    task1 = StandardTask(name="网络设备配置", task_code="T-01", 职业等级="中级工")
    task2 = StandardTask(name="安全防护部署", task_code="T-02", 职业等级="高级工")

    project1 = RealProject(name="医院网络改造项目", company_name="某公司")
    project2 = RealProject(name="企业安全防护项目", company_name="某公司")

    skill1 = ActionSkill(name="配置华为交换机", skill_category="网络配置")
    skill2 = ActionSkill(name="配置防火墙", skill_category="安全配置")
    skill3 = ActionSkill(name="编写Python脚本", skill_category="自动化")

    tool1 = Tool(name="华为S5700", tool_category="网络设备", vendor="华为")
    tool2 = Tool(name="USG6000", tool_category="安全设备", vendor="华为")
    tool3 = Tool(name="Python", tool_category="编程语言")

    builder.add_node(task1)
    builder.add_node(task2)
    builder.add_node(project1)
    builder.add_node(project2)
    builder.add_node(skill1)
    builder.add_node(skill2)
    builder.add_node(skill3)
    builder.add_node(tool1)
    builder.add_node(tool2)
    builder.add_node(tool3)

    # 添加边
    builder.add_edge(project1.id, skill1.id, RelationType.COMPOSED_OF)
    builder.add_edge(project1.id, skill3.id, RelationType.COMPOSED_OF)
    builder.add_edge(project2.id, skill2.id, RelationType.COMPOSED_OF)

    builder.add_edge(skill1.id, task1.id, RelationType.MAPS_TO)
    builder.add_edge(skill2.id, task2.id, RelationType.MAPS_TO)

    builder.add_edge(skill1.id, tool1.id, RelationType.OPERATES)
    builder.add_edge(skill2.id, tool2.id, RelationType.OPERATES)
    builder.add_edge(skill3.id, tool3.id, RelationType.OPERATES)

    print_graph_statistics(builder)

    output_dir = project_root / "data" / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "graph_visualization_demo.png"
    visualize_graph(builder, output_path=str(output_path))
    print(f"\n演示图谱已保存到: {output_path}")


if __name__ == "__main__":
    main()
