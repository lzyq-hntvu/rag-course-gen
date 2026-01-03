"""
批量构建职教能力图谱并生成过滤视图

支持大规模数据处理和分层可视化
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, List, Literal
from collections import Counter

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import matplotlib
import networkx as nx

# 配置matplotlib支持中文
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from src.graph import (
    VocationalGraphBuilder,
    EntityExtractor,
    EntityType,
    RelationType,
    ExperienceLevel,
    SkillComplexity,
    GraphEdge,
    entities_to_schema,
)
from src.graph.loader import JobDataLoader, load_job_records


# 节点类型配色方案
ENTITY_TYPE_COLORS = {
    EntityType.STANDARD_TASK.value: "#4CAF50",
    EntityType.REAL_PROJECT.value: "#2196F3",
    EntityType.ACTION_SKILL.value: "#FF9800",
    EntityType.TOOL.value: "#9C27B0",
}

# 复杂度配色
COMPLEXITY_COLORS = {
    "low": "#81C784",      # 浅绿
    "medium": "#FFB74D",   # 橙色
    "high": "#E57373",     # 红色
    None: "#9E9E9E",       # 灰色（未标注）
}

# 经验等级配色
EXPERIENCE_COLORS = {
    "entry": "#A5D6A7",       # 浅绿
    "junior": "#81C784",      # 绿色
    "mid_level": "#FFB74D",   # 橙色
    "senior": "#FF8A65",      # 深橙
    "expert": "#E57373",      # 红色
    None: "#9E9E9E",          # 灰色
}


def print_sample_node(builder: VocationalGraphBuilder) -> None:
    """打印样本节点的完整JSON"""
    print("\n" + "=" * 60)
    print("样本节点（Sample Nodes）")
    print("=" * 60)

    # 按类型各选一个样本
    for entity_type in EntityType:
        nodes = builder.get_nodes_by_type(entity_type)
        if nodes:
            node = nodes[0]
            print(f"\n【{entity_type.value}】样本:")
            print(json.dumps(node, ensure_ascii=False, indent=2))


def print_layer_statistics(builder: VocationalGraphBuilder) -> None:
    """打印分层统计信息"""
    stats = builder.get_statistics()

    print("\n" + "=" * 60)
    print("分层统计信息（Layer Statistics）")
    print("=" * 60)

    # 按经验等级统计项目
    print("\n按经验等级分布（Projects by Experience Level）:")
    projects = builder.get_nodes_by_type(EntityType.REAL_PROJECT)
    exp_dist = Counter(p.get("experience_level") for p in projects)
    for level, count in sorted(exp_dist.items()):
        print(f"  - {level or '未标注'}: {count}个")

    # 按复杂度统计技能
    print("\n按复杂度分布（Skills by Complexity）:")
    skills = builder.get_nodes_by_type(EntityType.ACTION_SKILL)
    comp_dist = Counter(s.get("complexity") for s in skills)
    for level, count in sorted(comp_dist.items()):
        print(f"  - {level or '未标注'}: {count}个")

    # 交叉统计：经验-复杂度关联
    print("\n交叉统计：经验等级对应的技能复杂度分布:")
    for exp_level in [None, "entry", "junior", "mid_level", "senior", "expert"]:
        exp_projects = [p for p in projects if p.get("experience_level") == exp_level]
        if not exp_projects:
            continue

        # 获取这些项目涉及的技能
        project_ids = {p["id"] for p in exp_projects}
        related_skills = []
        for edge in builder.graph.edges(data=True):
            if edge[0] in project_ids and edge[2].get("relation_type") == "composed_of":
                skill_node = builder.graph.nodes[edge[1]]
                if skill_node["entity_type"] == "action_skill":
                    related_skills.append(skill_node)

        if related_skills:
            comp_dist = Counter(s.get("complexity") for s in related_skills)
            print(f"\n  {exp_level or '未标注'} (涉及{len(related_skills)}个技能):")
            for comp_level, count in sorted(comp_dist.items()):
                print(f"    - {comp_level or '未标注'}: {count}个")


def visualize_filtered(
    builder: VocationalGraphBuilder,
    output_path: Optional[str] = None,
    filter_complexity: Optional[Literal["low", "medium", "high"]] = None,
    filter_experience: Optional[Literal["entry", "junior", "mid_level", "senior", "expert"]] = None,
    figsize: tuple = (16, 12),
) -> None:
    """
    生成过滤视图的可视化

    Args:
        builder: 图谱构建器
        output_path: 输出路径
        filter_complexity: 按复杂度过滤（只显示该复杂度的技能相关节点）
        filter_experience: 按经验等级过滤（只显示该经验等级的项目相关节点）
        figsize: 图像大小
    """
    # 确定要保留的节点
    nodes_to_keep = set()

    if filter_complexity:
        # 获取指定复杂度的技能节点
        skills = [
            n for n, d in builder.graph.nodes(data=True)
            if d.get("entity_type") == "action_skill" and d.get("complexity") == filter_complexity
        ]
        nodes_to_keep.update(skills)

        # 获取相关联的项目和任务（1跳邻居）
        for skill in skills:
            neighbors = builder.graph.neighbors(skill)
            nodes_to_keep.update(neighbors)

    elif filter_experience:
        # 获取指定经验等级的项目节点
        projects = [
            n for n, d in builder.graph.nodes(data=True)
            if d.get("entity_type") == "real_project" and d.get("experience_level") == filter_experience
        ]
        nodes_to_keep.update(projects)

        # 获取相关的技能（1跳邻居）
        for project in projects:
            neighbors = builder.graph.neighbors(project)
            nodes_to_keep.update(neighbors)

    # 创建子图
    if nodes_to_keep:
        subgraph = builder.graph.subgraph(nodes_to_keep)
    else:
        subgraph = builder.graph

    # 限制显示节点数量
    max_nodes = 200
    if subgraph.number_of_nodes() > max_nodes:
        degrees = dict(subgraph.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        top_node_ids = [node[0] for node in top_nodes]
        subgraph = subgraph.subgraph(top_node_ids)

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 计算布局
    pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)

    # 按实体类型和复杂度/经验等级绘制节点
    for entity_type in EntityType:
        node_ids = [
            n for n, d in subgraph.nodes(data=True)
            if d.get("entity_type") == entity_type.value
        ]

        if not node_ids:
            continue

        # 根据不同类型使用不同颜色
        if entity_type == EntityType.ACTION_SKILL and filter_complexity:
            node_color = [COMPLEXITY_COLORS.get(
                subgraph.nodes[n].get("complexity"), "#9E9E9E"
            ) for n in node_ids]
        elif entity_type == EntityType.REAL_PROJECT and filter_experience:
            node_color = [EXPERIENCE_COLORS.get(
                subgraph.nodes[n].get("experience_level"), "#9E9E9E"
            ) for n in node_ids]
        else:
            node_color = ENTITY_TYPE_COLORS.get(entity_type.value, "#9E9E9E")

        nx.draw_networkx_nodes(
            subgraph,
            pos,
            nodelist=node_ids,
            node_color=node_color,
            node_size=300,
            alpha=0.8,
            ax=ax,
        )

    # 绘制边
    for relation_type, style in {
        "composed_of": {"style": "solid", "color": "#2196F3"},
        "maps_to": {"style": "dashed", "color": "#4CAF50"},
        "operates": {"style": "dotted", "color": "#9C27B0"},
    }.items():
        edges = [
            (u, v)
            for u, v, d in subgraph.edges(data=True)
            if d.get("relation_type") == relation_type
        ]
        if edges:
            nx.draw_networkx_edges(
                subgraph,
                pos,
                edgelist=edges,
                edge_color=style["color"],
                style=style["style"],
                alpha=0.5,
                arrowsize=20,
                arrowstyle="->",
                ax=ax,
            )

    # 绘制标签
    labels = nx.get_node_attributes(subgraph, "name")
    labels = {k: v[:12] + "..." if len(v) > 12 else v for k, v in labels.items()}
    nx.draw_networkx_labels(
        subgraph,
        pos,
        labels,
        font_size=7,
        font_family="sans-serif",
        ax=ax,
    )

    # 标题
    if filter_complexity:
        title = f"职教能力图谱 - 复杂度过滤视图（{filter_complexity}）"
    elif filter_experience:
        title = f"职教能力图谱 - 经验等级过滤视图（{filter_experience}）"
    else:
        title = "职教能力图谱 - 完整视图"
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  过滤视图已保存到: {output_path}")


def build_batch(
    num_records: int = 50,
    use_priority: bool = True,
    security_filter: bool = False,
    save_path: Optional[str] = None,
) -> VocationalGraphBuilder:
    """
    批量构建图谱（两阶段串行流程：国标骨架 → JD抽取）

    Args:
        num_records: 处理的JD数量
        use_priority: 是否使用优先技术岗位
        security_filter: 是否启用安全领域关键词过滤
        save_path: 图谱保存路径

    Returns:
        VocationalGraphBuilder: 构建好的图谱
    """
    print(f"批量构建职教能力图谱 ({num_records}条JD) - 两阶段串行流程")
    print("=" * 70)

    extractor = EntityExtractor()
    builder = VocationalGraphBuilder()

    # ================================================================
    # 阶段1：提取国标骨架（构建标准任务库）
    # ================================================================
    print(f"\n[阶段1/2] 提取国标骨架...")
    print("-" * 70)

    standard_file = project_root / "data" / "mock" / "sample_standard.txt"
    if not standard_file.exists():
        print(f"  警告: 国标文件不存在: {standard_file}")
        print(f"  将跳过国标骨架提取，使用无约束抽取模式")
        candidate_task_names = []
    else:
        with open(standard_file, "r", encoding="utf-8") as f:
            standard_text = f.read()

        print(f"  - 正在从国标文件提取标准任务...")
        try:
            standard_result = extractor.extract(
                standard_text,
                source_document="national_standard",
                document_type="standard"
            )
            standard_tasks, _, _, _, _ = entities_to_schema(standard_result)

            # 先将国标任务加入图谱
            builder.add_entities_from_extractor(standard_tasks, [], [], [], [])

            candidate_task_names = [task.name for task in standard_tasks]
            print(f"  - 成功提取 {len(candidate_task_names)} 个标准任务（骨架）")
            for task_name in candidate_task_names[:5]:
                print(f"    · {task_name}")
            if len(candidate_task_names) > 5:
                print(f"    ... 等共 {len(candidate_task_names)} 个任务")
        except Exception as e:
            print(f"  - 国标抽取失败: {e}")
            print(f"  将使用无约束抽取模式")
            candidate_task_names = []

    # ================================================================
    # 阶段2：用骨架约束抽取 JD
    # ================================================================
    print(f"\n[阶段2/2] 用骨架约束抽取 JD...")
    print("-" * 70)

    # 加载 JD 数据
    print(f"\n  步骤2.1: 加载 JD 数据...")
    job_records = load_job_records(
        count=num_records,
        use_priority=use_priority,
        security_filter=security_filter,
        min_records=num_records // 2,
    )
    print(f"    - 成功加载: {len(job_records)}条JD")

    # 批量抽取
    print(f"\n  步骤2.2: 批量抽取 JD（使用国标骨架约束）...")
    success_count = 0
    fail_count = 0

    for i, record in enumerate(job_records):
        print(f"    - 处理 [{i+1}/{len(job_records)}]: {record.position_name}")

        try:
            result = extractor.extract(
                record.to_text(),
                source_document=f"{record.source_file}_{i}",
                document_type="jd",
                candidate_tasks=candidate_task_names if candidate_task_names else None,
            )
            # 解析抽取结果
            # 获取骨架任务的 name_to_id 映射
            skeleton_task_name_to_id = {}
            for node in builder.graph.nodes(data=True):
                if node[1].get("entity_type") == "standard_task":
                    skeleton_task_name_to_id[node[1].get("name")] = node[0]

            standard_tasks, real_projects, action_skills, tools, edges = entities_to_schema(
                result,
                external_name_to_id=skeleton_task_name_to_id
            )

            # 调试：检查 LLM 是否创建了 maps_to 关系
            maps_to_count = sum(1 for e in edges if e.relation_type == RelationType.MAPS_TO)
            if len(standard_tasks) > 0 or maps_to_count > 0:
                print(f"      [调试] JD创建了{len(standard_tasks)}个标准任务, {maps_to_count}个映射关系")
                for task in standard_tasks[:3]:
                    print(f"        - {task.name}")

            # 过滤掉 JD 自创的标准任务（只保留骨架任务）
            # 所有边已经正确指向骨架任务（通过 external_name_to_id），直接保留
            valid_edges = edges

            # 只添加 real_projects, action_skills, tools（不接受 JD 创建的新 standard_tasks）
            builder.add_entities_from_extractor(
                [], real_projects, action_skills, tools, valid_edges
            )
            success_count += 1
            print(f"      成功: {len(action_skills)}个技能, {len(tools)}个工具")
        except Exception as e:
            fail_count += 1
            print(f"      跳过: {e}")
            continue

    print(f"\n    - 抽取完成: 成功{success_count}条, 失败{fail_count}条")

    # 保存图谱
    if save_path:
        builder.save(save_path)
        print(f"\n  - 图谱已保存到: {save_path}")

    return builder


def main():
    """主函数"""
    # 构建图谱（启用安全领域过滤）
    builder = build_batch(num_records=50, use_priority=True, security_filter=True)

    # 打印统计信息
    stats = builder.get_statistics()
    print(f"\n{'='*60}")
    print("图谱统计信息")
    print("=" * 60)
    print(f"总节点数: {stats['total_nodes']}")
    print(f"总边数: {stats['total_edges']}")
    print(f"图谱密度: {stats['density']:.4f}")

    print("\n实体类型分布:")
    for entity_type, count in stats["entity_type_counts"].items():
        print(f"  - {entity_type}: {count}")

    # 打印分层统计
    print_layer_statistics(builder)

    # 打印样本节点
    print_sample_node(builder)

    # 生成可视化
    output_dir = project_root / "data" / "output"
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print("生成可视化图像")
    print("=" * 60)

    # 完整视图（截取高连通度节点）
    visualize_filtered(
        builder,
        output_path=str(output_dir / "graph_full.png"),
        figsize=(20, 16),
    )
    print("  - 完整视图: data/output/graph_full.png")

    # 低复杂度过滤视图（高职人才核心技能）
    visualize_filtered(
        builder,
        filter_complexity="low",
        output_path=str(output_dir / "graph_complexity_low.png"),
        figsize=(16, 12),
    )
    print("  - 低复杂度视图: data/output/graph_complexity_low.png")

    # 入门级经验过滤视图
    visualize_filtered(
        builder,
        filter_experience="entry",
        output_path=str(output_dir / "graph_experience_entry.png"),
        figsize=(16, 12),
    )
    print("  - 入门级视图: data/output/graph_experience_entry.png")

    print(f"\n{'='*60}")
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
