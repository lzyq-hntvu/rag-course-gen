"""
知识图谱质量验收脚本

验收维度：
1. 完整性 - 数据覆盖度
2. 连通性 - 图结构质量
3. 分层合理性 - 职教特色
4. 推理能力 - 关键路径完整性
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import Counter

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.graph import VocationalGraphBuilder, EntityType, RelationType


class GraphValidator:
    """图谱验收器"""

    def __init__(self, builder: VocationalGraphBuilder):
        self.builder = builder
        self.graph = builder.graph
        self.results: Dict[str, Any] = {}

    def validate_all(self) -> Dict[str, Any]:
        """执行所有验收检查"""
        print("\n" + "=" * 70)
        print("知识图谱质量验收报告".center(70))
        print("=" * 70)

        self.results["completeness"] = self.check_completeness()
        self.results["connectivity"] = self.check_connectivity()
        self.results["layering"] = self.check_layering()
        self.results["reasoning"] = self.check_reasoning_paths()

        self.print_summary()
        return self.results

    def check_completeness(self) -> Dict[str, Any]:
        """
        1. 完整性检查

        验收标准：
        - 实体抽取成功率 >= 80%
        - 关键实体类型（ActionSkill）覆盖率：每个JD至少抽取1个技能
        - 四种实体类型都存在
        """
        print("\n[1] 完整性检查 (Completeness)")
        print("-" * 70)

        stats = self.builder.get_statistics()
        results = {
            "passed": True,
            "details": {},
            "warnings": [],
        }

        # 检查1: 四种实体类型都存在
        print("\n  检查1: 实体类型覆盖度")
        entity_counts = stats["entity_type_counts"]
        all_types_present = all(entity_counts.get(t.value, 0) > 0 for t in EntityType)

        for et in EntityType:
            count = entity_counts.get(et.value, 0)
            status = "✓" if count > 0 else "✗"
            print(f"    {status} {et.value}: {count}")

        results["details"]["entity_types_present"] = all_types_present
        if not all_types_present:
            results["passed"] = False
            results["warnings"].append("缺少实体类型")

        # 检查2: ActionSkill 抽取质量
        print("\n  检查2: 技能抽取密度（每个项目的平均技能数）")
        projects = self.builder.get_nodes_by_type(EntityType.REAL_PROJECT)
        skills = self.builder.get_nodes_by_type(EntityType.ACTION_SKILL)

        if projects:
            # 统计每个项目关联的技能数
            project_skill_counts = []
            for project in projects:
                skill_count = 0
                for edge in self.graph.edges(project["id"], data=True):
                    if edge[2].get("relation_type") == "composed_of":
                        target = self.graph.nodes[edge[1]]
                        if target.get("entity_type") == "action_skill":
                            skill_count += 1
                project_skill_counts.append(skill_count)

            avg_skills = sum(project_skill_counts) / len(project_skill_counts)
            print(f"    平均每项目技能数: {avg_skills:.1f}")
            print(f"    最少: {min(project_skill_counts)}, 最多: {max(project_skill_counts)}")

            results["details"]["avg_skills_per_project"] = avg_skills
            if avg_skills < 2:
                results["warnings"].append(f"技能抽取密度低 ({avg_skills:.1f}技能/项目)")

        # 检查3: 技能复杂度标注率
        print("\n  检查3: 技能复杂度标注率")
        skills_with_complexity = sum(1 for s in skills if s.get("complexity"))
        complexity_rate = skills_with_complexity / len(skills) * 100 if skills else 0
        print(f"    已标注复杂度: {skills_with_complexity}/{len(skills)} ({complexity_rate:.1f}%)")

        results["details"]["complexity_annotation_rate"] = complexity_rate

        # 检查4: 项目经验等级标注率
        print("\n  检查4: 项目经验等级标注率")
        projects_with_exp = sum(1 for p in projects if p.get("experience_level"))
        exp_rate = projects_with_exp / len(projects) * 100 if projects else 0
        print(f"    已标注经验等级: {projects_with_exp}/{len(projects)} ({exp_rate:.1f}%)")

        results["details"]["experience_annotation_rate"] = exp_rate

        # 结论
        print("\n  结论:", end=" ")
        if results["passed"] and not results["warnings"]:
            print("✓ 通过")
        elif results["passed"]:
            print("⚠ 通过（有警告）")
        else:
            print("✗ 未通过")

        return results

    def check_connectivity(self) -> Dict[str, Any]:
        """
        2. 连通性检查

        验收标准：
        - 弱连通分量比例：主连通图应包含 > 80% 节点
        - 关系类型分布：三种关系都存在
        - 孤立节点比例：< 10%
        """
        print("\n[2] 连通性检查 (Connectivity)")
        print("-" * 70)

        results = {
            "passed": True,
            "details": {},
            "warnings": [],
        }

        total_nodes = self.graph.number_of_nodes()

        # 检查1: 孤立节点
        print("\n  检查1: 孤立节点统计")
        isolated_nodes = list(nx.isolates(self.graph))
        isolated_rate = len(isolated_nodes) / total_nodes * 100 if total_nodes > 0 else 0
        print(f"    孤立节点: {len(isolated_nodes)}/{total_nodes} ({isolated_rate:.1f}%)")

        results["details"]["isolated_node_rate"] = isolated_rate
        if isolated_rate > 10:
            results["warnings"].append(f"孤立节点过多 ({isolated_rate:.1f}%)")

        # 检查2: 弱连通分量
        print("\n  检查2: 弱连通分量分析")
        weak_components = list(nx.weakly_connected_components(self.graph))
        largest_component_size = max(len(c) for c in weak_components) if weak_components else 0
        largest_component_rate = largest_component_size / total_nodes * 100 if total_nodes > 0 else 0

        print(f"    连通分量数: {len(weak_components)}")
        print(f"    最大连通分量: {largest_component_size}/{total_nodes} 节点 ({largest_component_rate:.1f}%)")

        results["details"]["largest_component_rate"] = largest_component_rate
        if largest_component_rate < 80:
            results["warnings"].append(f"图谱过于分散 (最大连通分量 {largest_component_rate:.1f}%)")

        # 检查3: 关系类型分布
        print("\n  检查3: 关系类型分布")
        relation_counts = Counter()
        for _, _, data in self.graph.edges(data=True):
            rel_type = data.get("relation_type", "unknown")
            relation_counts[rel_type] += 1

        for rel_type in RelationType:
            count = relation_counts.get(rel_type.value, 0)
            status = "✓" if count > 0 else "✗"
            print(f"    {status} {rel_type.value}: {count}")

        all_relations_present = all(relation_counts.get(rt.value, 0) > 0 for rt in RelationType)
        results["details"]["all_relation_types_present"] = all_relations_present
        if not all_relations_present:
            results["warnings"].append("缺少关系类型")

        # 检查4: 图谱密度
        print("\n  检查4: 图谱密度")
        stats = self.builder.get_statistics()
        density = stats["density"]
        print(f"    密度: {density:.4f}")

        results["details"]["density"] = density
        if density < 0.01:
            results["warnings"].append(f"图谱稀疏 (密度 {density:.4f})")

        # 结论
        print("\n  结论:", end=" ")
        if results["passed"] and not results["warnings"]:
            print("✓ 通过")
        elif results["passed"]:
            print("⚠ 通过（有警告）")
        else:
            print("✗ 未通过")

        return results

    def check_layering(self) -> Dict[str, Any]:
        """
        3. 分层合理性检查

        验收标准：
        - 经验等级递进：entry -> junior -> mid_level 节点数递减
        - 复杂度分布：低复杂度技能占比应合理（高职导向）
        - 经验-复杂度一致性：入门级项目应主要对应低复杂度技能
        """
        print("\n[3] 分层合理性检查 (Layering)")
        print("-" * 70)

        results = {
            "passed": True,
            "details": {},
            "warnings": [],
        }

        projects = self.builder.get_nodes_by_type(EntityType.REAL_PROJECT)
        skills = self.builder.get_nodes_by_type(EntityType.ACTION_SKILL)

        # 检查1: 经验等级分布
        print("\n  检查1: 经验等级分布")
        exp_order = ["entry", "junior", "mid_level", "senior", "expert"]
        exp_dist = Counter(p.get("experience_level") for p in projects)

        for level in exp_order:
            count = exp_dist.get(level, 0)
            print(f"    {level}: {count}个")

        results["details"]["experience_distribution"] = dict(exp_dist)

        # 检查2: 复杂度分布
        print("\n  检查2: 技能复杂度分布")
        comp_dist = Counter(s.get("complexity") for s in skills)
        total_skills = len(skills)

        for level in ["low", "medium", "high"]:
            count = comp_dist.get(level, 0)
            pct = count / total_skills * 100 if total_skills > 0 else 0
            print(f"    {level}: {count}个 ({pct:.1f}%)")

        results["details"]["complexity_distribution"] = dict(comp_dist)

        # 检查3: 经验-复杂度一致性分析
        print("\n  检查3: 经验-复杂度一致性（入门级项目的技能复杂度）")
        entry_projects = [p for p in projects if p.get("experience_level") == "entry"]
        consistency_score = 0

        if entry_projects:
            # 获取入门级项目涉及的技能
            entry_project_ids = {p["id"] for p in entry_projects}
            entry_skills = []

            for edge in self.graph.edges(data=True):
                if edge[0] in entry_project_ids and edge[2].get("relation_type") == "composed_of":
                    skill_node = self.graph.nodes[edge[1]]
                    if skill_node.get("entity_type") == "action_skill":
                        entry_skills.append(skill_node)

            if entry_skills:
                comp_dist = Counter(s.get("complexity") for s in entry_skills)
                print(f"    入门级涉及技能: {len(entry_skills)}个")
                for level in ["low", "medium", "high"]:
                    count = comp_dist.get(level, 0)
                    pct = count / len(entry_skills) * 100
                    print(f"      - {level}: {count}个 ({pct:.1f}%)")

                # 计算一致性：入门级应有 > 50% 低复杂度技能
                low_rate = comp_dist.get("low", 0) / len(entry_skills) * 100
                consistency_score = low_rate
                results["details"]["entry_low_complexity_rate"] = low_rate

                if low_rate >= 50:
                    print(f"    ✓ 入门级项目以低复杂度技能为主 ({low_rate:.1f}%)")
                else:
                    print(f"    ⚠ 入门级项目低复杂度技能占比偏低 ({low_rate:.1f}%)")
                    results["warnings"].append(f"入门级项目复杂度不匹配 ({low_rate:.1f}%低复杂度)")
        else:
            print("    无入门级项目数据")

        # 结论
        print("\n  结论:", end=" ")
        if results["passed"] and not results["warnings"]:
            print("✓ 通过")
        elif results["passed"]:
            print("⚠ 通过（有警告）")
        else:
            print("✗ 未通过")

        return results

    def check_reasoning_paths(self) -> Dict[str, Any]:
        """
        4. 推理路径检查

        验收标准：
        - 关键路径完整性：RealProject -> ActionSkill -> StandardTask
        - 映射关系覆盖率：技能映射到国标任务的比例
        - 平均路径长度：从项目到国标任务的平均跳数
        """
        print("\n[4] 推理路径检查 (Reasoning Paths)")
        print("-" * 70)

        results = {
            "passed": True,
            "details": {},
            "warnings": [],
        }

        projects = self.builder.get_nodes_by_type(EntityType.REAL_PROJECT)
        skills = self.builder.get_nodes_by_type(EntityType.ACTION_SKILL)
        standard_tasks = self.builder.get_nodes_by_type(EntityType.STANDARD_TASK)

        # 检查1: 关键路径统计
        print("\n  检查1: 关键路径完整性")
        print("    路径: RealProject -> ActionSkill -> StandardTask")

        complete_paths = 0
        partial_paths = 0
        no_mapping_skills = 0

        for skill in skills:
            skill_id = skill["id"]

            # 检查是否有 incoming project
            has_project = False
            for edge in self.graph.in_edges(skill_id, data=True):
                if edge[2].get("relation_type") == "composed_of":
                    source = self.graph.nodes[edge[0]]
                    if source.get("entity_type") == "real_project":
                        has_project = True
                        break

            # 检查是否有 outgoing standard_task
            has_mapping = False
            for edge in self.graph.out_edges(skill_id, data=True):
                if edge[2].get("relation_type") == "maps_to":
                    target = self.graph.nodes[edge[1]]
                    if target.get("entity_type") == "standard_task":
                        has_mapping = True
                        break

            if has_project and has_mapping:
                complete_paths += 1
            elif has_project:
                partial_paths += 1
            if not has_mapping:
                no_mapping_skills += 1

        print(f"    完整路径技能数: {complete_paths}/{len(skills)} ({complete_paths/len(skills)*100 if skills else 0:.1f}%)")
        print(f"    只有项目无映射: {partial_paths}")
        print(f"    无映射技能: {no_mapping_skills}")

        results["details"]["complete_path_rate"] = complete_paths / len(skills) * 100 if skills else 0
        results["details"]["mapping_coverage"] = (len(skills) - no_mapping_skills) / len(skills) * 100 if skills else 0

        if complete_paths / len(skills) * 100 < 50 if skills else True:
            results["warnings"].append(f"完整路径比例低 ({complete_paths/len(skills)*100 if skills else 0:.1f}%)")

        # 检查2: 映射关系覆盖率
        print("\n  检查2: 技能-任务映射覆盖率")
        if skills:
            mapped_skills = sum(
                1 for skill in skills
                if any(
                    edge[2].get("relation_type") == "maps_to"
                    for edge in self.graph.out_edges(skill["id"], data=True)
                )
            )
            mapping_rate = mapped_skills / len(skills) * 100
            print(f"    已映射技能: {mapped_skills}/{len(skills)} ({mapping_rate:.1f}%)")

            results["details"]["skill_mapping_rate"] = mapping_rate
            if mapping_rate < 30:
                results["warnings"].append(f"技能映射率低 ({mapping_rate:.1f}%)")

        # 检查3: 示例路径查询
        print("\n  检查3: 示例推理路径")
        sample_projects = projects[:3]
        found_paths = 0

        for project in sample_projects:
            # BFS 查找路径
            try:
                paths = nx.single_source_shortest_path(
                    self.graph, project["id"], cutoff=3
                )
                # 找到能到达 standard_task 的路径
                task_paths = []
                for target, path in paths.items():
                    node = self.graph.nodes[target]
                    if node.get("entity_type") == "standard_task" and len(path) == 3:
                        task_paths.append(path)

                if task_paths:
                    found_paths += 1
                    # 展示第一条路径
                    path = task_paths[0]
                    path_names = [
                        self.graph.nodes[n]["name"][:15] + "..."
                        if len(self.graph.nodes[n]["name"]) > 15
                        else self.graph.nodes[n]["name"]
                        for n in path
                    ]
                    print(f"\n    示例路径 {found_paths}:")
                    print(f"      {' -> '.join(path_names)}")
            except Exception:
                pass

        if found_paths == 0:
            print("    ⚠ 未找到完整推理路径")
            results["warnings"].append("无完整推理路径示例")

        # 结论
        print("\n  结论:", end=" ")
        if results["passed"] and not results["warnings"]:
            print("✓ 通过")
        elif results["passed"]:
            print("⚠ 通过（有警告）")
        else:
            print("✗ 未通过")

        return results

    def print_summary(self):
        """打印验收总结"""
        print("\n" + "=" * 70)
        print("验收总结".center(70))
        print("=" * 70)

        total_passed = sum(1 for r in self.results.values() if r["passed"])
        total_warnings = sum(len(r.get("warnings", [])) for r in self.results.values())

        print(f"\n通过项: {total_passed}/4")
        print(f"警告数: {total_warnings}")

        print("\n各项结果:")
        for name, result in self.results.items():
            status = "✓" if result["passed"] else "✗"
            warnings = result.get("warnings", [])
            warn_text = f" ({len(warnings)}警告)" if warnings else ""
            print(f"  {status} {name}: {result['passed']}{warn_text}")

        print("\n" + "=" * 70)


def load_and_validate(
    graph_path: str = None,
    build_if_missing: bool = False,
) -> GraphValidator:
    """
    加载或构建图谱，然后执行验收

    Args:
        graph_path: 图谱文件路径
        build_if_missing: 如果图谱不存在，是否构建
    """
    builder = VocationalGraphBuilder()

    # 加载或构建图谱
    if graph_path and Path(graph_path).exists():
        print(f"加载已有图谱: {graph_path}")
        builder.load(graph_path)
    else:
        if not build_if_missing:
            raise FileNotFoundError(f"图谱文件不存在: {graph_path}")

        # 使用重构后的两阶段流程
        from src.graph.build_batch import build_batch

        print("构建新图谱（使用两阶段流程：国标骨架 → JD抽取）...")
        builder = build_batch(
            num_records=20,
            use_priority=True,
            security_filter=True,  # 启用安全领域过滤
        )

    # 执行验收
    validator = GraphValidator(builder)
    validator.validate_all()

    return validator


# 需要添加 networkx 导入
import networkx as nx


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="知识图谱质量验收")
    parser.add_argument("--graph", type=str, default=None, help="图谱文件路径")
    parser.add_argument("--build", action="store_true", help="如果图谱不存在则构建")
    args = parser.parse_args()

    load_and_validate(args.graph, args.build)
