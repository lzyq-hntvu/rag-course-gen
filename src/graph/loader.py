"""
数据加载器

从CSV文件加载JD数据，支持清洗和过滤
"""

import os
import csv
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class JobRecord:
    """职位记录"""
    position_name: str       # 职位名称
    position_detail: str     # 职位描述
    company: str             # 公司名称
    city: str                # 城市
    salary: str              # 薪资范围
    education: str           # 学历要求
    work_year: str           # 工作年限
    industry: str            # 行业
    source_file: str         # 来源文件

    def to_text(self) -> str:
        """转换为纯文本格式（用于LLM抽取）"""
        return f"""职位名称：{self.position_name}
公司名称：{self.company}
工作地点：{self.city}
薪资范围：{self.salary}
学历要求：{self.education}
工作年限：{self.work_year}
行业领域：{self.industry}

职位描述：
{self.position_detail}
"""


class JobDataLoader:
    """JD数据加载器"""

    # 默认的数据目录
    DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "job_spider" / "job_spider" / "WebSite" / "database" / "charts" / "csv_暂无用用处"

    # 优先使用的技术岗位
    PRIORITY_POSITIONS = [
        "python.csv",
        "java.csv",
        "cplusplus.csv",
        "c.csv",
        "go.csv",
        "javascript.csv",
        "node_js.csv",
        "php.csv",
        "android.csv",
        "ios.csv",
    ]

    # 安全领域关键词（用于领域过滤）
    SECURITY_KEYWORDS = [
        "安全", "漏洞", "防火墙", "渗透", "等级保护", "等保",
        "入侵检测", "IDS", "IPS", "VPN", "加密", "加密算法",
        "网络安全", "信息安全", "数据安全", "系统安全",
        "安全策略", "安全审计", "日志审计", "态势感知",
        "应急响应", "事件响应", "取证", "溯源",
        "WAF", "DDoS", "防病毒", "杀毒", "病毒防护",
        "HIDS", "NIDS", "SIEM", "SOC", "EDR",
        "风险评估", "威胁情报", "安全运营",
        "网络安全设备", "安全产品", "安全方案",
    ]

    def __init__(self, data_dir: Optional[Path] = None):
        """
        初始化加载器

        Args:
            data_dir: 数据目录，默认为项目内置目录
        """
        self.data_dir = data_dir or self.DEFAULT_DATA_DIR

    def list_available_files(self) -> List[str]:
        """列出可用的CSV文件（支持子目录）"""
        if not self.data_dir.exists():
            return []

        csv_files = []
        # 查找当前目录和所有子目录中的 CSV 文件
        for csv_file in self.data_dir.rglob("*.csv"):
            if csv_file.is_file():
                # 返回相对于 data_dir 的路径
                csv_files.append(str(csv_file.relative_to(self.data_dir)))
        return sorted(csv_files)

    def load_csv_file(
        self,
        filename: str,
        max_records: Optional[int] = None,
        random_sample: bool = False,
    ) -> List[JobRecord]:
        """
        加载单个CSV文件（支持子目录路径）

        Args:
            filename: CSV文件名（可以是相对路径，如 "lagou/python.csv"）
            max_records: 最大加载记录数（None表示全部）
            random_sample: 是否随机采样

        Returns:
            List[JobRecord]: 职位记录列表
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")

        records = []
        # 使用utf-8-sig处理BOM，并手动清理字段名
        with open(filepath, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            # 清理字段名（去除可能的空格和引号）
            field_map = {}
            for field in reader.fieldnames or []:
                cleaned = field.strip().strip('"')
                field_map[field] = cleaned

            for row in reader:
                # 创建清理后的行数据
                clean_row = {}
                for raw_field, value in row.items():
                    clean_field = field_map.get(raw_field, raw_field)
                    # 清理值（去除引号和多余空格），处理可能的列表类型
                    if isinstance(value, str):
                        clean_value = value.strip().strip('"').strip()
                    elif isinstance(value, list):
                        clean_value = str(value[0]) if value else ""
                    else:
                        clean_value = str(value).strip().strip('"').strip() if value else ""
                    clean_row[clean_field] = clean_value

                # 跳过空记录
                if not clean_row.get("jobDes") or not clean_row.get("positionName"):
                    continue

                record = JobRecord(
                    position_name=clean_row.get("positionName", ""),
                    position_detail=clean_row.get("jobDes", ""),
                    company=clean_row.get("company", ""),
                    city=clean_row.get("city", ""),
                    salary=clean_row.get("salary", ""),
                    education=clean_row.get("education", ""),
                    work_year=clean_row.get("workYear", ""),
                    industry=clean_row.get("industryField", ""),
                    source_file=filename,
                )
                records.append(record)

        # 随机采样
        if random_sample and max_records and len(records) > max_records:
            records = random.sample(records, max_records)
        # 或截取前N条
        elif max_records:
            records = records[:max_records]

        return records

    def load_multiple_files(
        self,
        filenames: List[str],
        max_records_per_file: Optional[int] = None,
        total_max_records: Optional[int] = None,
    ) -> List[JobRecord]:
        """
        加载多个CSV文件

        Args:
            filenames: CSV文件名列表
            max_records_per_file: 每个文件最大加载记录数
            total_max_records: 总最大记录数

        Returns:
            List[JobRecord]: 职位记录列表
        """
        all_records = []

        for filename in filenames:
            try:
                records = self.load_csv_file(filename, max_records_per_file)
                all_records.extend(records)
                if total_max_records and len(all_records) >= total_max_records:
                    all_records = all_records[:total_max_records]
                    break
            except FileNotFoundError:
                print(f"警告: 文件未找到，跳过: {filename}")
                continue

        return all_records

    def load_priority_positions(
        self,
        max_total_records: int = 100,
        records_per_file: int = 20,
    ) -> List[JobRecord]:
        """
        加载优先技术岗位数据

        Args:
            max_total_records: 总最大记录数
            records_per_file: 每个文件最大记录数

        Returns:
            List[JobRecord]: 职位记录列表
        """
        available_files = self.list_available_files()

        # 找到优先文件（支持子目录匹配）
        files_to_load = []
        for priority_file in self.PRIORITY_POSITIONS:
            # 检查文件是否在可用文件列表中（可能带子目录前缀）
            for avail_file in available_files:
                if avail_file.endswith(priority_file):
                    files_to_load.append(avail_file)
                    break

        if not files_to_load:
            print(f"警告: 未找到优先技术岗位文件，使用所有可用文件")
            files_to_load = available_files[:5]

        return self.load_multiple_files(
            files_to_load,
            max_records_per_file=records_per_file,
            total_max_records=max_total_records,
        )

    @staticmethod
    def filter_records(
        records: List[JobRecord],
        min_length: int = 50,
        keywords: Optional[List[str]] = None,
        exclude_keywords: Optional[List[str]] = None,
    ) -> List[JobRecord]:
        """
        过滤职位记录

        Args:
            records: 职位记录列表
            min_length: 职位描述最小长度
            keywords: 必须包含的关键词列表
            exclude_keywords: 必须排除的关键词列表

        Returns:
            List[JobRecord]: 过滤后的记录
        """
        filtered = []

        for record in records:
            # 检查长度
            if len(record.position_detail) < min_length:
                continue

            # 检查必须包含的关键词
            if keywords:
                detail_lower = record.position_detail.lower()
                if not any(kw.lower() in detail_lower for kw in keywords):
                    continue

            # 检查必须排除的关键词
            if exclude_keywords:
                detail_lower = record.position_detail.lower()
                if any(kw.lower() in detail_lower for kw in exclude_keywords):
                    continue

            filtered.append(record)

        return filtered


# ============================================================================
# 快捷函数
# ============================================================================
def load_job_records(
    count: int = 10,
    data_dir: Optional[Path] = None,
    use_priority: bool = True,
    security_filter: bool = False,
    min_records: int = 0,
) -> List[JobRecord]:
    """
    快捷加载JD记录

    Args:
        count: 加载记录数
        data_dir: 数据目录
        use_priority: 是否使用优先岗位
        security_filter: 是否启用安全领域关键词过滤
        min_records: 最小记录数（用于过滤后数量不足时的备用策略）

    Returns:
        List[JobRecord]: 职位记录列表
    """
    loader = JobDataLoader(data_dir)

    if use_priority:
        # 加载更多数据以应对过滤
        load_count = count * 5 if security_filter else count
        records = loader.load_priority_positions(
            max_total_records=load_count,
            records_per_file=min(50, load_count),
        )
    else:
        files = loader.list_available_files()[:3]
        records = loader.load_multiple_files(
            files,
            max_records_per_file=count // 3 + 1,
            total_max_records=count,
        )

    # 应用安全领域过滤
    if security_filter:
        before_count = len(records)
        records = JobDataLoader.filter_records(
            records,
            min_length=50,
            keywords=JobDataLoader.SECURITY_KEYWORDS,
        )
        print(f"  安全领域过滤: {before_count} → {len(records)} 条")
        # 过滤后数量不足的备用策略：返回所有可用记录
        if len(records) < min_records:
            print(f"  警告: 过滤后仅 {len(records)} 条，不足 {min_records} 条")
            records = records[:min_records]

    return records[:count]


def demo_data_loading():
    """演示数据加载功能"""
    print("JD数据加载器演示")
    print("=" * 50)

    loader = JobDataLoader()

    print("\n可用的CSV文件:")
    files = loader.list_available_files()
    for i, f in enumerate(files[:10]):
        print(f"  {i+1}. {f}")
    print(f"  ... 共 {len(files)} 个文件")

    print("\n加载优先技术岗位数据（前5条）:")
    records = loader.load_priority_positions(max_total_records=5, records_per_file=5)

    for i, record in enumerate(records):
        print(f"\n[{i+1}] {record.position_name} - {record.company}")
        print(f"    地点: {record.city} | 薪资: {record.salary}")
        print(f"    描述: {record.position_detail[:100]}...")

    return records


if __name__ == "__main__":
    demo_data_loading()
