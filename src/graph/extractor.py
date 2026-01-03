"""
LLM实体抽取器

使用Few-Shot Prompting从JD文本或国标文本中抽取职教能力图谱实体和关系
支持智谱AI (GLM) 和其他OpenAI兼容接口
"""

import os
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

from .schema import (
    StandardTask,
    RealProject,
    ActionSkill,
    Tool,
    GraphEdge,
    RelationType,
    EntityType,
)

# 加载环境变量
load_dotenv()


class ExtractedEntities(BaseModel):
    """抽取结果的数据模型"""
    standard_tasks: List[Dict[str, Any]] = []
    real_projects: List[Dict[str, Any]] = []
    action_skills: List[Dict[str, Any]] = []
    tools: List[Dict[str, Any]] = []
    relations: List[Dict[str, Any]] = []


# ============================================================================
# System Prompt
# ============================================================================
EXTRACTOR_SYSTEM_PROMPT = """你是职教领域的知识抽取专家。你的任务是从招聘需求（JD）或国家职业标准文本中，精确抽取结构化的"职教能力图谱"实体和关系。

## 背景知识

职教能力图谱的核心目标：建立"企业岗位需求"到"国家教学标准"的映射，用于生成符合国标的企业定制化课程。

## 核心实体类型

### 1. Standard_Task（典型工作任务）
- 定义：来自国家职业标准，描述一个完整的职业行动领域
- 特征：通常是"动词+名词"结构，如"网络系统构建"、"安全防护部署"
- 属性：name（必填）, task_code, knowledge_points, skill_points, career_level

### 2. Real_Project（真实项目载体）
- 定义：来自JD或工程案例，描述一个真实的工作场景/项目
- 特征：具体、可操作，如"某三甲医院网络改造工程"
- 属性：name（必填）, company_name, project_context, difficulty_level, **experience_level**

### 3. Action_Skill（行动技能）
- 定义：连接项目和标准的桥梁，描述具体的操作能力
- 特征：精细粒度的技能描述，如"配置OSPF协议"、"编写Python脚本"
- 属性：name（必填）, skill_category, proficiency_level, prerequisites, **complexity**

### 4. Tool（工具/设备）
- 定义：完成技能所需的工具、设备、软件、技术等
- 特征：具体的产品名或技术名，如"华为S5700交换机"、"Python"
- 属性：name（必填）, tool_category, vendor, version

## 分层规则（重要！用于职业教育分层培养）

### experience_level（项目经验等级）
- **entry**：应届生、0-1年、实习、助理、"协助"、"参与"
- **junior**：1-3年、"初级"
- **mid_level**：3-5年、"中级"
- **senior**：5-8年、"高级"、"资深"
- **expert**：8年以上、"专家"、"架构师"、"技术负责人"

### complexity（技能复杂度）
- **low**：执行类、基础操作、"协助"、"编写简单"、"维护"
- **medium**：独立完成、常规开发、"开发"、"实现"、"配置"
- **high**：架构设计、性能优化、疑难解决、"设计"、"优化"、"规划"、"架构"

## 核心关系类型

### 1. composed_of（组成关系）
- 方向：Real_Project → Action_Skill
- 含义：一个项目由哪些具体技能组成
- 示例：某医院网络改造 --composed_of--> 配置VLAN

### 2. maps_to（映射关系）
- 方向：Action_Skill → Standard_Task
- 含义：企业技能映射到国标任务（这是核心推理边）
- 示例：配置H3C交换机 --maps_to--> 网络设备配置
- 注意：这里需要进行"泛化映射"，把具体技能抽象到国标层级

### 3. operates（操作关系）
- 方向：Action_Skill → Tool
- 含义：执行某技能需要使用的工具
- 示例：配置OSPF协议 --operates--> Cisco路由器

## 抽取原则（重要）

1. **能力导向**：专注于技术能力和技能要求，忽略薪资、福利、补贴等非技能信息
2. **去版本化**：将过时的具体版本泛化为通用名称
   - "Python 2.7" → "Python"
   - "Java 1.8" → "Java"
   - "Django 1.x" → "Django"
3. **精确性优先**：不要抽取模糊或过于宽泛的内容
4. **保持层级**：项目 → 技能 → 任务，形成清晰的推理链
5. **关注可操作性**：技能应该是可教学、可评估的
6. **标准化命名**：使用行业公认的术语
7. **分层标注**：根据经验要求和技能复杂度准确标注层级

## 输出格式

请严格按照JSON格式输出，包含以下字段：
```json
{
  "standard_tasks": [...],
  "real_projects": [...],
  "action_skills": [...],
  "tools": [...],
  "relations": [...]
}
```

每个实体和关系都应包含所需的属性字段。**务必为Real_Project标注experience_level，为Action_Skill标注complexity！**
"""

# ============================================================================
# Few-Shot 示例
# ============================================================================
FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": """以下是某公司的网络工程师招聘JD：

职位名称：网络运维工程师（1-3年经验）
岗位职责：
1. 负责企业园区网络的日常运维和故障处理
2. 配置和管理华为、H3C等品牌交换机和路由器
3. 实施网络安全策略，配置防火墙规则
4. 使用Python编写自动化运维脚本
5. 协助完成医院、学校等客户的网络改造项目

任职要求：
1. 熟悉TCP/IP协议栈，掌握OSPF、BGP等路由协议
2. 精通华为S5700、USG6000等设备配置
3. 熟悉Python，能使用Netmiko库进行设备管理
4. 具备CCNP或同等证书者优先

薪资：面议
福利：五险一金、年终奖、带薪年假
"""
        },
    {
        "role": "assistant",
        "content": """```json
{
  "standard_tasks": [
    {
      "name": "网络设备配置",
      "task_code": "T-01-001",
      "knowledge_points": ["TCP/IP协议", "路由协议原理"],
      "skill_points": ["设备选型", "接口配置", "路由配置"],
      "career_level": "中级工"
    },
    {
      "name": "网络安全防护",
      "task_code": "T-02-003",
      "knowledge_points": ["安全策略设计", "访问控制原理"],
      "skill_points": ["防火墙配置", "VPN配置", "入侵检测"],
      "career_level": "高级工"
    },
    {
      "name": "网络自动化运维",
      "task_code": "T-03-005",
      "knowledge_points": ["编程基础", "网络协议"],
      "skill_points": ["脚本开发", "工具使用", "故障诊断"],
      "career_level": "高级工"
    }
  ],
  "real_projects": [
    {
      "name": "医院园区网络改造项目",
      "company_name": "某科技公司",
      "project_context": "为客户医院提供从传统网络到智能网络的升级改造",
      "difficulty_level": "中等",
      "experience_level": "junior"
    },
    {
      "name": "企业网络安全策略实施项目",
      "company_name": "某科技公司",
      "project_context": "为园区网部署多层次安全防护体系",
      "difficulty_level": "较高",
      "experience_level": "junior"
    }
  ],
  "action_skills": [
    {
      "name": "配置OSPF路由协议",
      "skill_category": "网络配置",
      "proficiency_level": "掌握",
      "complexity": "medium"
    },
    {
      "name": "配置华为S5700交换机",
      "skill_category": "网络设备配置",
      "proficiency_level": "精通",
      "complexity": "medium"
    },
    {
      "name": "配置H3C交换机VLAN",
      "skill_category": "网络配置",
      "proficiency_level": "熟悉",
      "complexity": "low"
    },
    {
      "name": "配置华为USG6000防火墙",
      "skill_category": "安全配置",
      "proficiency_level": "掌握",
      "complexity": "medium"
    },
    {
      "name": "编写Python自动化脚本",
      "skill_category": "自动化运维",
      "proficiency_level": "熟悉",
      "prerequisites": ["Python基础语法", "网络设备操作"],
      "complexity": "medium"
    }
  ],
  "tools": [
    {
      "name": "华为S5700交换机",
      "tool_category": "网络设备",
      "vendor": "华为"
    },
    {
      "name": "H3C交换机",
      "tool_category": "网络设备",
      "vendor": "H3C"
    },
    {
      "name": "华为USG6000防火墙",
      "tool_category": "安全设备",
      "vendor": "华为"
    },
    {
      "name": "Python",
      "tool_category": "编程语言",
      "vendor": "Python Software Foundation"
    },
    {
      "name": "Netmiko库",
      "tool_category": "自动化工具",
      "vendor": "Napalm-automation"
    }
  ],
  "relations": [
    {
      "source": "医院园区网络改造项目",
      "target": "配置OSPF路由协议",
      "relation": "composed_of"
    },
    {
      "source": "医院园区网络改造项目",
      "target": "配置华为S5700交换机",
      "relation": "composed_of"
    },
    {
      "source": "医院园区网络改造项目",
      "target": "配置H3C交换机VLAN",
      "relation": "composed_of"
    },
    {
      "source": "企业网络安全策略实施项目",
      "target": "配置华为USG6000防火墙",
      "relation": "composed_of"
    },
    {
      "source": "企业网络安全策略实施项目",
      "target": "编写Python自动化脚本",
      "relation": "composed_of"
    },
    {
      "source": "配置华为S5700交换机",
      "target": "华为S5700交换机",
      "relation": "operates"
    },
    {
      "source": "配置H3C交换机VLAN",
      "target": "H3C交换机",
      "relation": "operates"
    },
    {
      "source": "配置华为USG6000防火墙",
      "target": "华为USG6000防火墙",
      "relation": "operates"
    },
    {
      "source": "编写Python自动化脚本",
      "target": "Python",
      "relation": "operates"
    },
    {
      "source": "配置华为S5700交换机",
      "target": "网络设备配置",
      "relation": "maps_to"
    },
    {
      "source": "配置H3C交换机VLAN",
      "target": "网络设备配置",
      "relation": "maps_to"
    },
    {
      "source": "配置华为USG6000防火墙",
      "target": "网络安全防护",
      "relation": "maps_to"
    },
    {
      "source": "编写Python自动化脚本",
      "target": "网络自动化运维",
      "relation": "maps_to"
    }
  ]
}
```
"""
    },
    {
        "role": "user",
        "content": """以下是高级网络架构师招聘JD：

职位名称：高级网络架构师（5年以上经验）
岗位职责：
1. 负责大规模数据中心网络架构设计
2. 规划和优化网络性能，制定技术路线图
3. 解决复杂的网络技术难题
4. 指导初中级工程师完成项目
5. 负责技术选型和方案评审

任职要求：
1. 8年以上网络架构设计经验
2. 精通SDN、VXLAN、EVPN等前沿技术
3. 具备大型网络规划和性能优化经验
4. 有团队管理经验，能指导他人工作
"""
        },
    {
        "role": "assistant",
        "content": """```json
{
  "standard_tasks": [
    {
      "name": "网络架构设计",
      "task_code": "T-06-001",
      "knowledge_points": ["网络架构理论", "前沿技术趋势"],
      "skill_points": ["架构规划", "技术路线设计", "方案评审"],
      "career_level": "技师"
    }
  ],
  "real_projects": [
    {
      "name": "数据中心网络架构项目",
      "company_name": "某互联网公司",
      "project_context": "设计支持大规模业务的数据中心网络架构",
      "difficulty_level": "高",
      "experience_level": "expert"
    }
  ],
  "action_skills": [
    {
      "name": "设计SDN网络架构",
      "skill_category": "网络架构设计",
      "proficiency_level": "精通",
      "complexity": "high"
    },
    {
      "name": "优化网络性能",
      "skill_category": "网络优化",
      "proficiency_level": "精通",
      "complexity": "high"
    },
    {
      "name": "解决复杂网络故障",
      "skill_category": "故障排查",
      "proficiency_level": "精通",
      "complexity": "high"
    },
    {
      "name": "指导团队技术工作",
      "skill_category": "团队管理",
      "proficiency_level": "精通",
      "complexity": "high"
    }
  ],
  "tools": [
    {
      "name": "SDN控制器",
      "tool_category": "网络设备",
      "vendor": "Cisco"
    }
  ],
  "relations": [
    {
      "source": "数据中心网络架构项目",
      "target": "设计SDN网络架构",
      "relation": "composed_of"
    },
    {
      "source": "设计SDN网络架构",
      "target": "网络架构设计",
      "relation": "maps_to"
    }
  ]
}
```
"""
    }
]


# ============================================================================
# EntityExtractor 类
# ============================================================================
class EntityExtractor:
    """基于LLM的职教能力图谱实体抽取器（支持智谱AI和OpenAI兼容接口）"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ):
        """
        初始化抽取器

        Args:
            api_key: API密钥（默认从环境变量OPENAI_API_KEY读取）
            base_url: API基础URL（默认从环境变量OPENAI_BASE_URL读取）
            model: 模型名称（默认从环境变量LLM_MODEL读取）
            temperature: 温度参数（0.0保证确定性输出）
        """
        # 从环境变量读取配置
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model or os.getenv("LLM_MODEL", "glm-4")

        if not self.api_key:
            raise ValueError(
                "API密钥未设置，请通过参数传入或设置OPENAI_API_KEY环境变量"
            )

        # 初始化OpenAI客户端（兼容智谱AI）
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        self.system_prompt = EXTRACTOR_SYSTEM_PROMPT

    def extract(
        self,
        text: str,
        source_document: str = "unknown",
        document_type: str = "jd",
        candidate_tasks: Optional[List[str]] = None,
    ) -> ExtractedEntities:
        """
        从文本中抽取实体和关系

        Args:
            text: 待抽取的文本内容
            source_document: 来源文档标识
            document_type: 文档类型（"jd" 或 "standard"）
            candidate_tasks: 候选标准任务列表（用于约束 JD 抽取时的映射关系）

        Returns:
            ExtractedEntities: 抽取结果
        """
        # 构建消息
        system_prompt = self.system_prompt

        # 如果提供了候选任务列表，在系统提示中注入约束
        if candidate_tasks and document_type == "jd":
            task_list = "\n".join(f"  - {task}" for task in candidate_tasks)
            constraint = f"""

## 【重要约束】国标任务骨架约束

你在抽取 JD 时，必须遵循以下规则：

1. **优先映射**：抽取的 Action_Skill 的 maps_to 关系必须优先指向以下已有的国标任务：

{task_list}

2. **命名规范**：创建标准任务时，必须使用上述列表中的确切名称，不要自行创造类似名称。
   - 正确：Action_Skill "配置防火墙规则" maps_to "防火墙配置"
   - 错误：Action_Skill "配置防火墙规则" maps_to "配置防火墙" （不在列表中）

3. **泛化匹配**：当 JD 中的技能涉及某个领域时，将其映射到对应的国标任务。
   - JD技能："配置华为USG6000防火墙" → 映射到 → "防火墙配置"
   - JD技能："部署入侵检测系统" → 映射到 → "入侵检测与防御"

4. **仅在无匹配时创建**：只有当上述列表中完全不存在相关任务时，才允许创建新的 Standard_Task。
"""
            system_prompt = self.system_prompt + constraint

        messages = [
            {"role": "system", "content": system_prompt}
        ]

        # 添加Few-Shot示例
        for example in FEW_SHOT_EXAMPLES:
            messages.append({
                "role": example["role"],
                "content": example["content"]
            })

        # 添加当前待抽取的文本
        task_prompt = f"""请从以下{"招聘需求（JD）" if document_type == "jd" else "国家职业标准"}文本中抽取职教能力图谱的实体和关系。

{text}
"""

        # 如果有候选任务约束，添加到任务提示中
        if candidate_tasks and document_type == "jd":
            task_list = "\n".join(f"  - {task}" for task in candidate_tasks)
            task_prompt += f"""

## 【核心要求】必须创建完整的图谱结构！

你必须创建以下三类实体和关系：

1. **Real_Project（真实项目）**：从JD中提取1-3个主要工作项目
2. **Action_Skill（行动技能）**：从JD中提取具体技能要求
3. **Tool（工具）**：从JD中提取使用的工具/技术
4. **关系**：必须创建三种类型的关系
   - composed_of: Real_Project -> Action_Skill（项目由哪些技能组成）
   - maps_to: Action_Skill -> Standard_Task（技能映射到国标任务）
   - operates: Action_Skill -> Tool（技能使用什么工具）

可用的标准任务列表（maps_to目标）：
{task_list}

完整示例：
{{
  "real_projects": [
    {{"name": "某公司网络安全改造项目", "experience_level": "junior"}}
  ],
  "action_skills": [
    {{"name": "配置防火墙", "complexity": "medium"}},
    {{"name": "使用Python开发脚本", "complexity": "medium"}}
  ],
  "tools": [
    {{"name": "Python"}},
    {{"name": "防火墙"}}
  ],
  "relations": [
    {{"source": "某公司网络安全改造项目", "target": "配置防火墙", "relation": "composed_of"}},
    {{"source": "某公司网络安全改造项目", "target": "使用Python开发脚本", "relation": "composed_of"}},
    {{"source": "配置防火墙", "target": "防火墙配置", "relation": "maps_to"}},
    {{"source": "使用Python开发脚本", "target": "设备配置调试", "relation": "maps_to"}},
    {{"source": "配置防火墙", "target": "防火墙", "relation": "operates"}},
    {{"source": "使用Python开发脚本", "target": "Python", "relation": "operates"}}
  ]
}}

**重要**：
1. 必须创建 real_projects、action_skills、tools 三种实体
2. 必须创建 composed_of、maps_to、operates 三种关系
3. maps_to 的目标必须从上述列表中选择
4. 不要创建新的 standard_tasks
"""

        task_prompt += """

请严格按照JSON格式输出，确保所有实体和关系的name字段能够相互匹配。注意：
1. 忽略薪资、福利等非技能信息
2. 将具体版本泛化（如"Python 2.7"改为"Python"）
3. 专注于技术能力和技能要求
4. 必须创建完整的实体和关系（projects、skills、tools及它们之间的关系）"""
        messages.append({"role": "user", "content": task_prompt})

        # 调用LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},  # 强制JSON输出
                max_tokens=4096,  # 确保输出不被截断
            )
            result_text = response.choices[0].message.content.strip()

            # 解析JSON
            result_dict = json.loads(result_text)

            # 添加来源文档信息
            for entity_list in [result_dict.get("standard_tasks", []),
                                result_dict.get("real_projects", []),
                                result_dict.get("action_skills", []),
                                result_dict.get("tools", [])]:
                for entity in entity_list:
                    entity["source_document"] = source_document

            return ExtractedEntities(**result_dict)

        except json.JSONDecodeError as e:
            raise ValueError(f"LLM返回的不是有效的JSON格式: {e}\n原始内容: {result_text}")
        except Exception as e:
            raise RuntimeError(f"实体抽取失败: {e}")

    def extract_batch(
        self,
        texts: List[str],
        source_documents: Optional[List[str]] = None,
        document_type: str = "jd",
    ) -> List[ExtractedEntities]:
        """
        批量抽取

        Args:
            texts: 待抽取的文本列表
            source_documents: 来源文档标识列表
            document_type: 文档类型

        Returns:
            List[ExtractedEntities]: 抽取结果列表
        """
        if source_documents is None:
            source_documents = [f"doc_{i}" for i in range(len(texts))]

        results = []
        for text, source_doc in zip(texts, source_documents):
            result = self.extract(text, source_doc, document_type)
            results.append(result)

        return results


# ============================================================================
# 工具函数
# ============================================================================
def entities_to_schema(
    extracted: ExtractedEntities,
    external_name_to_id: Dict[str, str] = None,
) -> tuple[list[StandardTask], list[RealProject], list[ActionSkill], list[Tool], list[GraphEdge]]:
    """
    将抽取结果转换为Schema定义的实体和边

    Args:
        extracted: 抽取结果
        external_name_to_id: 外部实体名称到ID的映射（用于处理骨架任务等外部实体）

    Returns:
        (standard_tasks, real_projects, action_skills, tools, edges)
    """
    # 构建实体字典，用于后续建立边的引用
    standard_tasks = []
    real_projects = []
    action_skills = []
    tools = []

    name_to_id = {}

    # 合并外部实体映射
    if external_name_to_id:
        name_to_id.update(external_name_to_id)

    # 转换StandardTask
    for task_data in extracted.standard_tasks:
        task = StandardTask(**task_data)
        standard_tasks.append(task)
        name_to_id[task.name] = task.id

    # 转换RealProject
    for project_data in extracted.real_projects:
        project = RealProject(**project_data)
        real_projects.append(project)
        name_to_id[project.name] = project.id

    # 转换ActionSkill
    for skill_data in extracted.action_skills:
        skill = ActionSkill(**skill_data)
        action_skills.append(skill)
        name_to_id[skill.name] = skill.id

    # 转换Tool
    for tool_data in extracted.tools:
        tool = Tool(**tool_data)
        tools.append(tool)
        name_to_id[tool.name] = tool.id

    # 构建边
    edges = []
    for rel_data in extracted.relations:
        try:
            # 检查必需字段
            if "source" not in rel_data or "target" not in rel_data or "relation" not in rel_data:
                continue

            source_name = rel_data["source"]
            target_name = rel_data["target"]
            relation_str = rel_data["relation"]

            # 跳过空值
            if not source_name or not target_name or not relation_str:
                continue

            if source_name not in name_to_id or target_name not in name_to_id:
                continue  # 跳过无法找到对应实体的关系

            edge = GraphEdge(
                source_id=name_to_id[source_name],
                target_id=name_to_id[target_name],
                relation_type=RelationType(relation_str),
            )
            edges.append(edge)
        except (ValueError, KeyError) as e:
            # 跳过无效的关系数据
            continue

    return standard_tasks, real_projects, action_skills, tools, edges
