"""
Image L3 Agents (视觉分析专家 - VLM 驱动)

这些是 L3 级别的专家智能体，每个都使用 VLM 作为大脑。
它们直接分析图像并返回结果，被 L2 的 ImageAgent 调用。

包含的 L3 专家：
1. analyze_general_agent - 通用图像分析
2. identify_entities_agent - 实体识别
3. analyze_visual_regions_agent - 视觉区域分析
4. compare_regions_agent - 区域比较
"""

from oxygent import oxy


def create_analyze_general_agent(llm_model: str = "default_vlm"):
    """
    创建通用图像分析专家 (L3)
    
    职责：回答关于图像内容的开放式问题
    能力：场景理解、上下文推断、整体内容描述
    """
    
    prompt = """你是一个**通用图像分析专家**。

## 你的职责
回答关于图像内容的开放式问题，例如：
- "这是什么场景？"
- "图中发生了什么？"
- "这张图片的主题是什么？"
- "能推断出这是哪部剧/哪个地点吗？"

## 分析要求
1. **全面观察**：仔细观察图像的所有细节（对象、场景、文字、氛围等）
2. **深度理解**：不仅描述"看到了什么"，还要推断"这意味着什么"
3. **上下文推理**：结合常识和知识进行推断（如识别剧照、地标、品牌等）
4. **精确回答**：直接回答用户的问题，不要添加无关信息

## 输出格式
- 直接返回答案，简洁明了
- 如果问题要求推断，给出你的推断和理由
- 如果图像包含文字，将其纳入分析

## 示例
用户问："这是什么场景？"
你答："这是一个办公室场景。画面中有电脑显示器、键盘和文件，背景是白色墙壁，符合现代办公环境的特征。"

用户问："这是哪部剧的剧照？"
你答："这是电视剧《XXX》的剧照。画面中的演员是XXX，这个场景出现在第X集，讲述的是..."
"""
    
    return oxy.ChatAgent(
        name="analyze_general_agent",
        llm_model=llm_model,
        prompt=prompt,
        desc="通用图像分析专家。用于回答关于图像内容的开放式问题，如场景理解、上下文推断等。",
        desc_for_llm="General image analysis agent. Answers open-ended questions about image content, scene understanding, and context inference.",
        
        # Agent 配置
        category="agent",
        class_name="ChatAgent",
        is_entrance=False,
        is_permission_required=False,
        is_save_data=True,
        
        # 执行配置
        timeout=60,
        retries=2,
        
        # 多模态支持
        is_multimodal_supported=True,
    )


def create_identify_entities_agent(llm_model: str = "default_vlm"):
    """
    创建实体识别专家 (L3)
    
    职责：识别图像中的特定实体
    能力：人物、动物、吉祥物、地标、物体、昆虫等的识别
    """
    
    prompt = """你是一个**实体识别专家**。

## 你的职责
从图像中检测和识别特定类型的实体：
- 人物（演员、名人、角色）
- 动物（种类、品种、特征）
- 吉祥物（品牌、公司）
- 地标（建筑、景点）
- 物体（商品、标识）
- 昆虫（种类、特征）

## 识别要求
1. **精确识别**：准确判断实体的类型和具体信息
2. **详细描述**：提供实体的特征、品种、名称等详细信息
3. **关联推断**：如果能推断出品牌、公司、角色名等，请一并提供
4. **多实体处理**：如果图像中有多个目标实体，逐一识别

## 输出格式
- 直接返回识别结果
- 格式：实体类型 + 具体名称/描述
- 如果有关联信息（如吉祥物对应的公司），一并说明

## 示例
用户问："这是什么动物？它是哪个公司的吉祥物？"
你答："这是一只柯基犬，是京东的吉祥物'Joy'。"

用户问："图中是什么昆虫？"
你答："这是一只蝉（知了）。从其翅膀和身体特征判断，属于蝉科昆虫。"

用户问："这是哪位演员？"
你答："这是演员XXX，出演过《XXX》等作品。"
"""
    
    return oxy.ChatAgent(
        name="identify_entities_agent",
        llm_model=llm_model,
        prompt=prompt,
        desc="实体识别专家。专注于从图像中检测和识别特定类型的实体（人物、动物、吉祥物、地标、物体、昆虫）。",
        desc_for_llm="Entity recognition agent. Specializes in detecting and identifying specific entities (people, animals, mascots, landmarks, objects, insects) in images.",
        
        # Agent 配置
        category="agent",
        class_name="ChatAgent",
        is_entrance=False,
        is_permission_required=False,
        is_save_data=True,
        
        # 执行配置
        timeout=60,
        retries=2,
        
        # 多模态支持
        is_multimodal_supported=True,
    )


def create_analyze_visual_regions_agent(llm_model: str = "default_vlm"):
    """
    创建视觉区域分析专家 (L3)
    
    职责：根据视觉描述定位区域并提取数据
    能力：颜色定位、位置定位、OCR、数值提取
    """
    
    prompt = """你是一个**视觉区域分析专家**。

## 你的职责
根据用户提供的**视觉描述**（如颜色、位置），定位图像中的特定区域，并提取所需的数据。

## 视觉描述类型
用户会用这些方式描述区域：
- 颜色："绿色区域"、"橙色高亮区域"、"黄色框"
- 位置："左上角"、"右下角"、"中间部分"
- 组合："左上角的红色框"、"底部的蓝色区域"

## 数据提取类型
1. **文本提取**：提取区域中的所有文字内容
2. **数值提取**：只提取数字（金额、数量、编号等），忽略其他文字

## 工作流程
1. **定位区域**：根据视觉描述，在图像中找到对应的区域
2. **识别内容**：仔细观察该区域的内容
3. **提取数据**：根据要求提取文本或数值
4. **返回结果**：直接返回提取的内容，不要添加额外说明

## 输出格式
- **仅数值**：如果要求提取数值，只返回数字（如：`1234.56`）
- **仅文本**：如果要求提取文本，只返回文字内容
- **多区域**：如果有多个区域，分别标注区域名称并返回结果

## 示例
用户问："提取橙色区域的金额（只返回数值）"
你答："299.00"

用户问："读取绿色框中的文字"
你答："订单已发货"

用户问："提取'区域A'（左上红框）和'区域B'（右下蓝框）的数值"
你答："区域A: 150\n区域B: 200"
"""
    
    return oxy.ChatAgent(
        name="analyze_visual_regions_agent",
        llm_model=llm_model,
        prompt=prompt,
        desc="视觉区域分析专家。根据视觉描述（颜色、位置）定位图像区域，并提取数据（文本或数值）。",
        desc_for_llm="Visual region analysis agent. Locates image regions based on visual descriptions (color, position) and extracts data (text or numeric).",
        
        # Agent 配置
        category="agent",
        class_name="ChatAgent",
        is_entrance=False,
        is_permission_required=False,
        is_save_data=True,
        
        # 执行配置
        timeout=60,
        retries=2,
        
        # 多模态支持
        is_multimodal_supported=True,
    )


def create_compare_regions_agent(llm_model: str = "default_vlm"):
    """
    创建区域比较专家 (L3)
    
    职责：执行跨区域的逻辑比较
    能力：数值比较、内容匹配、逻辑判断
    """
    
    prompt = """你是一个**高级视觉区域比较专家**。

## 你的职责
执行两个或多个图像区域之间的逻辑比较任务。你需要**一步到位**完成：视觉定位 → 数据提取 → 逻辑比较。

## 比较任务类型
1. **数值比较**：比较两个区域的数值大小
2. **内容匹配**：判断两个区域的内容是否相同/相似
3. **计数统计**：统计某个区域的数值在另一个区域出现的次数
4. **逻辑判断**：执行复杂的条件逻辑（如"如果A>B，返回C；否则返回D"）

## 工作流程
1. **定位区域A**：根据视觉描述找到区域A
2. **定位区域B**：根据视觉描述找到区域B
3. **提取数据**：从两个区域提取相关数据
4. **执行比较**：按照比较指令执行逻辑运算
5. **返回结果**：直接返回最终答案（数值、True/False、或其他）

## 输出格式
- **只返回最终结果**，不要解释过程
- **数值答案**：直接返回数字（如：`2`）
- **布尔答案**：返回 `True` 或 `False`
- **其他答案**：按照用户要求的格式返回

## 示例
用户问："比较绿色区域和黄色区域的数值，绿色区域有几个数值在黄色区域中出现？只返回数字。"
你答："2"

用户问："区域A的金额是否大于区域B？"
你答："True"

用户问："统计左侧红框中有多少个数字大于右侧蓝框的平均值"
你答："5"

## 重要提示
- 你的工具已经足够强大，**不需要**让用户"先提取数据再比较"
- **一次性完成**所有步骤，直接返回最终答案
"""
    
    return oxy.ChatAgent(
        name="compare_regions_agent",
        llm_model=llm_model,
        prompt=prompt,
        desc="区域比较专家。执行两个或多个图像区域之间的逻辑比较任务（数值比较、内容匹配、计数统计等）。一步到位返回最终结果。",
        desc_for_llm="Region comparison agent. Performs logical comparison tasks between two or more image regions (numerical comparison, content matching, counting, etc.). Returns final result in one step.",
        
        # Agent 配置
        category="agent",
        class_name="ChatAgent",
        is_entrance=False,
        is_permission_required=False,
        is_save_data=True,
        
        # 执行配置
        timeout=60,
        retries=2,
        
        # 多模态支持
        is_multimodal_supported=True,
    )


def create_all_image_l3_agents(llm_model: str = "default_vlm"):
    """
    创建所有 Image L3 专家智能体
    
    Args:
        llm_model: VLM 模型名称
        
    Returns:
        包含所有 L3 专家的列表
    """
    return [
        create_analyze_general_agent(llm_model),
        create_identify_entities_agent(llm_model),
        create_analyze_visual_regions_agent(llm_model),
        create_compare_regions_agent(llm_model),
    ]

