import os
import re
import json
import hashlib
import asyncio
import logging
from typing import List, Optional, Dict
from dataclasses import dataclass

# ✅ 替换为 OpenAI SDK (DeepSeek 兼容)
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# ================= 配置区域 =================
API_KEY = "sk-114514"  # 替换你的 DeepSeek API Key
BASE_URL = "https://api.deepseek.com" # DeepSeek 官方地址

# 模型选择：
# deepseek-chat (即 DeepSeek-V3, 推荐，速度快，文采好)
# deepseek-reasoner (即 DeepSeek-R1, 逻辑强，但作为文学重写可能略慢且贵)
MODEL_NAME = "deepseek-chat"

# 【文件路径配置】
FILE_28_RAW = "hlm_28.txt"       # 待润色的后28回原始文件
TEMP_DIR = "temp_chapters_split" # 临时存放拆分后章节的目录
OUTPUT_DIR = "refined_chapters_pro_ds" # 输出目录改个名，区分一下
CACHE_FILE = "refiner_cache_ds.json"   # 缓存文件也区分一下

CHUNK_SIZE = 2000
CONCURRENCY_LIMIT = 3  # DeepSeek API 限速较为严格，如遇 429 请调低此数值 (如 2 或 3)

# ================= 黄金风格锚点 (Golden Anchors) =================
GOLDEN_SAMPLES = """
【范例一·人物神态（脂粉气）】
宝玉听了，此时心下早已明白了，也不答言，只嘻嘻的笑。黛玉道：“你也不用哄我。从今以后，我也不敢亲近二爷，二爷也全当没有我这么个人，便是。”说着，那泪珠儿早扑簌簌滚下来了。

【范例二·场景描写（富贵锦绣）】
只见入门便是曲折游廊，阶下石子漫成甬路。上面小小两三间房舍，一明两暗，里面都是合着地步打就的床几椅案。从里间房内又得一小门，出去则是后院，有大株梨花兼着芭蕉。

【范例三·叙事转折（苍凉感）】
且说那贾雨村此时官复原职，正春风得意，哪里记得当年的葫芦案？正是：身后有余忘缩手，眼前无路想回头。这日正坐堂，忽听有人击鼓，你道是谁？
"""
# ================= 日志配置 =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("RedRefiner-DS")

@dataclass
class SegmentTask:
    chapter_title: str
    index: int
    total: int
    text: str

class RedChamberEngine:
    def __init__(self, api_key: str, base_url: str, model_name: str):
        # ✅ 初始化 OpenAI 异步客户端
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        
        self.system_prompt = (
            "你是一位红学造诣极深的文学家，正在将《红楼梦》癸酉本的粗糙底稿，重写为曹雪芹前八十回的精美正文。\n"
            "【绝对禁令】：\n"
            "1. 严禁修改原剧情走向、人物关系和核心事件。你的工作是'润色'而非'改编'。\n"
            "2. 严禁出现现代词汇（如：情绪、心理、立刻、此时）。\n"
            "【风格要求】：\n"
            "1. 还原'脂粉气'与'书卷气'并存的白话文风。\n"
            "2. 对话要补全神态（冷笑、低头、拭泪）。\n"
            "3. 章节过渡使用'且说'、'不题'、'看官听说'等传统话术。"
        )
        
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, str]:
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_cache(self):
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _get_hash(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def smart_split(self, text: str, limit: int) -> List[str]:
        raw_paragraphs = text.split('\n')
        segments = []
        current_chunk = []
        current_len = 0

        for para in raw_paragraphs:
            para = para.strip()
            if not para: continue
            
            # 保护对话和诗词不被切断
            is_dialogue_start = para.endswith("道：") or para.endswith("说道") or para.endswith("笑道")
            
            current_chunk.append(para)
            current_len += len(para)

            if current_len >= limit and not is_dialogue_start:
                segments.append("\n".join(current_chunk))
                current_chunk = []
                current_len = 0
        
        if current_chunk:
            segments.append("\n".join(current_chunk))
        return segments

    async def refine_segment_async(self, task: SegmentTask, semaphore: asyncio.Semaphore) -> Optional[str]:
        input_hash = self._get_hash(task.text)
        if input_hash in self.cache:
            return self.cache[input_hash]

        user_content = f"""
{GOLDEN_SAMPLES}

==================================================
【当前进度】：正在重写《{task.chapter_title}》的第 {task.index}/{task.total} 部分。

【原始底稿】：
{task.text}

【重写指令】：
请以此底稿为骨架，重塑血肉。保留所有情节，仅修改语言风格。
请直接输出重写后的正文，无需任何解释或前缀。
"""
        retries = 3
        async with semaphore:
            for attempt in range(retries):
                try:
                    # ✅ DeepSeek (OpenAI) 调用方式
                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": user_content}
                        ],
                        temperature=1.0, 
                        max_tokens=8192,
                        stream=False
                    )
                    
                    result_text = response.choices[0].message.content
                    if result_text:
                        self.cache[input_hash] = result_text
                        return result_text
                        
                except Exception as e:
                    wait_time = 2 ** attempt
                    logger.warning(f"[{task.chapter_title}-{task.index}] 重试({attempt+1}/{retries}): {e}")
                    await asyncio.sleep(wait_time)
            
            logger.error(f"[{task.chapter_title}-{task.index}] ❌ 彻底失败")
            return None

    async def process_chapter(self, file_path: str):
        if not os.path.exists(file_path): return

        filename = os.path.basename(file_path)
        title = filename.split('.')[0]
        output_file = os.path.join(OUTPUT_DIR, f"refined_{filename}")
        
        if os.path.exists(output_file):
            logger.info(f"⏩ 跳过已存在: {filename}")
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        raw_segments = self.smart_split(content, CHUNK_SIZE)
        tasks = [SegmentTask(title, i+1, len(raw_segments), seg) for i, seg in enumerate(raw_segments)]

        logger.info(f"🚀 开始处理: {title} (共 {len(tasks)} 片段)")
        semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
        
        results = await tqdm_asyncio.gather(
            *[self.refine_segment_async(t, semaphore) for t in tasks],
            desc=f"Refining {title}"
        )

        if None in results:
            logger.error(f"❌ {title} 存在失败片段，跳过保存。")
            return

        full_text = f"{title}\n\n" + "\n".join(results)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        self._save_cache()
        logger.info(f"💾 {title} 保存成功。")

    async def run_batch(self, file_list: List[str]):
        if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
        for file_path in file_list:
            await self.process_chapter(file_path)

# ================= 核心辅助函数：拆分大文件（V3.1 终极修正版） =================
def split_big_file_to_temp(raw_file: str, output_dir: str) -> List[str]:
    """
    【V3.1 终极修正版】
    1. 使用正则行首匹配，解决“批语误切”问题。
    2. 加入【智能去重】，解决“重复标题导致文件名错位”问题。
    """
    if not os.path.exists(raw_file):
        logger.error(f"❌ 找不到源文件: {raw_file}")
        return []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(raw_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 正则：匹配行首的“第xx回”，且忽略正文中的引用
    pattern = re.compile(r"(^第[一二三四五六七八九十百]+回\s+.+?)(?=^第[一二三四五六七八九十百]+回|\Z)", re.DOTALL | re.MULTILINE)
    
    matches = list(pattern.finditer(content))
    generated_files = []
    
    # 【去重关键】
    seen_chapters = set()
    real_index = 0
    
    if not matches:
        logger.warning("⚠️ 未检测到标准回目，兜底处理。")
        out_path = os.path.join(output_dir, "chapter_full.txt")
        with open(out_path, "w", encoding='utf-8') as f: f.write(content)
        generated_files.append(out_path)
    else:
        logger.info(f"📖 扫描到 {len(matches)} 个片段，正在执行清洗去重...")
        for m in matches:
            chapter_content = m.group(1)
            first_line = chapter_content.strip().split('\n')[0]
            
            # 提取回目号（如“第九十六回”）用于查重
            chapter_num_match = re.search(r"(第[一二三四五六七八九十百]+回)", first_line)
            if not chapter_num_match: continue
            chapter_num = chapter_num_match.group(1)
            
            # 1. 查重
            if chapter_num in seen_chapters:
                logger.warning(f"🗑️ 丢弃重复/脚注片段: {first_line[:20]}...")
                continue
            
            # 2. 查空（过滤只有标题没有正文的废料）
            if len(chapter_content) < 50:
                logger.warning(f"🗑️ 丢弃过短片段: {first_line[:20]}...")
                continue
                
            seen_chapters.add(chapter_num)
            
            # 3. 文件名生成
            safe_name = re.sub(r'[\\/*?:"<>|]', "", first_line[:30]).strip()
            safe_name = safe_name.replace(" ", "_").replace("[^1]", "")
            
            fname = f"{real_index + 81:03d}_{safe_name}.txt"
            out_path = os.path.join(output_dir, fname)
            
            with open(out_path, "w", encoding='utf-8') as f:
                f.write(chapter_content)
            generated_files.append(out_path)
            real_index += 1
            
    return generated_files

# ================= 主流程 =================
async def main():
    if API_KEY.startswith("xxxxx"):
        print("❌ 请先填写 DeepSeek API Key")
        return

    # 1. 准备引擎
    engine = RedChamberEngine(API_KEY, BASE_URL, MODEL_NAME)

    # 2. 拆分原始文件（使用带去重的 V3.1 逻辑）
    chapter_files = split_big_file_to_temp(FILE_28_RAW, TEMP_DIR)
    
    if not chapter_files:
        print("❌ 没有可处理的文件。")
        return
        
    print(f"✅ 成功拆分出 {len(chapter_files)} 个有效章节，准备开始生成...")

    # 3. 批量执行
    await engine.run_batch(chapter_files)

# --- 标准 Python 脚本入口 ---
if __name__ == "__main__":
    # 使用 asyncio.run() 来运行顶层的异步函数
    # 这样就可以正确地启动事件循环并执行 main()
    asyncio.run(main())