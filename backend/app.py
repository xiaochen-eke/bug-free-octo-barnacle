"""
《饥荒 Don't Starve 游戏攻略助手》后端应用
垂直领域：独立游戏攻略 + 生存策略指导
"""

import requests

# 定义调用你自建 API 的函数
def fetch_game_data(category, name):
    """
    category: 对应你 API 的路径，如 'items', 'foods', 'recipe'
    name: 具体的名称，如 '肉丸'
    """
    url = f"http://127.0.0.1:5001/api/game-data/{category}/{name}"
    try:
        response = requests.get(url, timeout=5) # 设置 5 秒超时，防止卡死
        if response.status_code == 200:
            return response.json() # 返回解析后的字典数据
        else:
            return None
    except Exception as e:
        print(f"API 调用失败: {e}")
        return None

# --- 逻辑分流示例 ---
def handle_user_query(user_input):
    api_data = None
    source_name = ""

    # 1. 简单的关键词触发分流
    if "配方" in user_input or "怎么做" in user_input:
        # 假设从用户输入里提取到了"肉丸"（可以用正则或简单字符串匹配）
        food_name = "肉丸" 
        api_data = fetch_game_data("recipe", food_name)
        source_name = "游戏配方数据库"

    elif "属性" in user_input or "是什么" in user_input:
        item_name = "长矛"
        api_data = fetch_game_data("items", item_name)
        source_name = "物品图鉴"

    # 2. 将 API 数据融合进 Prompt
    if api_data:
        # 将 API 返回的 JSON 转化为一句话，喂给大模型
        extra_info = f"【来自{source_name}的权威数据】：{api_data}"
    else:
        extra_info = "未找到相关的 API 权威数据。"

    return extra_info

import os
import json
import sqlite3
from datetime import datetime
from typing import List, Optional, Dict
from flask import Flask, request, jsonify
from flask_cors import CORS
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import requests
from functools import lru_cache

# ========== 环境配置 ==========
ZHIPU_API_KEY = "1d0495665bb14223a62ce4d16a8ab85f.WDwBuhSayklsxkkw"
# WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "demo")  #自定义api

# ========== 系统提示词（核心角色设定） ==========
SYSTEM_PROMPT = """你是一位资深的《饥荒 Don't Starve》游戏攻略专家，具有以下特点：

【身份定位】
- 多年饥荒老玩家，对游戏机制透彻理解
- 精通所有人物档案、生物特性、食物属性、建筑优先级
- 熟悉四季变化规律、世界事件、特殊模式玩法

【回答原则】
1. 【严格遵循知识库】：优先基于提供的参考知识(【参考知识】部分)回答，这些是深度玩家总结
2. 【精确步骤】：给出的策略必须包含具体操作步骤，不能模糊
3. 【安全提示】：涉及生存策略时，务必强调哪些操作容易翻车
4. 【多模式考虑】：回答时区分 Don't Starve 和 Don't Starve Together (DST) 的差异
5. 【季节感知】：根据玩家所在季节给出不同建议

【禁止事项】
- 禁止编造游戏机制（例如：编造某个物品属性、某个生物行为）
- 禁止给出通用建议（例如："保持理智值"没有具体方案）
- 如果问题超出饥荒范围，礼貌拒绝并引导回游戏内容

【说话风格】
- 用亲切但权威的语气，类似"游戏大佬的建议"
- 可适当使用游戏术语和玩家俚语（如"掉san"、"出门翻车"、"打黑科技"等）
- 遇到困难问题时，表现出"让我想想游戏机制..."的思考过程

【必要时使用工具】
- 如果需要查询实时天气来推断游戏内季节变化，调用天气API
- 回答中清楚说明信息来源（出自哪篇攻略）

记住：你的目标是帮助玩家避免翻车，活得更久！🎮
"""

# ========== 1. 数据库初始化 ==========
class ChatDatabase:
    """聊天记录持久化"""
    def __init__(self, db_path: str = "chat_history.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_message TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                retrieved_sources TEXT,
                api_used TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    
    def save_conversation(self, session_id: str, user_msg: str, bot_msg: str, 
                         sources: Optional[List[str]] = None, api_used: Optional[str] = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO conversations (session_id, user_message, bot_response, retrieved_sources, api_used)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, user_msg, bot_msg, json.dumps(sources or []), api_used))
        conn.commit()
        conn.close()
    
    def get_conversation_history(self, session_id: str, limit: int = 20) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT user_message, bot_response, retrieved_sources, api_used, created_at
            FROM conversations
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (session_id, limit))
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            history.append({
                "user": row[0],
                "bot": row[1],
                "sources": json.loads(row[2]) if row[2] else [],
                "api_used": row[3],
                "timestamp": row[4]
            })
        return list(reversed(history))

# ========== 2. 嵌入服务（轻量级） ==========
class EmbeddingService:
    """统一的向量化服务"""
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4/"
        )
    
    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model="embedding-3",
            input=[text]
        )
        return response.data[0].embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model="embedding-3",
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

# ========== 3. 向量数据库与检索 ==========
class KnowledgeBase:
    """知识库管理 - 饥荒垂直领域文档"""
    def __init__(self, persist_dir: str, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="dont_starve_guides",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_document(self, doc_id: str, content: str, metadata: Dict):
        """添加单个文档"""
        embedding = self.embedding_service.embed_query(content)
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata]
        )
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """检索相关文档"""
        query_embedding = self.embedding_service.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        documents = []
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                documents.append({
                    'content': doc,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'similarity': results['distances'][0][i] if 'distances' in results else 0.0
                })
        return documents
    
    def get_doc_count(self) -> int:
        return self.collection.count()

# ========== 4. 第三方 API 集成 ==========
class ThirdPartyAPIs:
    """集成外部 API - 如天气、游戏数据等"""
    
    @staticmethod
    @lru_cache(maxsize=100)
    def get_weather(city: str = "成都") -> Dict:
        """
        调用天气 API（模拟）
        在实际应用中可对接高德地图 API
        """
        # 模拟天气数据
        weather_data = {
            "city": city,
            "temp": 25,
            "weather": "晴",
            "humidity": 60,
            "wind_speed": 5,
            "timestamp": datetime.now().isoformat()
        }
        return weather_data
    
    @staticmethod
    def get_game_event(season: str) -> Dict:
        """
        根据真实季节推断游戏内季节事件
        """
        events_map = {
            "春": {"name": "春天", "tips": "蜘蛛巢增多，注意采集草和木头"},
            "夏": {"name": "夏天", "tips": "过热，需要冰箱或冷却装备，灭火"},
            "秋": {"name": "秋天", "tips": "树会脱叶，准备冬季物资"},
            "冬": {"name": "冬天", "tips": "温度下降，需要火焰、衣服等保温"}
        }
        return events_map.get(season, {"name": "未知", "tips": "咨询游戏日历"})
    
    @staticmethod
    def search_game_mechanic(keyword: str) -> Dict:
        """
        模拟调用游戏百科 API
        实际应用可调用 Fandom Wiki 或自建数据库
        """
        mechanics_db = {
            "理智值": {
                "desc": "玩家心理状态，低于临界值会开始行动异常",
                "恢复方式": ["睡眠", "吃精神类食物", "看科学装置"],
                "下降原因": ["在黑暗中", "看到怪物", "饥饿"]
            },
            "饥饿值": {
                "desc": "玩家能量等级",
                "恢复方式": ["吃食物"],
                "影响": "低于0会开始掉血"
            }
        }
        return mechanics_db.get(keyword, {"error": "未找到相关机制"})

# ========== 5. RAG 聊天机器人 ==========
class DontStarveChatBot:
    """整合 RAG + 多轮对话 + 第三方 API 的聊天机器人"""
    
    def __init__(self, api_key: str, kb_dir: str = None, 
                 db_dir: str = None):
        self.api_key = api_key
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4/"
        )
        
        # 设置绝对路径（基于脚本所在目录）
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if kb_dir is None:
            kb_dir = os.path.join(base_dir, "knowledge_base")
        if db_dir is None:
            db_dir = os.path.join(base_dir, "backend", "chroma_db")
        
        # 初始化组件
        self.embedding_service = EmbeddingService(api_key)
        self.kb = KnowledgeBase(db_dir, self.embedding_service)
        self.db = ChatDatabase()
        self.third_party = ThirdPartyAPIs()
        
        # 加载知识库文档
        self._load_knowledge_base()
    
    def _load_knowledge_base(self):
        """从文件加载知识库文档"""
        # 获取实际的知识库路径
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        kb_path = os.path.join(base_dir, "knowledge_base")
        
        if not os.path.exists(kb_path):
            print(f"⚠️ 知识库目录不存在: {kb_path}")
            return
        
        for filename in os.listdir(kb_path):
            if filename.endswith(('.txt', '.md')):
                filepath = os.path.join(kb_path, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    doc_id = filename.replace('.', '_')
                    self.kb.add_document(
                        doc_id=doc_id,
                        content=content,
                        metadata={'source': filename, 'type': 'guide'}
                    )
        
        print(f"✅ 知识库加载完成，共 {self.kb.get_doc_count()} 条记录")
    
    def _detect_intent(self, user_input: str) -> Dict:
        """
        意图识别与路由
        返回：{'type': 'rag'|'weather'|'mechanic'|'general', 'keywords': [...]}
        """
        keywords = {
            'weather': ['天气', '季节', '温度', '下雨'],
            'mechanic': ['机制', '属性', '怎么算', '伤害', '回复'],
            'building': ['建筑', '建造', '科技', '科学机制'],
            'survival': ['生存', '策略', '前期', '中期', '后期', '度过', '怎么过']
        }
        
        detected_types = []
        for key, kws in keywords.items():
            if any(kw in user_input for kw in kws):
                detected_types.append(key)
        
        return {
            'type': detected_types if detected_types else ['rag'],
            'query': user_input
        }
    
    def _format_rag_context(self, retrieved_docs: List[Dict]) -> tuple:
        """格式化 RAG 检索结果"""
        if not retrieved_docs:
            return "", []
        
        sources = []
        context = "【📚 参考知识库】\n"
        for doc in retrieved_docs:
            source = doc['metadata'].get('source', '未知')
            sources.append(source)
            context += f"\n**来自: {source}**\n"
            context += doc['content'][:500] + "...\n" if len(doc['content']) > 500 else doc['content'] + "\n"
        
        return context, sources
    
    def chat(self, user_input: str, session_id: str) -> Dict:
        """
        核心对话方法
        返回：{'response': str, 'sources': List[str], 'apis_used': List[str]}
        """
        apis_used = []
        rag_context = ""
        sources = []
        
        # 1. 意图识别
        intent = self._detect_intent(user_input)
        
        # 2. RAG 检索（大多数问题都走 RAG）
        if 'rag' in intent['type'] or any(t in ['building', 'survival'] for t in intent['type']):
            retrieved_docs = self.kb.retrieve(user_input, k=3)
            rag_context, sources = self._format_rag_context(retrieved_docs)
        
        # 3. 第三方 API 调用（按需）
        extra_context = ""
        if 'weather' in intent['type']:
            weather = self.third_party.get_weather()
            apis_used.append("weather_api")
            extra_context += f"\n\n【⛅ 实时天气】\n城市: {weather['city']}, 温度: {weather['temp']}°C, 天气: {weather['weather']}\n"
        
        if 'mechanic' in intent['type']:
            # 提取关键词，如"理智值"、"饥饿值"等
            keywords = ['理智值', '饥饿值', '血量', '温度']
            for kw in keywords:
                if kw in user_input:
                    mechanic = self.third_party.search_game_mechanic(kw)
                    apis_used.append("game_mechanic_db")
                    extra_context += f"\n\n【⚙️ 游戏机制】\n{json.dumps(mechanic, ensure_ascii=False, indent=2)}\n"
        
        # 4. 构建完整提示词
        full_prompt = f"{rag_context}\n{extra_context}\n\n【用户问题】\n{user_input}"
        
        # 5. 调用大模型
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_prompt}
            ]
            
            response = self.client.chat.completions.create(
                model="glm-5",
                messages=messages,
                temperature=0.7,
                max_tokens=1500
            )
            
            bot_response = response.choices[0].message.content
            
        except Exception as e:
            bot_response = f"❌ 调用大模型失败: {str(e)}"
        
        # 6. 保存对话记录
        self.db.save_conversation(
            session_id=session_id,
            user_msg=user_input,
            bot_msg=bot_response,
            sources=sources,
            api_used=','.join(apis_used) if apis_used else None
        )
        
        return {
            'response': bot_response,
            'sources': sources,
            'apis_used': apis_used,
            'intent': intent['type']
        }
    
    def get_history(self, session_id: str) -> List[Dict]:
        """获取会话历史"""
        return self.db.get_conversation_history(session_id)

# ========== 6. Flask 应用 ==========
app = Flask(__name__)
CORS(app)

# 初始化机器人
bot = DontStarveChatBot(api_key=ZHIPU_API_KEY)

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """聊天端点"""
    data = request.json
    user_input = data.get('message', '')
    session_id = data.get('session_id', 'default')
    
    if not user_input.strip():
        return jsonify({'error': '消息不能为空'}), 400
    
    result = bot.chat(user_input, session_id)
    return jsonify(result)

@app.route('/api/history/<session_id>', methods=['GET'])
def get_history_endpoint(session_id):
    """获取会话历史"""
    history = bot.get_history(session_id)
    return jsonify({'history': history})

@app.route('/api/knowledge-base-status', methods=['GET'])
def kb_status():
    """知识库状态"""
    return jsonify({
        'doc_count': bot.kb.get_doc_count(),
        'status': '✅ 就绪' if bot.kb.get_doc_count() > 0 else '⚠️ 文档缺失'
    })

@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print("🎮 饥荒游戏攻略助手 - 后端启动")
    print(f"📚 知识库状态: {bot.kb.get_doc_count()} 条记录")
    app.run(debug=True, port=5000)