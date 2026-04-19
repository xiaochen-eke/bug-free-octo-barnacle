"""
《饥荒游戏数据 API 服务器》
这是一个独立的 API 服务，提供游戏数据、物品信息、生物数据等
可以与主聊天机器人分离部署或在同一服务器运行

用途：
1. 作为独立的微服务运行在 localhost:5001
2. 提供游戏数据查询接口
3. 支持物品、生物、建筑、食物等多种数据类型
"""


from flask import Flask
from flask_cors import CORS  # 1. 导入插件

app = Flask(__name__)
CORS(app)  # 2. 允许所有来源访问这个 API


import requests
from typing import Dict, List, Optional, Any
from functools import lru_cache
import json


from flask import Flask, request, jsonify
from datetime import datetime
import json

# 创建 Flask 应用
api_app = Flask(__name__)

# ========== 游戏数据库 ==========
# 这些数据模拟了游戏中的真实数据

GAME_DATABASE = {
    # ===== 物品数据库 =====
    "items": {
        "木头": {
            "id": "wood",
            "name": "木头",
            "description": "从树上砍下来的木头，是基础资源",
            "rarity": "common",
            "category": "resource",
            "uses": ["建筑", "燃料", "工具制作"],
            "durability": None,
            "stackable": True,
            "stack_size": 40,
            "weight": 0.5
        },
        "石头": {
            "id": "stone",
            "name": "石头",
            "description": "坚硬的石头，用于建筑和工具",
            "rarity": "common",
            "category": "resource",
            "uses": ["建筑", "工具制作"],
            "durability": None,
            "stackable": True,
            "stack_size": 40,
            "weight": 1.0
        },
        "燧石": {
            "id": "flint",
            "name": "燧石",
            "description": "尖锐的燧石，用于制作工具和生火",
            "rarity": "common",
            "category": "resource",
            "uses": ["工具制作", "火焰", "武器"],
            "durability": None,
            "stackable": True,
            "stack_size": 40,
            "weight": 0.3
        },
        "草": {
            "id": "grass",
            "name": "草",
            "description": "普通的草，可用于制作绳子和其他物品",
            "rarity": "common",
            "category": "resource",
            "uses": ["绳子制作", "食物"],
            "durability": None,
            "stackable": True,
            "stack_size": 40,
            "weight": 0.1
        },
        "蜘蛛丝": {
            "id": "spider_silk",
            "name": "蜘蛛丝",
            "description": "坚韧的蜘蛛丝，用于制作装备和武器",
            "rarity": "uncommon",
            "category": "resource",
            "uses": ["衣服制作", "武器", "工具"],
            "durability": None,
            "stackable": True,
            "stack_size": 40,
            "weight": 0.2
        },
        "蜂蜜": {
            "id": "honey",
            "name": "蜂蜜",
            "description": "甜美的蜂蜜，是高价值食物",
            "rarity": "uncommon",
            "category": "food",
            "uses": ["食物", "蛋糕制作", "治疗"],
            "durability": None,
            "stackable": True,
            "stack_size": 40,
            "weight": 0.3,
            "nutrition": {
                "hunger": 40,
                "health": 0,
                "sanity": 5
            }
        },
        "营火": {
            "id": "campfire",
            "name": "营火",
            "description": "提供光照和温暖的生活必需品",
            "rarity": "essential",
            "category": "building",
            "uses": ["光照", "温暖", "烹饪"],
            "durability": 200,
            "stackable": False,
            "requires": {"wood": 3, "flint": 1},
            "fuel_efficiency": 1.0
        },
        "矛": {
            "id": "spear",
            "name": "矛",
            "description": "简单的武器，用于对抗敌人",
            "rarity": "common",
            "category": "weapon",
            "uses": ["战斗"],
            "durability": 100,
            "stackable": False,
            "damage": 34,
            "requires": {"wood": 3, "flint": 2}
        },
        "蜘蛛矛": {
            "id": "spider_spear",
            "name": "蜘蛛矛",
            "description": "用蜘蛛丝加强的矛，伤害更高",
            "rarity": "uncommon",
            "category": "weapon",
            "uses": ["战斗"],
            "durability": 150,
            "stackable": False,
            "damage": 68,
            "requires": {"wood": 3, "spider_silk": 2, "flint": 1}
        }
    },

    # ===== 生物数据库 =====
    "creatures": {
        "蜘蛛": {
            "id": "spider",
            "name": "蜘蛛",
            "description": "常见的敌人，会从蜘蛛巢窝中出现",
            "health": 10,
            "damage": 8,
            "speed": 6,
            "rarity": "common",
            "danger_level": 2,
            "drops": [{"item": "spider_silk", "chance": 0.5}],
            "behavior": "会主动攻击玩家",
            "location": "地表各处",
            "spawn_time": "全时段"
        },
        "蜘蛛女王": {
            "id": "spider_queen",
            "name": "蜘蛛女王",
            "description": "强大的蜘蛛领主，极其危险",
            "health": 100,
            "damage": 40,
            "speed": 8,
            "rarity": "rare",
            "danger_level": 5,
            "drops": [{"item": "spider_gland", "chance": 1.0}],
            "behavior": "极具攻击性，会召唤小蜘蛛",
            "location": "蜘蛛洞穴",
            "spawn_time": "昼夜交替"
        },
        "兔子": {
            "id": "rabbit",
            "name": "兔子",
            "description": "温和的动物，可以通过陷阱捕捉",
            "health": 3,
            "damage": 0,
            "speed": 12,
            "rarity": "common",
            "danger_level": 0,
            "drops": [{"item": "meat", "chance": 0.8}],
            "behavior": "会逃跑，不主动攻击",
            "location": "草地区域",
            "spawn_time": "白天"
        },
        "蜜蜂": {
            "id": "bee",
            "name": "蜜蜂",
            "description": "会采蜜的昆虫，有时会攻击",
            "health": 1,
            "damage": 5,
            "speed": 10,
            "rarity": "common",
            "danger_level": 1,
            "drops": [{"item": "honey", "chance": 0.3}],
            "behavior": "防守蜂巢，但可以躲避",
            "location": "蜂巢附近",
            "spawn_time": "白天"
        },
        "眼球怪": {
            "id": "eye_of_cthulhu",
            "name": "眼球怪兽",
            "description": "来自黑暗的恐怖生物，极其危险",
            "health": 200,
            "damage": 50,
            "speed": 5,
            "rarity": "rare",
            "danger_level": 5,
            "drops": [{"item": "nightmare_fuel", "chance": 1.0}],
            "behavior": "会跟踪并攻击玩家，极具威胁",
            "location": "地下洞穴",
            "spawn_time": "夜晚"
        }
    },

    # ===== 建筑数据库 =====
    "buildings": {
        "营火": {
            "id": "campfire",
            "name": "营火",
            "description": "提供光照和温暖，可烹饪食物",
            "category": "essential",
            "tier": 1,
            "cost": {"wood": 3, "flint": 1},
            "crafting_time": 5,
            "uses": ["光照", "温暖", "烹饪"],
            "stats": {
                "light_range": 15,
                "warmth": 60,
                "fuel_consumption": 1.0,
                "cooking_efficiency": 1.0
            },
            "priority": "CRITICAL"
        },
        "烹饪锅": {
            "id": "cooking_pot",
            "name": "烹饪锅",
            "description": "可以制作高级食物",
            "category": "cooking",
            "tier": 2,
            "cost": {"gold": 3, "stone": 3, "wood": 6},
            "crafting_time": 10,
            "uses": ["烹饪", "食物合成"],
            "stats": {
                "recipes": 15,
                "efficiency": 1.2,
                "capacity": 4
            },
            "priority": "HIGH"
        },
        "冰箱": {
            "id": "fridge",
            "name": "冰箱",
            "description": "保存食物，防止腐烂",
            "category": "storage",
            "tier": 3,
            "cost": {"gold": 15, "wood": 8, "stone": 10},
            "crafting_time": 20,
            "uses": ["食物保存"],
            "stats": {
                "storage_capacity": 20,
                "rot_slowdown": 0.25,
                "slots": 6
            },
            "priority": "MEDIUM"
        },
        "科学机制": {
            "id": "science_machine",
            "name": "科学机制",
            "description": "进行科学研究，解锁新科技",
            "category": "research",
            "tier": 2,
            "cost": {"gold": 15, "stone": 10, "wood": 8},
            "crafting_time": 15,
            "uses": ["科技研究"],
            "stats": {
                "research_speed": 1.0,
                "science_points": 1
            },
            "priority": "HIGH"
        },
        "农场": {
            "id": "farm",
            "name": "农场",
            "description": "种植蔬菜和水果",
            "category": "farming",
            "tier": 2,
            "cost": {"grass": 6, "seed": 3, "wood": 4},
            "crafting_time": 8,
            "uses": ["种植", "食物生产"],
            "stats": {
                "plots": 1,
                "growth_time": 2,
                "yield": 4
            },
            "priority": "HIGH"
        }
    },

    # ===== 食物数据库 =====
    "foods": {
        "浆果": {
            "id": "berry",
            "name": "浆果",
            "description": "从灌木采集的浆果，可直接食用",
            "rarity": "common",
            "type": "raw",
            "nutrition": {
                "hunger": 9.4,
                "health": 0,
                "sanity": -1
            },
            "spoil_time": 6,
            "stackable": True
        },
        "烤肉": {
            "id": "cooked_meat",
            "name": "烤肉",
            "description": "烹饪过的肉类，营养丰富",
            "rarity": "common",
            "type": "cooked",
            "nutrition": {
                "hunger": 18.8,
                "health": 0,
                "sanity": 0
            },
            "spoil_time": 15,
            "stackable": True,
            "requires": "meat"
        },
        "蛋糕": {
            "id": "cake",
            "name": "蛋糕",
            "description": "美味的蛋糕，高价值食物和理智恢复",
            "rarity": "uncommon",
            "type": "cooked",
            "nutrition": {
                "hunger": 75,
                "health": 20,
                "sanity": 5
            },
            "spoil_time": 20,
            "stackable": True,
            "recipe": {
                "egg": 2,
                "honey": 1,
                "butter": 1
            }
        },
        "蜂蜜汉堡": {
            "id": "honey_burger",
            "name": "蜂蜜汉堡",
            "description": "含有蜂蜜的汉堡，营养均衡",
            "rarity": "uncommon",
            "type": "cooked",
            "nutrition": {
                "hunger": 75,
                "health": 12,
                "sanity": 0
            },
            "spoil_time": 15,
            "stackable": True,
            "recipe": {
                "cooked_meat": 1,
                "honey": 1,
                "vegetable": 1,
                "bread": 1
            }
        },
        "肉丸": {
            "id": "meatballs",
            "name": "肉丸",
            "description": "用肉类和蔬菜制作的肉丸",
            "rarity": "uncommon",
            "type": "cooked",
            "nutrition": {
                "hunger": 62.5,
                "health": 12,
                "sanity": 0
            },
            "spoil_time": 15,
            "stackable": True,
            "recipe": {
                "meat": 2,
                "vegetable": 1,
                "egg": 1
            }
        }
    },

    # ===== 季节数据库 =====
    "seasons": {
        "春": {
            "id": "spring",
            "name": "春天",
            "duration": 15,
            "temperature": 20,
            "description": "温暖的季节，蜘蛛开始活跃",
            "features": [
                "蜘蛛巢窝增多",
                "树木开始生长",
                "植物生长加速"
            ],
            "challenges": [
                "蜘蛛攻击",
                "资源竞争"
            ],
            "resources": {
                "wood": "充足",
                "grass": "充足",
                "spider_silk": "增加"
            }
        },
        "夏": {
            "id": "summer",
            "name": "夏天",
            "duration": 20,
            "temperature": 35,
            "description": "炎热的季节，容易着火",
            "features": [
                "温度升高",
                "容易着火",
                "食物加速腐烂"
            ],
            "challenges": [
                "过热死亡",
                "山林火灾",
                "食物腐烂"
            ],
            "resources": {
                "wood": "减少",
                "food_spoil": "加速"
            }
        },
        "秋": {
            "id": "autumn",
            "name": "秋天",
            "duration": 20,
            "temperature": 15,
            "description": "树木脱叶，准备冬季",
            "features": [
                "树木脱叶",
                "资源丰富",
                "温度下降"
            ],
            "challenges": [
                "树木掉叶",
                "食物准备"
            ],
            "resources": {
                "wood": "极多",
                "vegetable": "充足"
            }
        },
        "冬": {
            "id": "winter",
            "name": "冬天",
            "duration": 15,
            "temperature": -15,
            "description": "严寒季节，最具挑战",
            "features": [
                "温度极低",
                "食物匮乏",
                "敌人减少"
            ],
            "challenges": [
                "冻死风险",
                "食物短缺",
                "农场不生长"
            ],
            "resources": {
                "food": "极缺",
                "enemies": "减少"
            }
        }
    }
}

# ========== API 路由 ==========

@api_app.route('/api/game-data/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'service': 'Dont Starve Game Data API',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    }), 200


@api_app.route('/api/game-data/items', methods=['GET'])
def get_all_items():
    """获取所有物品列表"""
    return jsonify({
        'status': 'success',
        'count': len(GAME_DATABASE['items']),
        'items': list(GAME_DATABASE['items'].keys())
    }), 200


@api_app.route('/api/game-data/items/<item_name>', methods=['GET'])
def get_item(item_name):
    """获取特定物品详情"""
    if item_name in GAME_DATABASE['items']:
        item_data = GAME_DATABASE['items'][item_name]
        return jsonify({
            'status': 'success',
            'item_name': item_name,
            'data': item_data
        }), 200
    else:
        return jsonify({
            'status': 'error',
            'message': f'物品 "{item_name}" 未找到',
            'available_items': list(GAME_DATABASE['items'].keys())
        }), 404


@api_app.route('/api/game-data/creatures', methods=['GET'])
def get_all_creatures():
    """获取所有生物列表"""
    return jsonify({
        'status': 'success',
        'count': len(GAME_DATABASE['creatures']),
        'creatures': list(GAME_DATABASE['creatures'].keys())
    }), 200


@api_app.route('/api/game-data/creatures/<creature_name>', methods=['GET'])
def get_creature(creature_name):
    """获取特定生物详情"""
    if creature_name in GAME_DATABASE['creatures']:
        creature_data = GAME_DATABASE['creatures'][creature_name]
        return jsonify({
            'status': 'success',
            'creature_name': creature_name,
            'data': creature_data
        }), 200
    else:
        return jsonify({
            'status': 'error',
            'message': f'生物 "{creature_name}" 未找到',
            'available_creatures': list(GAME_DATABASE['creatures'].keys())
        }), 404


@api_app.route('/api/game-data/buildings', methods=['GET'])
def get_all_buildings():
    """获取所有建筑列表"""
    return jsonify({
        'status': 'success',
        'count': len(GAME_DATABASE['buildings']),
        'buildings': list(GAME_DATABASE['buildings'].keys())
    }), 200


@api_app.route('/api/game-data/buildings/<building_name>', methods=['GET'])
def get_building(building_name):
    """获取特定建筑详情"""
    if building_name in GAME_DATABASE['buildings']:
        building_data = GAME_DATABASE['buildings'][building_name]
        return jsonify({
            'status': 'success',
            'building_name': building_name,
            'data': building_data
        }), 200
    else:
        return jsonify({
            'status': 'error',
            'message': f'建筑 "{building_name}" 未找到',
            'available_buildings': list(GAME_DATABASE['buildings'].keys())
        }), 404


@api_app.route('/api/game-data/foods', methods=['GET'])
def get_all_foods():
    """获取所有食物列表"""
    return jsonify({
        'status': 'success',
        'count': len(GAME_DATABASE['foods']),
        'foods': list(GAME_DATABASE['foods'].keys())
    }), 200


@api_app.route('/api/game-data/foods/<food_name>', methods=['GET'])
def get_food(food_name):
    """获取特定食物详情"""
    if food_name in GAME_DATABASE['foods']:
        food_data = GAME_DATABASE['foods'][food_name]
        return jsonify({
            'status': 'success',
            'food_name': food_name,
            'data': food_data
        }), 200
    else:
        return jsonify({
            'status': 'error',
            'message': f'食物 "{food_name}" 未找到',
            'available_foods': list(GAME_DATABASE['foods'].keys())
        }), 404


@api_app.route('/api/game-data/seasons', methods=['GET'])
def get_all_seasons():
    """获取所有季节列表"""
    return jsonify({
        'status': 'success',
        'count': len(GAME_DATABASE['seasons']),
        'seasons': list(GAME_DATABASE['seasons'].keys())
    }), 200


@api_app.route('/api/game-data/seasons/<season_name>', methods=['GET'])
def get_season(season_name):
    """获取特定季节详情"""
    if season_name in GAME_DATABASE['seasons']:
        season_data = GAME_DATABASE['seasons'][season_name]
        return jsonify({
            'status': 'success',
            'season_name': season_name,
            'data': season_data
        }), 200
    else:
        return jsonify({
            'status': 'error',
            'message': f'季节 "{season_name}" 未找到',
            'available_seasons': list(GAME_DATABASE['seasons'].keys())
        }), 404


@api_app.route('/api/game-data/search', methods=['GET'])
def search_game_data():
    """搜索游戏数据（全文搜索）"""
    query = request.args.get('q', '').lower()
    category = request.args.get('category', 'all')  # items, creatures, buildings, foods, seasons
    
    if not query:
        return jsonify({
            'status': 'error',
            'message': '请提供搜索关键词'
        }), 400
    
    results = {
        'query': query,
        'results': {}
    }
    
    # 搜索物品
    if category in ['all', 'items']:
        item_matches = {
            name: data for name, data in GAME_DATABASE['items'].items()
            if query in name.lower() or query in data.get('description', '').lower()
        }
        if item_matches:
            results['results']['items'] = item_matches
    
    # 搜索生物
    if category in ['all', 'creatures']:
        creature_matches = {
            name: data for name, data in GAME_DATABASE['creatures'].items()
            if query in name.lower() or query in data.get('description', '').lower()
        }
        if creature_matches:
            results['results']['creatures'] = creature_matches
    
    # 搜索建筑
    if category in ['all', 'buildings']:
        building_matches = {
            name: data for name, data in GAME_DATABASE['buildings'].items()
            if query in name.lower() or query in data.get('description', '').lower()
        }
        if building_matches:
            results['results']['buildings'] = building_matches
    
    # 搜索食物
    if category in ['all', 'foods']:
        food_matches = {
            name: data for name, data in GAME_DATABASE['foods'].items()
            if query in name.lower() or query in data.get('description', '').lower()
        }
        if food_matches:
            results['results']['foods'] = food_matches
    
    if results['results']:
        return jsonify({
            'status': 'success',
            **results
        }), 200
    else:
        return jsonify({
            'status': 'success',
            'message': '未找到匹配的结果',
            'query': query,
            'category': category
        }), 200


@api_app.route('/api/game-data/recipe/<food_name>', methods=['GET'])
def get_recipe(food_name):
    """获取食物配方"""
    if food_name in GAME_DATABASE['foods']:
        food_data = GAME_DATABASE['foods'][food_name]
        if 'recipe' in food_data:
            return jsonify({
                'status': 'success',
                'food_name': food_name,
                'recipe': food_data['recipe'],
                'description': f"制作 {food_name} 需要："
            }), 200
        else:
            return jsonify({
                'status': 'success',
                'message': f'{food_name} 是原始食物，不需要配方',
                'food_name': food_name,
                'type': food_data.get('type', 'unknown')
            }), 200
    else:
        return jsonify({
            'status': 'error',
            'message': f'食物 "{food_name}" 未找到'
        }), 404


@api_app.route('/api/game-data/crafting/<building_name>', methods=['GET'])
def get_crafting_cost(building_name):
    """获取建筑的制作成本"""
    if building_name in GAME_DATABASE['buildings']:
        building_data = GAME_DATABASE['buildings'][building_name]
        return jsonify({
            'status': 'success',
            'building_name': building_name,
            'cost': building_data.get('cost', {}),
            'crafting_time': building_data.get('crafting_time', 0),
            'description': building_data.get('description', '')
        }), 200
    else:
        return jsonify({
            'status': 'error',
            'message': f'建筑 "{building_name}" 未找到'
        }), 404


@api_app.route('/api/game-data/tips', methods=['GET'])
def get_game_tips():
    """获取游戏提示"""
    tips = [
        "营火是第一天的最高优先级 - 必须在日落前 30 分钟完成！",
        "理智值低于 20% 时，玩家会陷入疯狂状态并开始自杀",
        "蛋糕是最好的理智值恢复食物，可恢复 +5 理智值",
        "蜘蛛女王极其危险，不要在没有准备的情况下靠近蜘蛛洞穴",
        "冬天是最具挑战的季节，需要提前准备足够的食物和燃料",
        "农场在冬天不会生长，必须依靠储备食物或陷阱狩猎",
        "睡眠是恢复理智值最有效的方法 (+1.7/秒)",
        "在黑暗中停留会不断掉理智值，每秒 -1",
        "烹饪所有肉类是必须的，生肉会导致理智值下降",
        "蜜蜂松饼是最好的食物和理智值恢复组合"
    ]
    
    import random
    selected_tips = random.sample(tips, min(3, len(tips)))
    
    return jsonify({
        'status': 'success',
        'tips': selected_tips,
        'total_available_tips': len(tips)
    }), 200


@api_app.route('/api/game-data/database-stats', methods=['GET'])
def get_database_stats():
    """获取游戏数据库统计"""
    return jsonify({
        'status': 'success',
        'statistics': {
            'items': len(GAME_DATABASE['items']),
            'creatures': len(GAME_DATABASE['creatures']),
            'buildings': len(GAME_DATABASE['buildings']),
            'foods': len(GAME_DATABASE['foods']),
            'seasons': len(GAME_DATABASE['seasons']),
            'total_entries': sum(len(v) for v in GAME_DATABASE.values())
        },
        'api_version': '1.0.0',
        'game': 'Dont Starve'
    }), 200


# ========== 错误处理 ==========

@api_app.errorhandler(404)
def not_found(error):
    """处理 404 错误"""
    return jsonify({
        'status': 'error',
        'message': '端点未找到',
        'available_endpoints': [
            '/api/game-data/health',
            '/api/game-data/items',
            '/api/game-data/creatures',
            '/api/game-data/buildings',
            '/api/game-data/foods',
            '/api/game-data/seasons',
            '/api/game-data/search?q=keyword',
            '/api/game-data/recipe/<food_name>',
            '/api/game-data/crafting/<building_name>',
            '/api/game-data/tips',
            '/api/game-data/database-stats'
        ]
    }), 404


@api_app.errorhandler(500)
def server_error(error):
    """处理 500 错误"""
    return jsonify({
        'status': 'error',
        'message': '服务器错误',
        'error': str(error)
    }), 500


if __name__ == '__main__':
    print("🎮 饥荒游戏数据 API 服务器启动")
    print("📍 运行地址: http://localhost:5001")
    print("")
    print("可用的 API 端点：")
    print("  GET  /api/game-data/health              - 健康检查")
    print("  GET  /api/game-data/items               - 获取所有物品")
    print("  GET  /api/game-data/items/<name>        - 获取物品详情")
    print("  GET  /api/game-data/creatures           - 获取所有生物")
    print("  GET  /api/game-data/creatures/<name>    - 获取生物详情")
    print("  GET  /api/game-data/buildings           - 获取所有建筑")
    print("  GET  /api/game-data/buildings/<name>    - 获取建筑详情")
    print("  GET  /api/game-data/foods               - 获取所有食物")
    print("  GET  /api/game-data/foods/<name>        - 获取食物详情")
    print("  GET  /api/game-data/seasons             - 获取所有季节")
    print("  GET  /api/game-data/seasons/<name>      - 获取季节详情")
    print("  GET  /api/game-data/search?q=keyword    - 全文搜索")
    print("  GET  /api/game-data/recipe/<food>       - 获取食物配方")
    print("  GET  /api/game-data/crafting/<building> - 获取建筑制作成本")
    print("  GET  /api/game-data/tips                - 获取游戏提示")
    print("  GET  /api/game-data/database-stats      - 获取数据库统计")
    print("")
    print("按 Ctrl+C 停止服务器")
    print("")
    
    api_app.run(debug=True, port=5001)