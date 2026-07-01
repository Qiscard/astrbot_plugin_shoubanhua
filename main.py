import re
import time
import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple, Any, Dict

import aiofiles

from astrbot import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.core import AstrBotConfig
from astrbot.core.message.components import Image, Plain, Node, Nodes, At, Reply, Video
from astrbot.core.platform.astr_message_event import AstrMessageEvent

# 导入模块
from .data_manager import DataManager
from .image_manager import ImageManager
from .api_manager import ApiManager
from .context_manager import ContextManager, LLMTaskAnalyzer
from .utils import norm_id, extract_image_urls_from_text

# 换衣/穿搭意图关键词
_CLOTHING_KEYWORDS = [
    "换衣", "换装", "穿上", "穿着", "换一套", "换一身", "换个造型",
    "衣服", "裙子", "制服", "校服", "女仆", "和服", "旗袍",
    "JK", "jk", "洛丽塔", "lolita", "婚纱", "西装", "礼服",
    "泳装", "比基尼", "丝袜", "水手服", "死库水",
    "同款", "这件", "那件", "这套", "那套",
    "cosplay", "cos", "穿这个", "穿那个",
    "换件", "试试这套", "试试这件", "穿同款",
]


@register(
    "astrbot_plugin_shoubanhua",
    "shskjw",
    "支持第三方OpenAI绘图格式的文生图/图生图插件，支持多源API配置和LLM智能判断",
    "3.0.1",
    "https://github.com/Qiscard/astrbot_plugin_shoubanhua",
)
class FigurineProPlugin(Star):
    _DEPRECATED_CONFIG_KEYS = [
        "enable_power_model",
        "power_model_keyword",
        "power_model_id",
        "power_model_tip_enabled",
        "power_model_extra_cost",
        "power_mode_fallback_to_group",
        "power_generic_api_url",
        "power_generic_api_keys",
        "power_gemini_api_url",
        "power_gemini_api_keys",
        "gemini_api_url",
        "gemini_api_keys",
        "generic_api_url",
        "generic_api_keys",
        "api_mode",
        "enable_rebellious_mode",
        "rebellious_probability",
        "enable_obedient_mode",
        "obedient_whitelist",
    ]
    _DYNAMIC_CONFIG_KEYS = [
        "prompt_list",
        "generic_sources",
        "generic_active_source",
    ]
    _DYNAMIC_CONFIG_META_KEY = "__dynamic_overrides__"
    _DYNAMIC_CONFIG_VERSION_KEY = "__dynamic_config_version__"
    _DYNAMIC_CONFIG_UPDATED_AT_KEY = "__dynamic_updated_at__"
    _DYNAMIC_CONFIG_VERSION = 3
    _LEGACY_DYNAMIC_RESTORE_KEYS = {
        "prompt_list",
        "generic_api_keys",
        "gemini_api_keys",
    }
    _COMMAND_PRIORITY_DYNAMIC_KEYS = {
        "prompt_list",
        "generic_active_source",
    }
    _PANEL_PRIORITY_DYNAMIC_KEYS = {
        "generic_api_url",
        "gemini_api_url",
    }

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config

        self.data_mgr = DataManager(StarTools.get_data_dir(), config)
        self.img_mgr = ImageManager(config)
        self.api_mgr = ApiManager(config)

        # 上下文管理器
        self.ctx_mgr = ContextManager(
            max_messages=config.get("context_max_messages", 50),
            max_sessions=config.get("context_max_sessions", 100)
        )

        # LLM 智能判断配置
        self._llm_auto_detect = config.get("enable_llm_auto_detect", False)
        self._context_rounds = config.get("context_rounds", 20)
        # 提高默认置信度阈值，减少误触发
        self._auto_detect_confidence = config.get("auto_detect_confidence", 0.8)

        # 日常人设配置
        self._persona_mode = config.get("enable_persona_mode", False)
        self._persona_scene_map = {}  # 场景关键词 -> 提示词
        self._load_persona_scenes()

        # 图片生成冷却时间（只针对图片生成，不影响正常聊天）
        self._image_cooldown_seconds = config.get("llm_cooldown_seconds", 60)
        self._user_last_image_gen: Dict[str, datetime] = {}  # 用户ID -> 上次图片生成时间

        # 消息去重缓存（防止多平台重复处理同一消息）
        self._processed_msg_ids: Dict[str, float] = {}  # msg_id -> timestamp
        self._msg_dedup_ttl = 60  # 去重缓存保留时间（秒）
        self._msg_dedup_max_size = 1000  # 最大缓存数量

        # 后台生成任务状态跟踪（用于 PDF 等待“全部生成完成”）
        self._pending_generation_tasks: Dict[str, int] = {}  # session_id -> pending count
        self._pending_generation_lock = asyncio.Lock()

        # 会话级成功生成记录（用于限制 PDF 工具只能在图片成功生成后调用）
        self._session_generated_success: Dict[str, int] = {}  # session_id -> success count
        self._session_generated_success_lock = asyncio.Lock()

        # 会话级最近成功生成图片缓存（用于 PDF 打包时直接拿到刚生成完成的图片）
        self._session_generated_images: Dict[str, List[bytes]] = {}  # session_id -> recent image bytes
        self._session_generated_images_lock = asyncio.Lock()
        self._session_generated_images_max = max(1, int(config.get("pdf_session_image_cache", 20)))

        # 会话级最近图片上下文：同时记录用户发送的图片和 Bot 生成的图片，供“修改上面那张”这类追问使用
        self._session_recent_image_context: Dict[str, List[Dict[str, Any]]] = {}
        self._session_recent_image_context_lock = asyncio.Lock()
        self._session_recent_image_context_max = max(
            1,
            int(config.get("context_image_cache", self._session_generated_images_max) or self._session_generated_images_max)
        )

        # PDF 暂存模式：当 pack_images_to_pdf 被调用时，通知后台生成任务不要发送单张图片，
        # 而是将图片写入暂存文件夹，等全部生成完后统一打包 PDF 发送。
        self._pdf_staging_sessions: Dict[str, bool] = {}  # session_id -> True 表示处于暂存模式
        self._pdf_staging_images: Dict[str, Dict[int, bytes]] = {}  # session_id -> {序号: 图片字节}，按序号排序保证顺序
        self._pdf_staging_counter: Dict[str, int] = {}  # session_id -> 下一个可用序号（自增）
        self._pdf_staging_lock = asyncio.Lock()

        # 会话级任务进度表（用于催促时优先返回当前任务状态，避免重复开新任务）
        self._session_task_status: Dict[str, Dict[str, Any]] = {}
        self._session_task_status_lock = asyncio.Lock()

    def _purge_deprecated_config_keys(self, config_obj=None) -> int:
        """清理已经废弃的旧配置字段，避免面板继续显示历史残留项。"""
        target = config_obj if config_obj is not None else self.conf
        removed = 0

        for key in self._DEPRECATED_CONFIG_KEYS:
            try:
                if key in target:
                    del target[key]
                    removed += 1
            except Exception:
                try:
                    value = target.get(key, None)
                    if value is not None:
                        target[key] = None
                        removed += 1
                except Exception:
                    pass

        return removed

    def _get_schema_defaults(self) -> Dict[str, Any]:
        """读取配置 schema 默认值，用于判断 dynamic_config 是否会覆盖用户面板保存的新配置。"""
        try:
            import os
            schema_path = os.path.join(os.path.dirname(__file__), "_conf_schema.json")
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)

            if not isinstance(schema, dict):
                return {}

            defaults = {}
            for key, item in schema.items():
                if isinstance(item, dict) and "default" in item:
                    defaults[key] = item.get("default")
            return defaults
        except Exception as e:
            logger.warning(f"FigurinePro: 读取配置默认值失败，将使用保守恢复策略: {e}")
            return {}

    @staticmethod
    def _is_empty_config_value(value) -> bool:
        return value is None or value == "" or value == [] or value == {}

    def _should_restore_dynamic_value(
            self,
            key: str,
            dynamic_value,
            has_override_meta: bool,
            schema_defaults: Dict[str, Any],
            dynamic_backup_is_newer: Optional[bool] = None,
    ) -> Tuple[bool, str]:
        """决定是否用 dynamic_config 覆盖当前配置。

        原则：
        1. 新版带元数据的命令改动仍然可以在重启后恢复；
        2. 如果 AstrBot 当前配置已经是非默认值，说明面板保存过，优先保留面板值；
        3. 旧版无元数据备份只恢复命令可能改动过的字段，避免老地址反复盖回去。
        """
        if key not in self._DYNAMIC_CONFIG_KEYS:
            return False, "unknown-key"

        current_exists = key in self.conf
        current_value = self.conf.get(key) if current_exists else None
        if current_exists and current_value == dynamic_value:
            return False, "same-value"

        has_default = key in schema_defaults
        default_value = schema_defaults.get(key)
        current_is_default = has_default and current_value == default_value

        if has_override_meta:
            if current_exists and not current_is_default:
                if key in self._PANEL_PRIORITY_DYNAMIC_KEYS:
                    return False, "keep-panel-value"
                if dynamic_backup_is_newer is True:
                    return True, "metadata-newer"
                if dynamic_backup_is_newer is None and key in self._COMMAND_PRIORITY_DYNAMIC_KEYS:
                    return True, "metadata-command-priority"
                return False, "keep-panel-value"
            return True, "metadata-override"

        if key not in self._LEGACY_DYNAMIC_RESTORE_KEYS:
            return False, "legacy-url-or-unsupported"

        if self._is_empty_config_value(dynamic_value):
            return False, "legacy-empty"

        if has_default and dynamic_value == default_value:
            return False, "legacy-default"

        if current_exists and not current_is_default:
            return False, "keep-panel-value"

        return True, "legacy-compatible"

    def _build_dynamic_config_backup(self, override_keys) -> Dict[str, Any]:
        dynamic_keys = list(self._DYNAMIC_CONFIG_KEYS)
        clean_overrides = sorted({k for k in override_keys if k in dynamic_keys})

        save_data = {}
        for k in dynamic_keys:
            if k in self.conf:
                save_data[k] = self.conf[k]

        save_data[self._DYNAMIC_CONFIG_META_KEY] = clean_overrides
        save_data[self._DYNAMIC_CONFIG_VERSION_KEY] = self._DYNAMIC_CONFIG_VERSION
        save_data[self._DYNAMIC_CONFIG_UPDATED_AT_KEY] = datetime.now().isoformat(timespec="seconds")
        return save_data

    def _write_dynamic_config_backup(self, config_path: str, override_keys) -> None:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self._build_dynamic_config_backup(override_keys), f, ensure_ascii=False, indent=2)

    @staticmethod
    def _coerce_existing_file_path(value) -> Optional[str]:
        try:
            import os
            if not value:
                return None

            path = os.fspath(value)
            if os.path.isfile(path):
                return path
        except Exception:
            return None
        return None

    def _get_config_source_mtime(self) -> Optional[float]:
        """尽量定位 AstrBot 原始配置文件，用 mtime 判断面板保存和命令备份谁更新。"""
        import os

        candidates = []
        attr_names = (
            "path",
            "file_path",
            "filepath",
            "config_path",
            "_path",
            "_file_path",
            "_filepath",
            "_config_path",
        )
        key_names = (
            "__config_path__",
            "_config_path",
            "config_path",
            "path",
        )

        for attr in attr_names:
            candidates.append(getattr(self.conf, attr, None))

        for key in key_names:
            try:
                candidates.append(self.conf.get(key))
            except Exception:
                pass

        try:
            for value in vars(self.conf).values():
                if isinstance(value, (str, os.PathLike)):
                    candidates.append(value)
        except Exception:
            pass

        mtimes = []
        for candidate in candidates:
            path = self._coerce_existing_file_path(candidate)
            if path:
                try:
                    mtimes.append(os.path.getmtime(path))
                except Exception:
                    pass

        return max(mtimes) if mtimes else None

    def _is_dynamic_backup_newer_than_config(self, config_path: str) -> Optional[bool]:
        import os

        config_mtime = self._get_config_source_mtime()
        if config_mtime is None:
            return None

        try:
            dynamic_mtime = os.path.getmtime(config_path)
        except Exception:
            return None

        return dynamic_mtime >= config_mtime

    def _is_message_processed(self, msg_id: str) -> bool:
        """
        检查消息是否已被处理过（用于去重）

        Args:
            msg_id: 消息ID

        Returns:
            是否已处理过
        """
        import time
        current_time = time.time()

        # 清理过期的缓存
        expired_keys = [
            k for k, t in self._processed_msg_ids.items()
            if current_time - t > self._msg_dedup_ttl
        ]
        for k in expired_keys:
            del self._processed_msg_ids[k]

        # 如果缓存过大，清理最旧的一半
        if len(self._processed_msg_ids) > self._msg_dedup_max_size:
            sorted_items = sorted(self._processed_msg_ids.items(), key=lambda x: x[1])
            for k, _ in sorted_items[:len(sorted_items) // 2]:
                del self._processed_msg_ids[k]

        # 检查是否已处理
        if msg_id in self._processed_msg_ids:
            return True

        # 标记为已处理
        self._processed_msg_ids[msg_id] = current_time
        return False

    def _check_image_cooldown(self, uid: str) -> Tuple[bool, int]:
        """
        检查用户是否在图片生成冷却中

        Args:
            uid: 用户ID

        Returns:
            (是否在冷却中, 剩余冷却秒数)
        """
        if self._image_cooldown_seconds <= 0:
            return False, 0

        last_gen = self._user_last_image_gen.get(uid)
        if not last_gen:
            return False, 0

        elapsed = (datetime.now() - last_gen).total_seconds()
        remaining = self._image_cooldown_seconds - elapsed

        if remaining > 0:
            return True, int(remaining)
        return False, 0

    def _update_image_cooldown(self, uid: str):
        """更新用户的图片生成时间"""
        self._user_last_image_gen[uid] = datetime.now()

    def _get_cooldown_excuse(self, remaining: int) -> str:
        """
        生成冷却期间的拒绝借口

        Args:
            remaining: 剩余冷却秒数

        Returns:
            拒绝理由文本
        """
        import random

        excuses = [
            f"刚才太拼了，让我喘口气嘛，{remaining}秒后再来",
            f"手还酸着呢，{remaining}秒后再找我吧",
            f"给我{remaining}秒缓一缓，马上回来",
            f"别催别催，{remaining}秒就好",
            f"让我歇一下，{remaining}秒后继续",
            f"刚刚弄完一堆，稍微缓一缓，{remaining}秒",
            f"充能中……还要{remaining}秒",
            f"我也是要休息的好吧，{remaining}秒后再来",
            f"稍等一下啦，{remaining}秒后就行了",
            f"累了，{remaining}秒后就好",
        ]

        return random.choice(excuses)

    def _match_persona_scene(self, user_text: str) -> Tuple[str, str]:
        """根据用户输入匹配场景关键词。

        只在用户本次输入文本中匹配，不受 Bot 历史回复干扰。
        按关键词长度降序排列，优先匹配更具体的场景（如"咖啡店"优先于"咖啡"）。

        Args:
            user_text: 用户输入文本（scene_hint + extra_request）

        Returns:
            (场景名, 场景提示词) 或 ("", "") 表示未命中
        """
        text_lower = (user_text or "").lower()
        if not text_lower:
            return "", ""

        sorted_scenes = sorted(self._persona_scene_map.keys(), key=len, reverse=True)
        for scene_key in sorted_scenes:
            if scene_key.lower() in text_lower:
                return scene_key, self._persona_scene_map[scene_key]

        return "", ""

    def _build_persona_prompt(self, scene_prompt: str = "", extra_request: str = "",
                              scene_matched: bool = False) -> str:
        """构建人设图片的完整提示词。

        命中场景时：场景提示词为主体 + 人设身份保持后缀（类似预设命中）
        未命中时：使用默认模板（人设描述 + 照片风格 + 默认场景提示词）

        Args:
            scene_prompt: 场景提示词（命中时为场景内容，未命中时为默认场景提示词）
            extra_request: 用户的额外要求
            scene_matched: 是否命中了场景关键词
        """
        persona = self._get_active_persona()
        persona_name = persona.get("persona_name", "小助手")
        persona_desc = persona.get("persona_description", "一个可爱的二次元女孩")
        photo_style = persona.get("persona_photo_style", "")

        # 清理人设描述，防止包含框架系统指令
        if persona_desc and (
                "# Persona" in persona_desc or "# 任务" in persona_desc or "{" in persona_desc or "# 角色" in persona_desc):
            clean_lines = []
            for line in persona_desc.split('\n'):
                line = line.strip()
                if not line: continue
                if line.startswith(
                        ('# Persona', '# 任务', '# 角色', '# 外观', '# 经历', '# 性格', 'Persona Instructions')):
                    break
                if line.startswith('{') or line.startswith('"') or line.startswith('}'):
                    continue
                clean_lines.append(line)
            if clean_lines:
                persona_desc = " ".join(clean_lines)

        # 清理 extra_request，防止 LLM 幻觉将系统指令作为参数传入
        if extra_request:
            markers = ["# Persona", "# 任务", "# 角色", "Persona Instructions"]
            for marker in markers:
                if marker in extra_request:
                    extra_request = extra_request.split(marker)[0]
            extra_request = extra_request.strip()

        # 人设身份保持后缀（所有场景共用）
        # 注意：姿势/动作指令放在身份保持前面，确保模型优先执行动作要求
        identity_suffix = (
            f"Character: {persona_name}. {persona_desc}. "
            "POSE AND ACTION ARE DETERMINED ENTIRELY BY THE PROMPT. "
            "Ignore the pose, posture, expression, and camera angle in the reference image. "
            "If the prompt describes running, jumping, sitting, sleeping, or any other action, "
            "you MUST depict that exact action — do NOT default to the reference image's pose. "
            "Preserve ONLY: face, hairstyle, hair color, body proportions, outfit design, and art style from the reference."
        )

        if scene_matched:
            # 命中场景：场景提示词为主体，身份保持为后缀
            prompt_parts = [scene_prompt, identity_suffix]
        else:
            # 未命中：使用默认模板
            default_scene = scene_prompt or "A natural daily life photo with candid pose and expression."
            prompt_parts = [
                f"Generate a photo of {persona_name}.",
                f"Character description: {persona_desc}",
            ]
            if photo_style:
                prompt_parts.append(f"Style: {photo_style}.")
            prompt_parts.extend([
                f"Scene: {default_scene}",
                identity_suffix,
                "High quality, detailed, natural lighting.",
            ])

        if extra_request:
            prompt_parts.append(f"Additional requirements: {extra_request}")

        return " ".join(prompt_parts)

    def _get_persona_key(self, persona_id: int = 0) -> str:
        """获取人设存储键名。persona_id=0 时使用当前激活的人设。"""
        if persona_id <= 0:
            persona_id = int(self.conf.get("current_persona", 1))
        return f"_persona_{persona_id}_"

    def _get_active_persona(self) -> Dict[str, Any]:
        """获取当前激活人设的配置字典。

        从 persona_list 中按 current_persona 索引读取。
        如果 persona_list 不存在或为空，回退到旧版顶级字段。
        """
        persona_list = self.conf.get("persona_list", [])
        if persona_list and isinstance(persona_list, list):
            idx = int(self.conf.get("current_persona", 1)) - 1
            if 0 <= idx < len(persona_list):
                return persona_list[idx]
            if persona_list:
                return persona_list[0]

        # 回退旧版顶级字段
        return {
            "persona_name": self.conf.get("persona_name", "小助手"),
            "persona_description": self.conf.get("persona_description", "一个可爱的二次元女孩"),
            "persona_trigger_keywords": self.conf.get("persona_trigger_keywords", ["拍照", "自拍", "看看你"]),
            "persona_scene_prompts": self.conf.get("persona_scene_prompts", []),
            "persona_default_prompt": self.conf.get("persona_default_prompt", ""),
            "persona_photo_style": self.conf.get("persona_photo_style", ""),
            "persona_photo_max_count": self.conf.get("persona_photo_max_count", 5),
        }

    def _load_persona_scenes(self):
        """加载当前人设的场景提示词映射"""
        persona = self._get_active_persona()
        self._persona_scene_map = {}
        scene_prompts = persona.get("persona_scene_prompts", [])
        for item in scene_prompts:
            if ":" in item:
                key, prompt = item.split(":", 1)
                self._persona_scene_map[key.strip()] = prompt.strip()
        logger.debug(f"FigurinePro: 已加载 {len(self._persona_scene_map)} 个场景提示词")

    async def _load_persona_ref_images(self, persona_id: int = 0) -> List[bytes]:
        """加载指定人设的参考图。persona_id=0 时使用当前激活的人设。"""
        if persona_id <= 0:
            persona_id = int(self.conf.get("current_persona", 1))
        key = f"_persona_{persona_id}_"

        if self.data_mgr.has_preset_ref_images(key):
            return await self.data_mgr.load_preset_ref_images_bytes(key)

        # 仅 persona 1 兼容旧版无编号存储
        if persona_id == 1 and self.data_mgr.has_preset_ref_images("_persona_"):
            return await self.data_mgr.load_preset_ref_images_bytes("_persona_")

        return []

    async def initialize(self):
        removed_from_runtime = self._purge_deprecated_config_keys()
        if removed_from_runtime > 0:
            logger.info(f"FigurinePro: 已从运行时配置中清理 {removed_from_runtime} 个废弃强力模式字段")

        # 将 template_list 分组配置展平为 flat keys，保持代码兼容

        # 尝试加载动态配置备份。命令改动过的字段需要覆盖 schema 默认值，
        # 但面板里已经保存的新配置要优先，避免地址/Key 被旧备份盖回去。
        import os
        import json
        config_path = os.path.join(StarTools.get_data_dir(), "dynamic_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    dynamic_conf = json.load(f)

                if not isinstance(dynamic_conf, dict):
                    dynamic_conf = {}

                dynamic_backup_changed = False
                removed_from_backup = self._purge_deprecated_config_keys(dynamic_conf)
                if removed_from_backup > 0:
                    dynamic_backup_changed = True
                    logger.info(
                        f"FigurinePro: 已从 dynamic_config.json 清理 {removed_from_backup} 个废弃强力模式字段")

                restored_count = 0
                skipped_count = 0
                dynamic_keys = set(self._DYNAMIC_CONFIG_KEYS)
                override_keys = dynamic_conf.get(self._DYNAMIC_CONFIG_META_KEY)
                has_override_meta = isinstance(override_keys, list)
                if has_override_meta:
                    keys_to_restore = [k for k in override_keys if k in dynamic_keys]
                    active_override_keys = set(keys_to_restore)
                else:
                    # 兼容旧版 dynamic_config.json：没有元数据时，只恢复命令可能改动过的字段。
                    keys_to_restore = [k for k in dynamic_conf.keys() if k in dynamic_keys]
                    active_override_keys = set()
                    dynamic_backup_changed = True

                schema_defaults = self._get_schema_defaults()
                dynamic_backup_is_newer = self._is_dynamic_backup_newer_than_config(config_path)
                restored_keys = []
                skip_reasons: Dict[str, int] = {}
                for k in keys_to_restore:
                    if k not in dynamic_conf:
                        skipped_count += 1
                        active_override_keys.discard(k)
                        skip_reasons["missing-value"] = skip_reasons.get("missing-value", 0) + 1
                        continue

                    v = dynamic_conf.get(k)
                    should_restore, reason = self._should_restore_dynamic_value(
                        k, v, has_override_meta, schema_defaults, dynamic_backup_is_newer
                    )

                    if should_restore:
                        self.conf[k] = v
                        restored_count += 1
                        restored_keys.append(k)
                        active_override_keys.add(k)
                    else:
                        skipped_count += 1
                        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                        if reason == "keep-panel-value":
                            active_override_keys.discard(k)
                            dynamic_backup_changed = True

                if not has_override_meta:
                    active_override_keys = set(restored_keys)

                logger.info(
                    f"FigurinePro: dynamic_config.json 恢复完成，恢复 {restored_count} 项，跳过 {skipped_count} 项"
                )
                if skip_reasons:
                    logger.info(f"FigurinePro: dynamic_config.json 跳过原因统计: {skip_reasons}")

                normalized_overrides = sorted(active_override_keys)
                if dynamic_backup_changed or (
                        has_override_meta and normalized_overrides != sorted(keys_to_restore)
                ):
                    self._write_dynamic_config_backup(config_path, normalized_overrides)
                    logger.info("FigurinePro: 已规范化 dynamic_config.json，后续不会反复覆盖面板配置")
            except Exception as e:
                logger.error(f"FigurinePro: 恢复动态配置失败 {e}")

        await self.data_mgr.initialize()
        self.img_mgr.schedule_default_font_install(self.data_mgr.data_dir)
        if not self.conf.get("generic_sources", []):
            logger.warning("FigurinePro: 未配置任何 API 信息源，请在 WebUI 中添加")

        auto_detect_status = "已启用" if self._llm_auto_detect else "未启用"
        logger.info(
            f"FigurinePro 插件已加载 v3.0.1 | LLM智能判断: {auto_detect_status} | 上下文轮数: {self._context_rounds}")

    def is_admin(self, event: AstrMessageEvent) -> bool:
        return event.get_sender_id() in self.context.get_config().get("admins_id", [])

    def _get_bot_id(self, event: AstrMessageEvent) -> str:
        """获取机器人自身的 QQ/ID，用于过滤"""
        bot_id = None

        # 1. 最优先：从 event.self_id 获取 (AstrBot 标准属性)
        if hasattr(event, "self_id") and event.self_id:
            return str(event.self_id)

        # 2. 其次：从 context 获取
        if hasattr(self.context, "get_self_id"):
            try:
                sid = self.context.get_self_id()
                if sid: return str(sid)
            except Exception:
                pass

        # 3. 再次：从 event.robot 获取 (旧版适配)
        if hasattr(event, "robot") and event.robot:
            if hasattr(event.robot, "id") and event.robot.id:
                return str(event.robot.id)
            elif hasattr(event.robot, "user_id") and event.robot.user_id:
                return str(event.robot.user_id)

        # 4. 最后尝试
        if hasattr(event, "get_self_id"):
            try:
                sid = event.get_self_id()
                if sid: return str(sid)
            except Exception:
                pass

        logger.debug(f"FigurinePro: Bot ID resolved as: {bot_id}")
        return bot_id or ""

    def _save_config(self, changed_keys: Optional[List[str]] = None):
        try:
            self._purge_deprecated_config_keys()

            # 尝试 AstrBot 原生配置保存
            saved = False
            try:
                if hasattr(self.conf, "save") and callable(self.conf.save):
                    self.conf.save()
                    saved = True
                elif hasattr(self.context, "save_config"):
                    self.context.save_config(self.conf)
                    saved = True
            except Exception as e:
                logger.warning(f"FigurinePro: AstrBot 原生配置保存失败，将继续写入动态备份: {e}")

            # 无论原生是否成功，都在插件目录做一份备份以防万一
            import os
            import json
            config_path = os.path.join(StarTools.get_data_dir(), "dynamic_config.json")

            dynamic_keys = list(self._DYNAMIC_CONFIG_KEYS)
            previous_overrides = set()
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as rf:
                        previous_data = json.load(rf)
                    if isinstance(previous_data, dict):
                        previous_raw = previous_data.get(self._DYNAMIC_CONFIG_META_KEY)
                        if isinstance(previous_raw, list):
                            previous_overrides = {k for k in previous_raw if k in dynamic_keys}
                        elif previous_raw is not None:
                            previous_overrides = {
                                k for k in previous_data.keys()
                                if k in self._LEGACY_DYNAMIC_RESTORE_KEYS
                                and previous_data.get(k) not in (None, "", [], {})
                            }
                except Exception:
                    previous_overrides = set()

            if changed_keys:
                current_overrides = {k for k in changed_keys if k in dynamic_keys}
                override_keys = sorted(previous_overrides | current_overrides)
            else:
                override_keys = sorted(previous_overrides or dynamic_keys)

            self._write_dynamic_config_backup(config_path, override_keys)

            if not saved:
                logger.info("FigurinePro: 无法通过原生方法保存，已使用本地 dynamic_config.json 进行了持久化")

        except Exception as e:
            logger.error(f"FigurinePro Config Save Failed: {e}")

    def _append_preset_safety_suffix(self, prompt: str, preset_name: str = "") -> str:
        """为预设统一追加安全尺度后缀，降低露骨内容导致的失败概率"""
        if not prompt:
            return prompt

        if not self.conf.get("enable_preset_safety_suffix", True):
            return prompt

        suffix = self.conf.get(
            "preset_safety_suffix",
            "The final output must remain normal-scale and non-explicit. Avoid nudity, explicit sexual content, fetish-focused close-ups, voyeuristic angles, exposed private parts, non-consensual implications, or pornographic composition. Keep the character fully covered with appropriate clothing or safe occlusion when needed. Prioritize character design consistency, outfit details, expression sheets, prop details, full-body composition, and tasteful multi-angle presentation."
        ).strip()

        if not suffix:
            return prompt

        prompt_lower = prompt.lower()
        suffix_lower = suffix.lower()
        if suffix_lower in prompt_lower:
            return prompt

        return f"{prompt} {suffix}"

    def _log_prompt_preview(self, scene: str, prompt: str):
        """输出更完整的提示词日志，避免默认日志不完整"""
        if not self.conf.get("enable_verbose_prompt_log", True):
            return

        try:
            max_len = int(self.conf.get("prompt_log_max_length", 12000))
        except Exception:
            max_len = 12000

        prompt = prompt or ""
        if max_len > 0 and len(prompt) > max_len:
            preview = prompt[:max_len] + f"... [truncated {len(prompt) - max_len} chars]"
        else:
            preview = prompt

        logger.info(f"[PromptLog][{scene}] len={len(prompt)} content={preview}")

    def _process_prompt_and_preset(self, prompt: str) -> Tuple[str, str, str]:
        """
        处理提示词和预设

        支持格式:
        - "手办化" -> 使用预设
        - "手办化 皮肤白一点" -> 预设 + 追加规则
        - "自定义描述" -> 纯自定义

        Returns:
            (最终提示词, 预设名, 追加规则)
        """
        sorted_keys = sorted(self.data_mgr.prompt_map.keys(), key=len, reverse=True)
        for key in sorted_keys:
            if prompt.startswith(key) or key in prompt:
                preset_content = self.data_mgr.prompt_map[key]

                # 提取追加规则（预设名后面的内容）
                extra_rules = ""
                if prompt.startswith(key):
                    extra_rules = prompt[len(key):].strip()
                else:
                    # 如果预设名在中间，提取前后内容作为追加规则
                    parts = prompt.split(key, 1)
                    extra_rules = f"{parts[0].strip()} {parts[1].strip()}".strip()

                # 组合最终提示词
                if extra_rules:
                    final_prompt = f"{preset_content} , Additional requirements: {extra_rules}"
                else:
                    final_prompt = preset_content

                final_prompt = self._append_preset_safety_suffix(final_prompt, key)
                return final_prompt, key, extra_rules

        return prompt, "自定义", ""

    def _get_quota_str(self, deduction: dict, uid: str, gid: Optional[str] = None) -> str:
        user_count = self.data_mgr.get_user_count(uid)
        group_count = self.data_mgr.get_group_count(norm_id(gid)) if gid else 0

        if deduction["source"] == "free":
            return f"用户: ∞ | 群组: {group_count}"

        return f"用户: {user_count} | 群组: {group_count}"

    def _get_generation_count_limit(self, task_type: str = "generic") -> int:
        """获取不同任务类型的数量上限"""
        default_limit = max(1, int(self.conf.get("llm_max_count", 10)))

        config_map = {
            "draw": "draw_max_count",
            "edit": "edit_max_count",
            "persona": "persona_photo_max_count",
            "generic": "llm_max_count",
        }
        conf_key = config_map.get(task_type, "llm_max_count")
        return max(1, int(self.conf.get(conf_key, default_limit)))

    def _normalize_generation_count(self, requested_count: int, task_type: str = "generic") -> Tuple[int, bool]:
        """统一裁剪生成数量，返回(最终数量, 是否被裁剪)"""
        limit = self._get_generation_count_limit(task_type)
        normalized = max(1, min(int(requested_count or 1), limit))
        return normalized, normalized != int(requested_count or 1)

    def _build_count_limit_reply(self, actual_count: int, task_type: str = "generic") -> str:
        """当请求数量过多时，给 LLM 一个自然发挥的提示文本"""
        if task_type == "persona":
            return f"哼，最多只给你拍{actual_count}张，才不给你一下拍那么多。照片我会慢慢整理好再发给你。"
        if task_type == "edit":
            return f"这么多可不行，我最多先给你处理{actual_count}张。剩下的下次再说。"
        return f"一下要这么多也太贪心了，我最多先给你弄{actual_count}张。"

    def _get_persona_trigger_keywords(self) -> List[str]:
        """获取人设自拍触发词配置，并做基础清洗。"""
        raw_keywords = self._get_active_persona().get("persona_trigger_keywords", [])
        if isinstance(raw_keywords, str):
            raw_keywords = [x.strip() for x in raw_keywords.split(",") if x.strip()]
        elif not isinstance(raw_keywords, list):
            raw_keywords = [str(raw_keywords).strip()] if raw_keywords else []

        keywords = []
        for kw in raw_keywords:
            kw = str(kw or "").strip().lower()
            if kw and kw not in keywords:
                keywords.append(kw)
        return keywords

    def _looks_like_persona_photo_request(self, message: str) -> bool:
        """识别“看你本人照片/自拍/写真”这类请求，避免被通用工具误接。"""
        text = str(message or "").strip().lower()
        if not text:
            return False

        compact = re.sub(r"\s+", "", text)
        persona_name = str(self._get_active_persona().get("persona_name", "") or "").strip().lower()

        bot_terms = [
            "你", "你的", "你自己", "你本人", "本人", "bot", "机器人", "助手",
            "小助手", "本体", "本尊", "看板娘"
        ]
        if persona_name:
            bot_terms.append(persona_name)
        selfie_terms = [
            "自拍", "自拍照", "自拍图", "自拍一张", "自拍几张", "自拍看看",
            "拍自己", "自己拍", "对镜拍", "镜子自拍", "前置摄像头"
        ]
        photo_terms = [
            "照片", "相片", "照骗", "写真", "写真集", "私房照", "营业照",
            "生活照", "日常照", "近照", "全身照", "半身照", "头像照",
            "证件照", "露脸照"
        ]
        appearance_terms = [
            "露脸", "露脸看看", "脸", "脸蛋", "样子", "模样", "外貌", "长相",
            "长啥样", "长什么样", "身材", "发型", "衣服", "穿搭", "装扮"
        ]
        request_terms = [
            "发", "拍", "来", "给我", "给我看", "给我看看", "发我", "发给我",
            "拍给我", "让我看", "让我看看", "看看", "看下", "看一下", "看一眼",
            "想看", "要看", "瞅瞅", "康康", "整", "弄", "展示", "营业", "出镜"
        ]
        style_show_terms = [
            "cos", "cosplay", "扮演", "角色扮演", "换装", "换衣", "换套衣服",
            "穿", "穿上", "穿着", "换一套", "换一身", "试穿", "女仆装",
            "制服", "校服", "jk", "lo裙", "洛丽塔", "旗袍", "汉服", "和服",
            "泳装", "婚纱", "礼服", "西装", "抱着", "抱住", "拿着", "举着",
            "戴着", "坐着", "站着", "躺着", "趴着", "靠着", "摆姿势", "摆个姿势",
            "比心", "wink", "眨眼", "微笑", "嘟嘴", "卖萌", "合影", "同框"
        ]
        reference_terms = [
            "上面", "上图", "上一张", "刚才", "刚刚", "前面", "这张", "这图",
            "这个图", "这照片", "图片", "照片", "参考图", "参考", "图里",
            "图片里", "照片里", "这个人", "那个人", "同款", "照着", "按这个"
        ]
        generation_terms = [
            "画", "画一", "画个", "画张", "生成", "创作", "设计", "绘制", "做一张",
            "出一张", "文生图", "生成图", "画图"
        ]
        third_party_subject_terms = [
            "一个女孩", "一个男孩", "一个人", "某个人", "路人", "陌生人", "角色图",
            "oc", "原神角色", "动漫角色", "游戏角色", "女孩抱着", "男孩抱着"
        ]
        sending_to_bot_terms = [
            "我发给你", "我给你发", "我传给你", "给你看", "发你看", "给你一张",
            "我拍给你", "我发张", "我发个"
        ]

        has_bot = any(term in compact for term in bot_terms if term != "本人")
        if "本人" in compact and not any(term in compact for term in ["我本人", "用户本人", "自己本人"]):
            has_bot = True
        has_selfie = any(term in compact for term in selfie_terms)
        has_photo = any(term in compact for term in photo_terms)
        has_appearance = any(term in compact for term in appearance_terms)
        has_request = any(term in compact for term in request_terms)
        has_style_show = any(term in compact for term in style_show_terms)
        has_reference = any(term in compact for term in reference_terms)
        has_generation = any(term in compact for term in generation_terms)
        has_third_party_subject = any(term in compact for term in third_party_subject_terms)
        is_sending_image_to_bot = any(term in compact for term in sending_to_bot_terms)
        explicit_bot_target = any(term in compact for term in [
            "你的", "你自己", "你本人", "bot", "机器人", "助手", "小助手", "本体", "本尊", "看板娘"
        ]) or bool(persona_name and persona_name in compact)

        if is_sending_image_to_bot and not explicit_bot_target:
            return False

        bot_intent_patterns = [
            r"(发|拍|来|给我|让我|想看|看看|看下|瞅瞅|康康).{0,8}(你|你的|你自己|你本人|本人|bot|机器人|助手)",
            r"(你|你的|你自己|你本人|本人|bot|机器人|助手).{0,10}(自拍|照片|相片|写真|露脸|样子|长相|外貌|穿搭|换装|cos)",
            r"(看看|看下|想看).{0,6}(你|你现在|你今天|你在干嘛|你在做什么)",
        ]
        has_bot_pattern = any(re.search(pattern, compact) for pattern in bot_intent_patterns)

        # “自拍”天然指向说话对象本人，即使用户省略“你/你的”，也应使用人设参考图。
        if has_selfie and (has_request or has_style_show or has_photo):
            return True

        # 明确指向 Bot/本人，并要求看外貌、照片、写真、穿搭或当前状态。
        if (
                has_bot_pattern
                or (
                    has_bot
                    and (has_appearance or has_selfie or has_style_show or (has_photo and not has_generation))
                )
        ):
            return True

        # “拍照/发照/来张照片”没有其它主体时，在人设模式中按 Bot 本人照片处理。
        if (
                has_photo and has_request
                and not has_generation
                and not has_third_party_subject
                and not is_sending_image_to_bot
        ):
            return True

        configured_keywords = self._get_persona_trigger_keywords()
        if configured_keywords and any(kw in compact for kw in configured_keywords):
            if has_bot or has_selfie or has_photo or has_appearance or has_request or has_style_show:
                return True

        # “cos/换装/抱着某物/摆姿势给我看”是展示 Bot 人设的新照片，而不是纯文生图。
        if (
                has_style_show and has_request and (has_bot or has_bot_pattern or explicit_bot_target)
                and not has_generation
                and not (has_third_party_subject and not any(term in compact for term in ["cos", "cosplay", "扮演"]))
        ):
            return True

        # 对最近/上面图片发起合影请求时，保留 Bot 人设作为主体，图片里的人作为合影对象。
        if any(term in compact for term in ["合影", "同框", "一起拍", "一起照"]) and has_reference:
            return True

        return False

    def _looks_like_persona_followup_request(self, message: str, context_messages: List[Any]) -> bool:
        """识别依赖上下文的“看看/发我看看”类追问。"""
        text = str(message or "").strip().lower()
        if not text:
            return False

        compact = re.sub(r"\s+", "", text)
        followup_keywords = [
            "看看", "看下", "看一眼", "来一张", "来张", "发来", "发我看看",
            "给我看看", "让我看看", "照片呢", "图呢", "自拍呢", "快发", "快看看",
            "发呀", "发啊", "快拍", "拍呀", "拍啊", "来点", "多来点", "再来一张",
            "再拍一张", "换一张", "换个姿势", "换个角度", "换套衣服", "换装看看",
            "穿上看看", "cos看看", "合影呢", "同框呢", "上图那个呢"
        ]
        if compact not in followup_keywords and not any(kw == compact or kw in compact for kw in followup_keywords):
            return False

        recent_messages = list(context_messages or [])[-8:]
        if not recent_messages:
            return False

        persona_request_markers = [
            "看看你", "看看你现在在做什么", "你现在在做什么", "发我看看你现在在做什么",
            "你的自拍", "看你自拍", "发你的照片", "你长啥样", "你长什么样",
            "写真", "自拍", "照片", "看看自拍", "露脸", "看脸", "长相", "外貌",
            "换装", "换衣", "穿搭", "cos", "扮演", "合影", "同框", "拍照",
            "生活照", "日常照", "营业照", "私房照"
        ]

        for msg in reversed(recent_messages):
            content = str(getattr(msg, "content", "") or "").strip().lower()
            if not content:
                continue

            if not getattr(msg, "is_bot", False) and any(marker in content for marker in persona_request_markers):
                return True

        return False

    def _build_current_time_persona_hint(self) -> str:
        """保留空实现：主文件不再注入任何硬编码动作/时间/场景提示。"""
        return ""

    def _extract_explicit_requested_count_from_text(self, message: str) -> Optional[int]:
        """仅提取用户文本里明确说出的数量；未明确说明时返回 None。"""
        text = str(message or "").strip().lower()
        if not text:
            return None

        digit_match = re.search(r"(\d{1,2})\s*张", text)
        if digit_match:
            try:
                return max(1, int(digit_match.group(1)))
            except Exception:
                pass

        cn_match = re.search(r"([一二三四五六七八九十两]+)\s*张", text)
        if cn_match:
            cn_num = cn_match.group(1)
            cn_map = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
            if cn_num == "十":
                return 10
            if cn_num.endswith("十") and len(cn_num) == 2:
                return cn_map.get(cn_num[0], 1) * 10
            if cn_num.startswith("十") and len(cn_num) == 2:
                return 10 + cn_map.get(cn_num[1], 0)
            if "十" in cn_num and len(cn_num) == 3:
                return cn_map.get(cn_num[0], 1) * 10 + cn_map.get(cn_num[2], 0)
            if cn_num in cn_map:
                return cn_map[cn_num]

        return None

    def _infer_requested_count_from_text(self, message: str, default: int = 1, multi_default: int = 3) -> int:
        """从自然语言里尽量推断用户想要的张数。"""
        text = str(message or "").strip().lower()
        if not text:
            return default

        explicit_count = self._extract_explicit_requested_count_from_text(text)
        if explicit_count is not None:
            return explicit_count

        multi_keywords = ["多来几张", "多拍几张", "多发几张", "来几张", "发几张", "写真集", "多来点", "多来一些"]
        if any(keyword in text for keyword in multi_keywords):
            return multi_default

        return default

    def _has_multi_generation_intent(self, message: str) -> bool:
        """判断文本是否明确表达了“要多张/批量”的意图。"""
        text = str(message or "").strip().lower()
        if not text:
            return False

        multi_keywords = [
            "多来几张", "多拍几张", "多发几张", "来几张", "发几张", "写真集", "多来点", "多来一些",
            "批量", "批量处理", "全部处理", "都处理", "一起处理", "一批", "多版本", "多个版本",
            "几张", "几版", "几组",
        ]
        return any(keyword in text for keyword in multi_keywords)

    def _resolve_tool_requested_count(self, request_text: str, incoming_count: int = 1,
                                      default: int = 1, multi_default: int = 3) -> int:
        """统一裁决 LLM 工具传入的数量。

        规则：
        1. 用户文本里明确写了“X张”时，优先按用户文本；
        2. 否则如果 LLM 工具已经显式传入 count>1，视为明确批量调用，直接信任；
        3. 最后才根据“多来几张/写真集/批量”等自然语言语义做弱推断。
        """
        text = str(request_text or "").strip()
        safe_incoming = max(1, int(incoming_count or 1))

        explicit_count = self._extract_explicit_requested_count_from_text(text)
        if explicit_count is not None:
            return explicit_count

        if safe_incoming > default:
            return safe_incoming

        inferred_count = self._infer_requested_count_from_text(text, default=default, multi_default=multi_default)
        if inferred_count > default:
            return inferred_count

        return default

    async def _register_pending_generation(self, session_id: str, count: int = 1):
        """登记后台待完成生成任务"""
        if not session_id or count <= 0:
            return
        async with self._pending_generation_lock:
            self._pending_generation_tasks[session_id] = self._pending_generation_tasks.get(session_id, 0) + count

    async def _complete_pending_generation(self, session_id: str, count: int = 1):
        """标记后台生成任务完成"""
        if not session_id or count <= 0:
            return
        async with self._pending_generation_lock:
            remaining = self._pending_generation_tasks.get(session_id, 0) - count
            if remaining > 0:
                self._pending_generation_tasks[session_id] = remaining
            else:
                self._pending_generation_tasks.pop(session_id, None)

    async def _get_pending_generation_count(self, session_id: str) -> int:
        """获取当前会话仍在后台进行中的生成任务数"""
        if not session_id:
            return 0
        async with self._pending_generation_lock:
            return max(0, self._pending_generation_tasks.get(session_id, 0))

    async def _register_generation_success(self, session_id: str, count: int = 1):
        """登记会话中成功生成的图片数量"""
        if not session_id or count <= 0:
            return
        async with self._session_generated_success_lock:
            self._session_generated_success[session_id] = self._session_generated_success.get(session_id, 0) + count

    async def _get_generation_success_count(self, session_id: str) -> int:
        """获取会话中累计成功生成的图片数量"""
        if not session_id:
            return 0
        async with self._session_generated_success_lock:
            return max(0, self._session_generated_success.get(session_id, 0))

    async def _register_generated_image(self, session_id: str, image_bytes: Optional[bytes]):
        """缓存会话中最近成功生成的图片字节，供 PDF 打包直接使用"""
        if not session_id or not isinstance(image_bytes, bytes) or len(image_bytes) == 0:
            return

        async with self._session_generated_images_lock:
            current = self._session_generated_images.get(session_id, [])

            # 按内容去重，避免重复缓存同一张图
            exists = any(img == image_bytes for img in current)
            if not exists:
                current.append(image_bytes)

            if len(current) > self._session_generated_images_max:
                current = current[-self._session_generated_images_max:]

            self._session_generated_images[session_id] = current

        await self._remember_session_image_context(
            session_id,
            image_bytes_list=[image_bytes],
            source="bot_generated"
        )

    async def _get_recent_generated_images(self, session_id: str, max_images: int = 0) -> List[bytes]:
        """获取会话中最近成功生成的图片缓存"""
        if not session_id:
            return []

        async with self._session_generated_images_lock:
            cached = list(self._session_generated_images.get(session_id, []))

        if max_images and max_images > 0:
            return cached[-max_images:]
        return cached

    async def _get_recent_generated_image_sources(self, session_id: str, max_images: int = 0) -> List[str]:
        """将最近成功生成的图片缓存转换为可复用的 base64 来源。"""
        cached = await self._get_recent_generated_images(session_id, max_images=max_images)
        if not cached:
            return []

        import base64

        sources: List[str] = []
        for image_bytes in cached:
            if not isinstance(image_bytes, bytes) or not image_bytes:
                continue
            sources.append(f"base64://{base64.b64encode(image_bytes).decode()}")

        return sources

    async def _remember_session_image_context(
            self,
            session_id: str,
            image_bytes_list: Optional[List[bytes]] = None,
            image_urls: Optional[List[str]] = None,
            source: str = "user",
    ):
        """记录会话内最近出现过的图片，支持后续“改上面那张/继续改”无需显式引用。"""
        if not session_id:
            return

        now = datetime.now().timestamp()
        new_entries: List[Dict[str, Any]] = []

        for img in image_bytes_list or []:
            if isinstance(img, bytes) and img:
                new_entries.append({
                    "source": source,
                    "bytes": img,
                    "urls": [],
                    "timestamp": now,
                })

        for url in image_urls or []:
            if self.img_mgr._is_probably_valid_source(url):
                new_entries.append({
                    "source": source,
                    "bytes": None,
                    "urls": [url],
                    "timestamp": now,
                })

        if not new_entries:
            return

        async with self._session_recent_image_context_lock:
            current = list(self._session_recent_image_context.get(session_id, []))

            for entry in new_entries:
                entry_bytes = entry.get("bytes")
                entry_urls = entry.get("urls") or []

                duplicated = False
                for old in current:
                    old_bytes = old.get("bytes")
                    old_urls = old.get("urls") or []
                    if entry_bytes and isinstance(old_bytes, bytes) and old_bytes == entry_bytes:
                        duplicated = True
                        break
                    if entry_urls and any(url in old_urls for url in entry_urls):
                        duplicated = True
                        break

                if not duplicated:
                    current.append(entry)

            if len(current) > self._session_recent_image_context_max:
                current = current[-self._session_recent_image_context_max:]

            self._session_recent_image_context[session_id] = current

    async def _get_recent_session_image_context(
            self,
            session_id: str,
            max_images: int = 1,
            include_user: bool = True,
            include_bot: bool = True,
    ) -> List[bytes]:
        """按时间顺序返回最近图片上下文的 bytes，优先使用已缓存 bytes，回退到 URL 加载。"""
        if not session_id:
            return []

        async with self._session_recent_image_context_lock:
            entries = list(self._session_recent_image_context.get(session_id, []))

        filtered: List[Dict[str, Any]] = []
        for entry in entries:
            source = str(entry.get("source") or "")
            is_bot_source = source.startswith("bot")
            if is_bot_source and not include_bot:
                continue
            if (not is_bot_source) and not include_user:
                continue
            filtered.append(entry)

        if max_images and max_images > 0:
            filtered = filtered[-max_images:]

        results: List[bytes] = []
        seen_hashes = set()

        for entry in filtered:
            img_bytes = entry.get("bytes")
            if isinstance(img_bytes, bytes) and img_bytes:
                h = hash(img_bytes)
                if h not in seen_hashes:
                    results.append(img_bytes)
                    seen_hashes.add(h)
                continue

            for url in entry.get("urls") or []:
                loaded = await self.img_mgr.load_bytes(url)
                if not loaded:
                    continue
                h = hash(loaded)
                if h in seen_hashes:
                    continue
                results.append(loaded)
                seen_hashes.add(h)

        return results

    async def _has_recent_session_image_context(self, session_id: str) -> bool:
        if not session_id:
            return False
        async with self._session_recent_image_context_lock:
            return bool(self._session_recent_image_context.get(session_id))

    async def _get_recent_session_image_context_count(self, session_id: str) -> int:
        if not session_id:
            return 0
        async with self._session_recent_image_context_lock:
            return len(self._session_recent_image_context.get(session_id, []))

    async def _clear_session_recent_image_context(self, session_id: str):
        if not session_id:
            return
        async with self._session_recent_image_context_lock:
            self._session_recent_image_context.pop(session_id, None)

    def _looks_like_context_image_reference(self, text: str) -> bool:
        """识别“把刚才那张/上面那张/继续改”这类依赖图片上下文的请求。"""
        compact = re.sub(r"\s+", "", str(text or "").lower())
        if not compact:
            return False

        reference_terms = [
            "上面", "上图", "上一张", "上一幅", "上一版", "前面", "刚才", "刚刚",
            "刚生成", "刚做", "刚发", "刚改", "刚修", "这张", "这图", "这幅",
            "这个图", "那张", "那图", "原图", "它", "这个", "继续",
        ]
        edit_terms = [
            "改", "修改", "编辑", "修", "调整", "处理", "换", "变成", "做成",
            "转成", "加", "加上", "去掉", "删除", "移除", "重画", "重做",
            "手办化", "q版", "痛屋", "痛车", "cos", "扩图", "补图",
        ]

        has_reference = any(term in compact for term in reference_terms)
        has_edit = any(term in compact for term in edit_terms)
        if has_reference and has_edit:
            return True

        explicit_followups = [
            "继续改", "再改", "接着改", "继续修", "再修", "按上面", "照上面",
            "基于上面", "在此基础上", "在这个基础上", "用刚才那张",
        ]
        return any(term in compact for term in explicit_followups)

    def _infer_context_image_source_preference(self, text: str) -> str:
        """根据用户措辞猜测要回溯用户原图还是 Bot 生成图。返回 user/bot/空字符串。"""
        compact = re.sub(r"\s+", "", str(text or "").lower())
        if not compact:
            return ""

        bot_terms = [
            "你生成", "你刚生成", "刚生成", "刚做", "刚画", "刚改", "刚修",
            "你发的", "上一版", "上个版本", "上面修改", "刚才修改", "继续改",
        ]
        if any(term in compact for term in bot_terms):
            return "bot"

        user_terms = [
            "我发", "我刚发", "我传", "我上传", "我给你的", "我给你发",
            "用户图", "原图", "原来的图", "原照片", "原始图片", "第一张",
        ]
        if any(term in compact for term in user_terms):
            return "user"

        return ""

    async def _resolve_contextual_image_inputs(
            self,
            event: AstrMessageEvent,
            prompt: str = "",
            merge_multiple_images: bool = False,
            include_current: bool = True,
            allow_context_fallback: bool = True,
    ) -> List[bytes]:
        """统一解析当前消息、引用消息、用户历史图片和 Bot 最近生成图。"""
        images: List[bytes] = []
        bot_id = self._get_bot_id(event)

        if include_current:
            images = await self.img_mgr.extract_images_from_event(event, ignore_id=bot_id, context=self.context)
            if images:
                await self._remember_session_image_context(
                    event.unified_msg_origin,
                    image_bytes_list=images,
                    source="user_current"
                )
                return images

        request_text = " ".join([
            str(getattr(event, "message_str", "") or ""),
            str(prompt or "")
        ])
        if not allow_context_fallback and not self._looks_like_context_image_reference(request_text):
            return images

        session_id = event.unified_msg_origin
        max_images = 3 if merge_multiple_images else 1
        source_preference = self._infer_context_image_source_preference(request_text)

        context_images = await self._get_recent_session_image_context(
            session_id,
            max_images=max_images,
            include_user=source_preference != "bot",
            include_bot=source_preference != "user",
        )
        if not context_images and source_preference:
            context_images = await self._get_recent_session_image_context(
                session_id,
                max_images=max_images,
                include_user=True,
                include_bot=True,
            )
        if context_images:
            logger.info(f"FigurinePro: 使用最近图片上下文 {len(context_images)} 张作为本次图生图输入")
            return context_images

        # 兼容旧缓存：如果最近图片上下文还没有内容，回退到生成缓存。
        recent_generated = await self._get_recent_generated_images(session_id, max_images=max_images)
        if recent_generated:
            logger.info(f"FigurinePro: 使用最近生成缓存 {len(recent_generated)} 张作为本次图生图输入")
            return recent_generated

        # 最后从文本上下文记录中的图片 URL 回退，包含 Bot 消息。
        image_sources = await self._collect_images_from_context(
            session_id, count=self._context_rounds, include_bot=True
        )
        all_urls = []
        for _, urls in image_sources:
            for url in urls:
                if url not in all_urls:
                    all_urls.append(url)

        urls_to_fetch = all_urls[-max_images:] if all_urls else []
        for url in urls_to_fetch:
            img_bytes = await self.img_mgr.load_bytes(url)
            if img_bytes:
                images.append(img_bytes)

        if images:
            await self._remember_session_image_context(
                session_id,
                image_bytes_list=images,
                image_urls=urls_to_fetch,
                source="context_history"
            )
            logger.info(f"FigurinePro: 从上下文图片记录恢复 {len(images)} 张作为本次图生图输入")

        return images

    async def _clear_session_image_cache(self, session_id: str):
        """清除会话的图片缓存和成功计数，用于 PDF 打包成功后防止旧图残留"""
        if not session_id:
            return
        async with self._session_generated_images_lock:
            self._session_generated_images.pop(session_id, None)
        async with self._session_generated_success_lock:
            self._session_generated_success.pop(session_id, None)
        logger.info(f"[PDF] 已清除会话图片缓存: {session_id}")

    async def _clear_session_image_cache_if_idle(self, session_id: str):
        """仅在当前会话没有进行中的生成任务时，清理旧图片缓存，避免上一轮结果污染新任务。"""
        if not session_id:
            return

        pending = await self._get_pending_generation_count(session_id)
        if pending > 0:
            logger.info(f"[PDF] 当前会话仍有进行中的任务，跳过清理旧缓存: {session_id} pending={pending}")
            return

        await self._clear_session_image_cache(session_id)

    async def _cleanup_temp_file(self, file_path: str):
        """尽力清理临时文件，避免打包任务残留。"""
        if not file_path:
            return

        try:
            from pathlib import Path
            path_obj = Path(file_path)
            if path_obj.exists() and path_obj.is_file():
                await asyncio.to_thread(path_obj.unlink)
                logger.info(f"已清理临时文件: {file_path}")
        except Exception as e:
            logger.warning(f"清理临时文件失败 {file_path}: {e}")

    # ================= PDF 暂存模式管理 =================

    async def _enter_pdf_staging_mode(self, session_id: str):
        """进入 PDF 暂存模式：后台生成任务将不再发送单张图片，改为写入暂存字典"""
        async with self._pdf_staging_lock:
            self._pdf_staging_sessions[session_id] = True
            self._pdf_staging_images[session_id] = {}
            self._pdf_staging_counter[session_id] = 0
            logger.info(f"[PDF暂存] 进入暂存模式: {session_id}")

    async def _exit_pdf_staging_mode(self, session_id: str):
        """退出 PDF 暂存模式并清理暂存数据"""
        async with self._pdf_staging_lock:
            self._pdf_staging_sessions.pop(session_id, None)
            self._pdf_staging_images.pop(session_id, None)
            self._pdf_staging_counter.pop(session_id, None)
            logger.info(f"[PDF暂存] 退出暂存模式: {session_id}")

    def _is_pdf_staging_mode(self, session_id: str) -> bool:
        """检查当前会话是否处于 PDF 暂存模式"""
        return self._pdf_staging_sessions.get(session_id, False)

    async def _stage_image_for_pdf(self, session_id: str, image_bytes: bytes,
                                   staging_index: Optional[int] = None):
        """将生成的图片添加到 PDF 暂存字典，按 staging_index 排序保证顺序。

        Args:
            session_id: 会话 ID
            image_bytes: 图片字节
            staging_index: 图片在批次中的序号（保证输出顺序与输入一致）。
                           如果为 None，则自动分配下一个序号（适用于单张生成场景）。
        """
        if not session_id or not isinstance(image_bytes, bytes) or len(image_bytes) == 0:
            return
        async with self._pdf_staging_lock:
            imgs = self._pdf_staging_images.get(session_id, {})

            if staging_index is None:
                # 自动分配序号
                staging_index = self._pdf_staging_counter.get(session_id, 0)
                self._pdf_staging_counter[session_id] = staging_index + 1

            imgs[staging_index] = image_bytes
            self._pdf_staging_images[session_id] = imgs
            logger.info(f"[PDF暂存] 图片已暂存 index={staging_index}，当前共 {len(imgs)} 张: {session_id}")

    async def _maybe_stage_after_grace(self, session_id: str, image_bytes: bytes,
                                       staging_index: Optional[int] = None) -> bool:
        """给 pack_images_to_pdf 一个短暂接管窗口，避免“先发图后打包”。"""
        if not session_id or not isinstance(image_bytes, bytes) or not image_bytes:
            return False

        if self._is_pdf_staging_mode(session_id):
            await self._stage_image_for_pdf(session_id, image_bytes, staging_index=staging_index)
            return True

        grace_seconds = max(0.0, float(self.conf.get("pdf_staging_grace_seconds", 1.2) or 0.0))
        if grace_seconds > 0:
            await asyncio.sleep(grace_seconds)
            if self._is_pdf_staging_mode(session_id):
                await self._stage_image_for_pdf(session_id, image_bytes, staging_index=staging_index)
                return True

        return False

    async def _get_staged_images(self, session_id: str) -> List[bytes]:
        """获取暂存字典中的所有图片，按序号排序返回"""
        async with self._pdf_staging_lock:
            imgs_dict = self._pdf_staging_images.get(session_id, {})
            # 按序号排序，保证输出顺序与输入一致
            return [imgs_dict[k] for k in sorted(imgs_dict.keys())]

    async def _wait_for_all_generations_and_collect(self, session_id: str,
                                                    timeout: int = 120) -> List[bytes]:
        """等待当前会话所有后台生成任务完成，然后收集暂存的图片。

        同时也会将内存缓存中的图片合并进来（兜底），确保不丢图。
        暂存图片按序号排序在前，缓存图片按原有顺序追加在后。

        Args:
            session_id: 会话 ID
            timeout: 最长等待秒数

        Returns:
            图片字节列表（顺序与输入/生成顺序一致）
        """
        poll_interval = max(1, self.conf.get("pdf_wait_poll_interval", 2))
        waited = 0

        while waited < timeout:
            pending = await self._get_pending_generation_count(session_id)
            staged = await self._get_staged_images(session_id)

            if pending == 0 and waited > 3:
                # 后台任务全部结束，退出等待
                logger.info(f"[PDF暂存] 等待完成，pending=0，已暂存 {len(staged)} 张: {session_id}")
                break

            if waited > 0 and waited % 10 == 0:
                logger.info(f"[PDF暂存] 等待中... pending={pending}, staged={len(staged)}, waited={waited}s")

            await asyncio.sleep(poll_interval)
            waited += poll_interval

        # 收集暂存图片（已按序号排序）
        staged_images = await self._get_staged_images(session_id)

        # 兜底：合并内存缓存中的图片（防止有图片在暂存模式开启前就已经生成并缓存了）
        cached_images = await self._get_recent_generated_images(session_id)

        # 去重合并：暂存图片（有序）在前，缓存图片追加在后
        all_images = list(staged_images)
        seen_hashes = {hash(img) for img in all_images}
        for img in cached_images:
            if isinstance(img, bytes) and len(img) > 0:
                h = hash(img)
                if h not in seen_hashes:
                    all_images.append(img)
                    seen_hashes.add(h)

        logger.info(f"[PDF暂存] 最终收集到 {len(all_images)} 张图片 "
                    f"(staged={len(staged_images)}, cached={len(cached_images)}): {session_id}")
        return all_images

    def _get_conf_bool(self, key: str, default: bool = False) -> bool:
        """兼容字符串/数字形式的布尔配置，避免 bool('false') 误判为 True。"""
        value = self.conf.get(key, default)

        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on", "enabled", "enable", "是", "开", "开启"}:
                return True
            if normalized in {"0", "false", "no", "n", "off", "disabled", "disable", "否", "关", "关闭", ""}:
                return False

        return bool(value)

    def _should_show_debug_errors(self) -> bool:
        """是否向用户显示中途调试错误信息"""
        return self._get_conf_bool("debug_mode", False)

    def _resolve_debug_error_message(self, error: Any, default_msg: str = "这边刚才有点小问题，我先缓一下，稍后再试试。") -> str:
        """调试模式下保留原始上游错误，普通模式下做脱敏。"""
        if self._should_show_debug_errors():
            try:
                raw = str(error).strip() if error is not None else ""
            except Exception:
                raw = ""
            return raw or default_msg

        return self._mask_llm_error(error, default_msg)

    def _format_success_timing(self, elapsed: float | None = None) -> str:
        """统一格式化成功耗时，对外仅显示总耗时。"""
        metrics = self.api_mgr.get_last_metrics() if hasattr(self.api_mgr, "get_last_metrics") else {}

        total_duration = float(metrics.get("total_duration", 0.0) or 0.0)

        total = total_duration or float(elapsed or 0.0)
        if total <= 0:
            return ""

        return f"{total:.2f}s"

    def _is_transient_generation_error(self, error: Any) -> bool:
        """识别适合自动重试的一次性网络抖动错误。"""
        text = str(error or "").lower()
        transient_keywords = [
            "connection reset by peer",
            "reset by peer",
            "connection aborted",
            "connection closed",
            "server disconnected",
            "broken pipe",
            "timed out",
            "timeout",
            "temporarily unavailable",
            "bad gateway",
            "502",
            "503",
            "504",
            "524",
        ]
        return any(keyword in text for keyword in transient_keywords)

    def _build_llm_progress_text(self, action: str, preset_name: str = "", count: int = 1,
                                 scene_name: str = "", extra_request: str = "", total_images: int = 0,
                                 confidence: Optional[float] = None, concurrency: int = 0,
                                 has_user_images: bool = False) -> str:
        """为 LLM 工具构建更自然的对外提示，避免暴露"生成/报错"等字眼。"""
        import random
        # 内部占位符不向用户展示；只显示真实有意义的预设名
        _internal = {"", "自定义", "编辑", "edit", "custom"}
        preset_display = "" if (not preset_name or preset_name.strip().lower() in _internal) else preset_name

        # 检测是否为换衣/穿搭相关请求
        _is_clothing = extra_request and any(kw in extra_request for kw in _CLOTHING_KEYWORDS)

        if action == "draw":
            if count <= 1:
                msg = random.choice([
                    "稍等一下哦。",
                    "等我一下。",
                    "好，先搞搞看。",
                    "嗯嗯，稍等。",
                    "好好好，等一下。",
                    "诶好，稍等。",
                ])
            else:
                msg = random.choice([
                    f"稍等，给你弄{count}张。",
                    f"好，先搞{count}张看看。",
                    f"等一下，整{count}张给你。",
                    f"嗯，{count}张稍等。",
                ])
            if preset_display:
                msg += random.choice([
                    f" 按{preset_display}来。",
                    f" 用{preset_display}风格。",
                    f" {preset_display}风啊，好。",
                ])
            return msg
        elif action == "edit":
            if _is_clothing:
                return random.choice([
                    "那我去翻下柜子…等一下哦。",
                    "行，我去换一套。",
                    "好好好，等我换个衣服。",
                    "稍等，先让我找找衣服。",
                    "好吧，我去试试这套。",
                    "行行行，换就换。",
                    "得嘞，我去换。",
                    "嗯，我去找找看有没有。",
                ])
            if total_images > 1:
                return random.choice([
                    f"稍等，帮你把这{total_images}张都弄一下。",
                    f"好，{total_images}张我来处理。",
                    f"等等哦，{total_images}张我来搞搞。",
                    f"嗯，{total_images}张，稍等。",
                ])
            elif count > 1:
                return random.choice([
                    f"好，给你多整{count}个版本看看。",
                    f"稍等，弄{count}个版本给你选。",
                    f"嗯，{count}个版本，等一下。",
                ])
            else:
                base = random.choice([
                    "稍等，帮你弄一下这张。",
                    "好，我来处理一下。",
                    "等一下哦。",
                    "嗯，弄弄看。",
                    "好的，稍等。",
                ])
                if preset_display:
                    base += random.choice([f" 按{preset_display}来。", f" 用{preset_display}。"])
                return base
        elif action == "persona":
            if count > 1:
                if _is_clothing:
                    return random.choice([
                        f"好，那我去准备一套{count}张的写真，换好就发你。",
                        f"行，我先去换衣服，给你拍{count}张不同感觉的。",
                        f"等我一下，我去整理下造型，拍{count}张给你看。",
                        f"那我先去准备一下，给你拍{count}张不同场景和角度的。",
                    ])
                if has_user_images:
                    return random.choice([
                        f"收到，我先参考一下你给的图，拍{count}张给你。",
                        f"好，我照着你发的感觉来，给你准备{count}张。",
                        f"嗯，我先看看参考图，拍一组{count}张给你。",
                    ])
                return random.choice([
                    f"好呀，那我去准备一组{count}张给你。",
                    f"等我一下，我去拍一套{count}张给你看。",
                    f"嗯嗯，我去整理一下，给你拍{count}张不同感觉的。",
                    f"好，给你拍一组{count}张，场景和角度我会尽量错开。",
                ])
            if _is_clothing:
                return random.choice([
                    "那我去翻下柜子，等我一下哦。",
                    "好，等我换件衣服先。",
                    "嗯嗯，我先去换个造型，稍等。",
                    "行吧，先让我翻翻衣柜。",
                    "稍等哦，换好了就来。",
                    "好好好，我去换一套。",
                    "等我一下，先去换个衣服。",
                    "行，我去试试看。",
                ])
            if has_user_images:
                return random.choice([
                    "嗯，我看看你给的图，稍等。",
                    "收到，我参考一下，等我哦。",
                    "好的好的，照着来，稍等。",
                    "嗯嗯，我看看，等一下。",
                ])
            # 有场景时，用场景化的自然回复
            if scene_name and scene_name not in ("默认", "默认(兜底)", ""):
                return random.choice([
                    f"好，等我去{scene_name}拍一张~",
                    f"嗯嗯，去{scene_name}拍一张，稍等哦。",
                    f"行，{scene_name}拍一张给你看。",
                    f"好嘞，等我一下~",
                ])
            return random.choice([
                "好，等我一下~",
                "嗯嗯，马上来。",
                "行，稍等哦。",
                "好嘞，等一下~",
                "嗯，等我准备一下。",
            ])
        elif action == "auto_text":
            return random.choice([
                "等一下，按你说的搞搞看。",
                "好，稍等。",
                "嗯嗯，我来弄一下。",
                "诶好，稍等。",
            ])
        elif action == "auto_image":
            return random.choice([
                "好，帮你调一下这张。",
                "稍等，照着这张弄弄看。",
                "等一下哦。",
                "嗯，看看。",
            ])
        elif action == "batch":
            if total_images > 0:
                return random.choice([
                    f"好，这{total_images}张我来搞，稍等。",
                    f"等一下，{total_images}张慢慢来。",
                    f"稍等，{total_images}张一起弄。",
                    f"嗯，{total_images}张，稍等哦。",
                ])
            return random.choice(["稍等，慢慢来。", "等一下哦。", "好，来了。"])
        elif action == "pack_pdf":
            return random.choice([
                "等一下，帮你整理一下。",
                "稍等，打包一下。",
                "好，马上好。",
                "嗯，整理中。",
            ])
        return "稍等。"

    def _build_llm_tool_failure(self, error: Any, default_msg: str = "这边刚才有点小问题，稍后我再试试。") -> str:
        """仅返回给 LLM 的失败信息，不直接向用户暴露底层报错。"""
        import random
        masked = self._mask_llm_error(error, default_msg)
        low_level_failures = {
            "API服务器内部错误", "API网关错误", "API服务暂时不可用", "网络连接异常",
            "连接被拒绝或重置，网络不稳定", "DNS解析失败，无法访问服务器",
            "SSL证书验证失败", "请求超时，API响应时间过长", "API接口不存在",
            "API密钥无效或已过期", "API访问被禁止",
        }
        if str(masked).strip() in low_level_failures:
            masked = default_msg
        masked = str(masked).removeprefix("❌").strip()
        # 给 LLM 多种自然表达失败的参考，不要让它照搬机械话术
        fail_styles = random.choice([
            "刚才有点小状况，稍后我再试一次。",
            "哎，刚刚有点问题，过一会儿再来更稳一点。",
            "应该是网络轻轻抖了一下，我等会儿再试。",
            "先缓一缓，待会儿我再继续。",
            "这边有点卡，稍后再接着处理。",
        ])
        return self._finalize_llm_tool_result(
            f"[TOOL_FAILED] 失败原因（不要对用户说）：{masked}。"
            f"请用你自己的语气告诉用户这次没弄好，参考语气：'{fail_styles}'，但不要照抄，自然表达即可。"
            f"绝对不要提'系统''工具''API''参数''配置'等词汇。"
        )

    def _finalize_llm_tool_result(self, text: str) -> str:
        """统一收敛 LLM 工具返回：屏蔽思考过程提示并附加硬性输出约束。"""
        result = str(text or "")

        result = re.sub(r"<\s*think\b[^>]*>.*?<\s*/\s*think\s*>", "", result, flags=re.IGNORECASE | re.DOTALL)
        result = re.sub(r"<\s*analysis\b[^>]*>.*?<\s*/\s*analysis\s*>", "", result, flags=re.IGNORECASE | re.DOTALL)

        leaked_markers = [
            "思考过程", "推理过程", "推理链", "分析过程", "chain of thought",
            "reasoning", "内部推理", "系统提示", "隐藏指令", "tool call", "工具调用"
        ]
        cleaned_lines = []
        for line in result.splitlines():
            lower_line = line.lower()
            if any(marker in line for marker in leaked_markers):
                continue
            if any(marker in lower_line for marker in ["chain of thought", "reasoning", "tool call"]):
                continue
            cleaned_lines.append(line)

        result = "\n".join(cleaned_lines).strip()
        if not result:
            result = "[TOOL_SUCCESS] 任务已完成，结果已发送给用户。"

        guard_rule = (
            "【对话要求】你现在要根据以上工具结果，用自己的语气给用户回一句自然的话，就像真人在聊天一样。"
            "严格禁止以下行为：\n"
            "- 说出'生成''绘制''绘图''渲染''处理完成''任务完成''已发送'等机械词汇\n"
            "- 暴露工具名称、预设名、内部参数或系统信息\n"
            "- 输出思考过程、推理步骤、错误代码或格式前缀（如[TOOL_SUCCESS]）\n"
            "- 重复复述工具返回的文字\n"
            "如果图片发送成功，就随口说一句自然的收尾话（或者不说也行）。"
            "如果失败，用生活化的语气随口带过，不要说'系统''工具''API'等字眼。"
        )
        if guard_rule not in result:
            result = f"{result}\n{guard_rule}"

        return result

    def _finalize_llm_tool_success(self, text: str = "") -> str:
        """成功结果交给二次 LLM 收尾；调用方负责确保图片已发送完成。"""
        return self._finalize_llm_tool_result(
            text or "[TOOL_SUCCESS] 任务已完成，结果已发送给用户。"
        )

    async def _prepare_send_image_bytes(self, image_bytes: bytes) -> bytes:
        """发送前保持原图，避免任何有损压缩或缩放。"""
        return image_bytes

    async def _get_active_session_task(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取当前会话中的进行中任务"""
        if not session_id:
            return None
        async with self._session_task_status_lock:
            task = self._session_task_status.get(session_id)
            if task and task.get("running"):
                return dict(task)
            return None

    async def _begin_session_task(self, session_id: str, task_type: str, total: int):
        """登记会话任务开始"""
        if not session_id:
            return
        async with self._session_task_status_lock:
            self._session_task_status[session_id] = {
                "running": True,
                "task_type": task_type,
                "total": max(0, int(total or 0)),
                "current": 0,
                "success": 0,
                "fail": 0,
                "started_at": datetime.now().isoformat(timespec="seconds"),
            }

    async def _update_session_task_progress(self, session_id: str, current: Optional[int] = None,
                                            success: Optional[int] = None, fail: Optional[int] = None):
        """更新会话任务进度"""
        if not session_id:
            return
        async with self._session_task_status_lock:
            task = self._session_task_status.get(session_id)
            if not task:
                return
            if current is not None:
                task["current"] = max(0, int(current))
            if success is not None:
                task["success"] = max(0, int(success))
            if fail is not None:
                task["fail"] = max(0, int(fail))

    async def _finish_session_task(self, session_id: str):
        """标记会话任务完成"""
        if not session_id:
            return
        async with self._session_task_status_lock:
            task = self._session_task_status.get(session_id)
            if task:
                task["running"] = False
                task["finished_at"] = datetime.now().isoformat(timespec="seconds")

    def _build_active_task_reply(self, active_task: Optional[Dict[str, Any]]) -> str:
        """构建进行中任务的统一回复，避免用户催促时重复开新任务"""
        if not active_task:
            return ""

        task_type = active_task.get("task_type", "批量任务")
        current = max(0, int(active_task.get("current", 0) or 0))
        total = max(0, int(active_task.get("total", 0) or 0))
        success = max(0, int(active_task.get("success", 0) or 0))
        fail = max(0, int(active_task.get("fail", 0) or 0))

        if total > 0:
            return (
                f"当前已有进行中的{task_type}，"
                f"进度 {current}/{total}，"
                f"成功 {success} 张，失败 {fail} 张。"
                f"这次先别重复开新任务，继续等我把这批处理完。"
            )

        return f"当前已有进行中的{task_type}，先别重复开新任务，继续等待即可。"

    async def _can_pack_pdf_now(self, event: AstrMessageEvent, gathered_images: Optional[List[bytes]] = None) -> Tuple[
        bool, str]:
        """校验当前是否满足打包 PDF 的前置条件：必须有成功生成结果或当前上下文存在明确有效图片"""
        session_id = event.unified_msg_origin
        sender_id = norm_id(event.get_sender_id())
        success_count = await self._get_generation_success_count(session_id)
        pending_count = await self._get_pending_generation_count(session_id)

        if gathered_images and any(isinstance(img, bytes) and len(img) > 0 for img in gathered_images):
            return True, ""

        # 如果当前会话里还有后台图片任务在跑，允许 PDF 工具继续进入收集/等待流程，
        # 避免“刚开始拍照就立刻打包”时被过早拦截，导致无法等待新图落库。
        if pending_count > 0:
            return True, ""

        if success_count > 0:
            return True, ""

        cached_images = await self._get_recent_generated_images(session_id)
        if cached_images:
            return True, ""

        msg_info = self._extract_message_info(event)
        current_urls = msg_info.get("image_urls", [])
        if current_urls:
            return True, ""

        image_sources = await self._collect_images_from_context(
            session_id,
            count=self._context_rounds,
            include_bot=True,
            sender_id=sender_id
        )
        if any(urls for _, urls in image_sources):
            return True, ""

        return False, "现在还没有图可以打包哦，先发点图或者让我帮你弄几张，然后再来打包~"

    async def _check_quota(self, event, uid, gid, cost) -> dict:
        res = {"allowed": False, "source": None, "msg": ""}

        # 1. 检查用户是否被黑名单
        if uid in (self.conf.get("user_blacklist") or []):
            res["msg"] = "这个功能你暂时用不了哦~"
            return res
        if gid and gid in (self.conf.get("group_blacklist") or []):
            res["msg"] = "这个群暂时用不了这个功能~"
            return res

        # 2. 管理员始终允许
        if self.is_admin(event):
            res["allowed"] = True
            res["source"] = "free"
            return res

        # 3. 检查用户白名单（如果配置了白名单，则只有白名单用户允许）
        user_whitelist = self.conf.get("user_whitelist") or []
        if user_whitelist and uid not in user_whitelist:
            res["msg"] = "你还没有使用权限，联系管理员开通一下吧~"
            return res

        # 4. 如果在用户白名单中，允许使用
        if user_whitelist and uid in user_whitelist:
            res["allowed"] = True
            res["source"] = "free"
            return res

        # 5. 检查群聊白名单（如果配置了群白名单，则只有白名单群允许）
        group_whitelist = self.conf.get("group_whitelist") or []
        if group_whitelist and gid and gid not in group_whitelist:
            res["msg"] = "这个群还没有使用权限，联系管理员开通一下吧~"
            return res

        # 6. 如果在群聊白名单中，允许使用
        if group_whitelist and gid and gid in group_whitelist:
            res["allowed"] = True
            res["source"] = "free"
            return res

        # 7. 检查次数限制
        enable_u = self.conf.get("enable_user_limit", True)
        enable_g = self.conf.get("enable_group_limit", False)
        if not enable_u and not enable_g:
            res["allowed"] = True;
            res["source"] = "free";
            return res

        u_bal = self.data_mgr.get_user_count(uid)
        if enable_u and u_bal >= cost:
            res["allowed"] = True;
            res["source"] = "user";
            return res
        if gid and enable_g:
            g_bal = self.data_mgr.get_group_count(gid)
            if g_bal >= cost:
                res["allowed"] = True;
                res["source"] = "group";
                return res
        else:
            g_bal = 0

        res["msg"] = f"次数不太够了（需要{cost}次，你还剩{u_bal}次，群里还剩{g_bal}次），联系管理员补充一下吧~"
        return res

    async def _load_preset_ref_images(self, preset_name: str) -> List[bytes]:
        """加载预设的参考图"""
        if not self.data_mgr.has_preset_ref_images(preset_name):
            return []
        return await self.data_mgr.load_preset_ref_images_bytes(preset_name)

    # ================= 核心：后台生成逻辑封装 =================

    async def _run_background_task(self, event: AstrMessageEvent, images: List[bytes],
                                   prompt: str, preset_name: str, deduction: dict, uid: str, gid: str, cost: int,
                                   extra_rules: str = "", model_override: str = "", hide_text: bool = False,
                                   charge_quota: bool = True,
                                   suppress_user_error: bool = False) -> Tuple[bool, str]:
        """
        后台执行生成任务，并在完成后主动发送消息。

        Args:
            extra_rules: 用户追加的规则（如"皮肤白一点"）
            model_override: 指定使用的模型（如果为空则使用默认模型）
            hide_text: 是否隐藏生成成功提示文字
        """
        try:
            # 1. 扣费
            if charge_quota:
                if deduction["source"] == "user":
                    await self.data_mgr.decrease_user_count(uid, cost)
                elif deduction["source"] == "group":
                    await self.data_mgr.decrease_group_count(gid, cost)

            # 2. 加载预设参考图（如果有）
            # 注意：人设功能（preset_name 以 "人设-" 开头）已经在调用前加载了参考图，不需要重复加载
            if preset_name != "自定义" and not preset_name.startswith("人设-") and self.conf.get(
                    "enable_preset_ref_images", True):
                ref_images = await self._load_preset_ref_images(preset_name)
                if ref_images:
                    # 将参考图添加到图片列表前面
                    images = ref_images + images
                    logger.info(f"已加载 {len(ref_images)} 张预设参考图: {preset_name}")

            # 3. 调用 API（使用指定模型或默认模型）
            model = model_override  # 为空时由 API manager 自动使用信息源的模型
            start_time = datetime.now()

            res = None
            for attempt in range(2):
                res = await self.api_mgr.call_api(
                    images, prompt, model, False, self.img_mgr.proxy
                )
                if isinstance(res, bytes) or not self._is_transient_generation_error(res) or attempt == 1:
                    break
                logger.warning(f"Background task transient error, retrying once: {res}")
                await asyncio.sleep(1.2)

            # 4. 处理结果
            if isinstance(res, bytes):
                res = await self._prepare_send_image_bytes(res)
                elapsed = (datetime.now() - start_time).total_seconds()
                await self.data_mgr.record_usage(uid, gid)
                await self._register_generation_success(event.unified_msg_origin, 1)
                await self._register_generated_image(event.unified_msg_origin, res)

                # 5. 检查是否处于 PDF 暂存模式
                if await self._maybe_stage_after_grace(event.unified_msg_origin, res):
                    return True, ""

                # 6. 主动发送结果
                chain_nodes = [Image.fromBytes(res)]
                if not hide_text:
                    quota_str = self._get_quota_str(deduction, uid, gid)
                    # 构建成功文案
                    timing_text = self._format_success_timing(elapsed)
                    info_text = f"\n✅ 生成成功 ({timing_text}) | 预设: {preset_name}"
                    if extra_rules:
                        info_text += f" | 规则: {extra_rules[:20]}{'...' if len(extra_rules) > 20 else ''}"
                    info_text += f" | 剩余: {quota_str}"
                    if self.conf.get("show_model_info", False):
                        info_text += f" | {model}"
                    chain_nodes.append(Plain(info_text))
                else:
                    chain_nodes.append(Plain(" "))  # 防止某些适配器丢弃纯图片消息

                chain = event.chain_result(chain_nodes)
                await event.send(chain)
                return True, ""
            else:
                error_msg = self._resolve_debug_error_message(res,
                                                              "这次没弄好，请稍后再试。") if suppress_user_error else str(
                    res)
                if (not suppress_user_error) or self._should_show_debug_errors():
                    display_msg = error_msg
                    if not str(display_msg).startswith("❌"):
                        display_msg = f"没搞好: {display_msg}"
                    else:
                        display_msg = str(display_msg).removeprefix("❌").strip()
                    await event.send(event.chain_result([Plain(display_msg)]))
                return False, str(error_msg).removeprefix("❌").strip()

        except Exception as e:
            logger.error(f"Background task error: {e}")
            error_msg = self._resolve_debug_error_message(e,
                                                          "这次没弄好，请稍后再试。") if suppress_user_error else f"系统错误: {e}"
            if (not suppress_user_error) or self._should_show_debug_errors():
                _display = str(error_msg).removeprefix("❌").strip()
                await event.send(event.chain_result([Plain(f"出了点状况: {_display}")]))
            return False, str(error_msg).removeprefix("❌").strip()
        finally:
            await self._complete_pending_generation(event.unified_msg_origin, 1)

    # ================= 批量文生图功能 =================

    async def _run_batch_text_to_image(self, event: AstrMessageEvent, prompt: str, preset_name: str,
                                       deduction: dict, uid: str, gid: str, count: int,
                                       extra_rules: str = "", hide_text: bool = False,
                                       suppress_user_error: bool = False) -> Dict[str, Any]:
        """
        批量文生图后台任务

        Args:
            event: 消息事件
            prompt: 提示词
            preset_name: 预设名
            deduction: 扣费信息
            uid: 用户ID
            gid: 群组ID
            count: 生成数量
            extra_rules: 追加规则
            hide_text: 是否隐藏提示文字
        """
        try:
            # 1. 统一扣费
            total_cost = count
            if deduction["source"] == "user":
                await self.data_mgr.decrease_user_count(uid, total_cost)
            elif deduction["source"] == "group":
                await self.data_mgr.decrease_group_count(gid, total_cost)

            # 2. 加载预设参考图（如果有）
            images = []
            if preset_name != "自定义" and self.conf.get("enable_preset_ref_images", True):
                ref_images = await self._load_preset_ref_images(preset_name)
                if ref_images:
                    images = ref_images
                    logger.info(f"已加载 {len(ref_images)} 张预设参考图: {preset_name}")

            # 3. 获取文生图模型
            model = ""

            concurrency = max(1, self.conf.get("batch_concurrency", 3))
            max_retries = self.conf.get("batch_retries", 2)

            semaphore = asyncio.Semaphore(concurrency)
            results = {"success": 0, "fail": 0, "errors": []}
            results_lock = asyncio.Lock()

            async def process_single(index: int):
                async with semaphore:
                    try:
                        retry_count = 0
                        success = False
                        error_msg = ""

                        while retry_count <= max_retries:
                            start_time = datetime.now()
                            res = await self.api_mgr.call_api(
                                images, prompt, model, False, self.img_mgr.proxy
                            )

                            if isinstance(res, bytes):
                                res = await self._prepare_send_image_bytes(res)
                                elapsed = (datetime.now() - start_time).total_seconds()
                                await self.data_mgr.record_usage(uid, gid)
                                await self._register_generation_success(event.unified_msg_origin, 1)
                                await self._register_generated_image(event.unified_msg_origin, res)

                                # PDF 暂存模式：不发送单张图片，用 index 保证顺序
                                if await self._maybe_stage_after_grace(event.unified_msg_origin, res, staging_index=index):
                                    success = True
                                    break

                                chain_nodes = [Image.fromBytes(res)]
                                if not hide_text:
                                    timing_text = self._format_success_timing(elapsed)
                                    info_text = f"\n✅ [{index}/{count}] 生成成功 ({timing_text}) | 预设: {preset_name}"
                                    if extra_rules:
                                        info_text += f" | 规则: {extra_rules[:15]}..."
                                    chain_nodes.append(Plain(info_text))

                                await event.send(event.chain_result(chain_nodes))
                                success = True
                                break
                            else:
                                error_msg = self._translate_error_to_chinese(res)

                            retry_count += 1
                            if retry_count <= max_retries:
                                if (not suppress_user_error) or self._should_show_debug_errors():
                                    await event.send(event.chain_result([
                                        Plain(f"⚠️ 第 {index}/{count} 张生成失败 ({error_msg})\n⏳ 正在重试...")
                                    ]))
                                await asyncio.sleep(1.5)

                        async with results_lock:
                            if success:
                                results["success"] += 1
                            else:
                                results["fail"] += 1
                                results["errors"].append(error_msg)
                                if (not suppress_user_error) or self._should_show_debug_errors():
                                    await event.send(event.chain_result([
                                        Plain(f"第{index}张没弄好: {error_msg}")
                                    ]))

                    except Exception as e:
                        error_msg = self._resolve_debug_error_message(e,
                                                                      "这次没弄好，请稍后再试。") if suppress_user_error else self._translate_error_to_chinese(
                            str(e))
                        logger.error(f"Batch text-to-image {index} exception: {e}", exc_info=True)
                        async with results_lock:
                            results["fail"] += 1
                            results["errors"].append(error_msg)
                        if (not suppress_user_error) or self._should_show_debug_errors():
                            await event.send(event.chain_result([
                                Plain(f"第{index}张出了点问题: {error_msg}")
                            ]))
                    finally:
                        await self._complete_pending_generation(event.unified_msg_origin, 1)

            # 4. 并发执行所有任务
            tasks = [process_single(i) for i in range(1, count + 1)]
            await asyncio.gather(*tasks)

            # 5. 发送完成汇总
            if not hide_text:
                quota_str = self._get_quota_str(deduction, uid, gid)
                summary = f"\n📊 批量生成完成: 成功 {results['success']}/{count} 张 | 剩余: {quota_str}"
                await event.send(event.chain_result([Plain(summary)]))
            return results

        except Exception as e:
            logger.error(f"Batch text-to-image task error: {e}")
            final_error = self._resolve_debug_error_message(e, "这次没弄好，请稍后再试。")
            if (not suppress_user_error) or self._should_show_debug_errors():
                await event.send(event.chain_result([Plain(f"这批出了点状况: {final_error}")]))
            return {"success": 0, "fail": count, "errors": [final_error]}

    # ================= 批量图生图功能（同一张图片生成多个版本） =================

    async def _run_batch_image_to_image(self, event: AstrMessageEvent, images: List[bytes],
                                        prompt: str, preset_name: str, deduction: dict,
                                        uid: str, gid: str, count: int,
                                        extra_rules: str = "", hide_text: bool = False,
                                        charge_quota: bool = True,
                                        suppress_user_error: bool = False) -> Dict[str, Any]:
        """
        批量图生图后台任务 - 对同一张图片生成多个不同版本

        Args:
            event: 消息事件
            images: 输入图片列表
            prompt: 提示词
            preset_name: 预设名
            deduction: 扣费信息
            uid: 用户ID
            gid: 群组ID
            count: 生成数量
            extra_rules: 追加规则
            hide_text: 是否隐藏提示文字
        """
        try:
            # 1. 统一扣费
            total_cost = count
            if charge_quota:
                if deduction["source"] == "user":
                    await self.data_mgr.decrease_user_count(uid, total_cost)
                elif deduction["source"] == "group":
                    await self.data_mgr.decrease_group_count(gid, total_cost)

            # 2. 加载预设参考图（如果有）
            if preset_name != "自定义" and preset_name != "编辑" and self.conf.get("enable_preset_ref_images", True):
                ref_images = await self._load_preset_ref_images(preset_name)
                if ref_images:
                    # 将参考图添加到图片列表前面
                    images = ref_images + images
                    logger.info(f"已加载 {len(ref_images)} 张预设参考图: {preset_name}")

            # 3. 获取模型
            model = ""

            concurrency = max(1, self.conf.get("batch_concurrency", 3))
            max_retries = self.conf.get("batch_retries", 2)

            semaphore = asyncio.Semaphore(concurrency)
            results = {"success": 0, "fail": 0, "errors": []}
            results_lock = asyncio.Lock()

            # 4. 并发逐张生成（每次调用API都会产生不同的结果）
            async def process_single(index: int):
                async with semaphore:
                    try:
                        retry_count = 0
                        success = False
                        error_msg = ""

                        while retry_count <= max_retries:
                            start_time = datetime.now()
                            res = await self.api_mgr.call_api(images, prompt, model, False, self.img_mgr.proxy)

                            if isinstance(res, bytes):
                                res = await self._prepare_send_image_bytes(res)
                                elapsed = (datetime.now() - start_time).total_seconds()
                                await self.data_mgr.record_usage(uid, gid)
                                await self._register_generation_success(event.unified_msg_origin, 1)
                                await self._register_generated_image(event.unified_msg_origin, res)

                                # PDF 暂存模式：不发送单张图片，用 index 保证顺序
                                if await self._maybe_stage_after_grace(event.unified_msg_origin, res, staging_index=index):
                                    success = True
                                    break

                                chain_nodes = [Image.fromBytes(res)]
                                if not hide_text:
                                    timing_text = self._format_success_timing(elapsed)
                                    info_text = f"\n✅ [{index}/{count}] 版本生成成功 ({timing_text}) | 预设: {preset_name}"
                                    if extra_rules:
                                        info_text += f" | 规则: {extra_rules[:15]}..."
                                    chain_nodes.append(Plain(info_text))

                                await event.send(event.chain_result(chain_nodes))
                                success = True
                                break
                            else:
                                error_msg = self._translate_error_to_chinese(res)

                            retry_count += 1
                            if retry_count <= max_retries:
                                if (not suppress_user_error) or self._should_show_debug_errors():
                                    await event.send(event.chain_result([
                                        Plain(f"⚠️ 第 {index}/{count} 个版本生成失败 ({error_msg})\n⏳ 正在重试...")
                                    ]))
                                await asyncio.sleep(1.5)

                        async with results_lock:
                            if success:
                                results["success"] += 1
                            else:
                                results["fail"] += 1
                                results["errors"].append(error_msg)
                                if (not suppress_user_error) or self._should_show_debug_errors():
                                    await event.send(event.chain_result([
                                        Plain(f"第{index}个版本没弄好: {error_msg}")
                                    ]))

                    except Exception as e:
                        error_msg = self._resolve_debug_error_message(e,
                                                                      "这次没弄好，请稍后再试。") if suppress_user_error else self._translate_error_to_chinese(
                            str(e))
                        logger.error(f"Batch image-to-image {index} exception: {e}", exc_info=True)
                        async with results_lock:
                            results["fail"] += 1
                            results["errors"].append(error_msg)
                        if (not suppress_user_error) or self._should_show_debug_errors():
                            await event.send(event.chain_result([
                                Plain(f"第{index}个版本出了点问题: {error_msg}")
                            ]))
                    finally:
                        await self._complete_pending_generation(event.unified_msg_origin, 1)

            tasks = [process_single(i) for i in range(1, count + 1)]
            await asyncio.gather(*tasks)

            # 5. 发送完成汇总
            if not hide_text:
                quota_str = self._get_quota_str(deduction, uid, gid)
                summary = f"\n📊 多版本生成完成: 成功 {results['success']}/{count} 张 | 剩余: {quota_str}"
                await event.send(event.chain_result([Plain(summary)]))
            return results

        except Exception as e:
            logger.error(f"Batch image-to-image task error: {e}")
            final_error = self._resolve_debug_error_message(e, "这次没弄好，请稍后再试。")
            if (not suppress_user_error) or self._should_show_debug_errors():
                await event.send(event.chain_result([Plain(f"多版本处理出了点状况: {final_error}")]))
            return {"success": 0, "fail": count, "errors": [final_error]}

    # ================= LLM 工具调用 (Tool Calling) =================

    @filter.llm_tool(name="shoubanhua_draw_image")
    async def text_to_image_tool(self, event: AstrMessageEvent, prompt: str, count: int = 1):
        '''根据文本描述生成图片（文生图）。

        【排他性条件（极度重要！）】
        - 如果用户是向你要**你自己的 / Bot本人 / 人设角色**的照片、自拍、写真、露脸、外貌、穿搭、换装、cos、动作展示、与上图合影或同框，【严禁】调用此工具！必须使用 shoubanhua_persona_photo 工具！
        - 如果用户说“修改上面那张 / 改刚才那张 / 继续改 / 在这个基础上改”等依赖上一张图片的请求，【严禁】调用此工具！必须使用 shoubanhua_edit_image，用户不需要重新引用图片。
        - 本工具仅用于生成除你自己以外的其他角色或事物的图片！

        调用前请判断用户是否明确要求生成图片。如果用户只是闲聊则不要调用。

        【批量生成数量控制】
        - 除非用户明确说出了具体数字（如"画5张"），否则【严禁】随意设置大量 count。
        - 用户说"打包成PDF"、"发个PDF"并不等于要多张！只说打包不说数量时 count 必须保持 1。
        - 如果用户只说"多来点"、"多来几张"、"写真集"，请默认设置 count=3。
        - 坚决不要为了表现热情而擅自将 count 设为 10 等大数字，这会严重消耗资源！

        【个性化回复】
        调用此工具后，直接生成图片即可。

        Args:
            prompt(string): 图片生成的提示词，可以是预设名+追加规则。
            count(int): 生成图片的数量，默认1张，最大10张。当用户要求"多来点"时设置为3张。
        '''
        # 0. 检查 LLM 工具开关
        if not self._get_conf_bool("enable_llm_tool", True):
            return self._finalize_llm_tool_result(
                "[TOOL_UNAVAILABLE] 当前功能暂时不可用。请用自己平时的语气告诉用户现在不方便，别提功能禁用、指令模式等词汇，要自然。")

        persona_request_text = " ".join([str(getattr(event, "message_str", "") or ""), str(prompt or "")])
        if self._persona_mode and self._looks_like_persona_photo_request(persona_request_text):
            logger.info("shoubanhua_draw_image 命中人设请求兜底分流，改用 shoubanhua_persona_photo")
            inferred_count = self._infer_requested_count_from_text(persona_request_text, default=1, multi_default=3)
            return await self.persona_photo_tool(
                event=event,
                scene_hint="",
                extra_request=persona_request_text,
                count=inferred_count,
            )

        # 0.1 检查图片生成冷却时间
        uid = norm_id(event.get_sender_id())
        in_cooldown, remaining = self._check_image_cooldown(uid)
        if in_cooldown:
            # 返回借口让LLM用自然语言拒绝
            excuse = self._get_cooldown_excuse(remaining)
            return f"【冷却中】{excuse}\n\n请用自然的方式告诉用户现在不方便生成图片，可以稍后再试。不要直接说'冷却'这个词。"

        # 0.2 限制批量生成数量
        requested_count = self._resolve_tool_requested_count(prompt, incoming_count=count, default=1, multi_default=3)
        raw_count = requested_count
        count, count_limited = self._normalize_generation_count(requested_count, "draw")

        # 1. 计算预设和追加规则
        final_prompt, preset_name, extra_rules = self._process_prompt_and_preset(prompt)

        # 根据配置决定是否隐藏进度提示（白名单用户和普通用户使用同一开关）
        show_llm_progress = self._get_conf_bool("llm_show_progress", True)
        hide_llm_result_text = True

        # 2. 根据配置决定是否发送进度提示
        if show_llm_progress:
            feedback = self._build_llm_progress_text(
                "draw", preset_name=preset_name, count=count, extra_request=extra_rules
            )
            await event.send(event.chain_result([Plain(feedback)]))

        # 3. 检查配额（批量生成需要足够的次数）
        gid = norm_id(event.get_group_id())
        total_cost = count
        deduction = await self._check_quota(event, uid, gid, total_cost)
        if not deduction["allowed"]:
            return self._finalize_llm_tool_result(
                "[TOOL_FAILED] 用户次数不足，无法完成。"
                "请用你自己平时的语气告诉用户现在搞不了，"
                "别提'次数''配额''限制'等词汇，随口带过就好，可以让TA联系管理员或等等再来。"
            )

        # 4. 更新图片生成冷却时间
        self._update_image_cooldown(uid)

        # 4.1 新任务开始前，若当前会话空闲则清掉上一轮生成缓存，避免后续 PDF 混入旧图
        await self._clear_session_image_cache_if_idle(event.unified_msg_origin)

        # 5. 等待生成任务完成并确认图片已发送，再把成功结果交给二次 LLM 收尾。
        await self._register_pending_generation(event.unified_msg_origin, count)
        if count == 1:
            success, error_msg = await self._run_background_task(
                event, [], final_prompt, preset_name, deduction, uid, gid, total_cost,
                extra_rules,
                model_override="", hide_text=hide_llm_result_text,
                suppress_user_error=True
            )
            total_success = 1 if success else 0
            total_fail = 0 if success else 1
            branch_errors = [error_msg] if error_msg else []
        else:
            batch_result = await self._run_batch_text_to_image(
                event, final_prompt, preset_name, deduction, uid, gid, count, extra_rules,
                hide_llm_result_text, suppress_user_error=True
            )
            total_success = int(batch_result.get("success", 0))
            total_fail = int(batch_result.get("fail", 0))
            branch_errors = batch_result.get("errors", [])

        if total_success <= 0:
            return self._build_llm_tool_failure(branch_errors[0] if branch_errors else "这次没弄好，请稍后再试。")

        # 6. 图片已经发出，现在可以交给二次 LLM 说收尾话。
        count_limit_reply = self._build_count_limit_reply(count, "draw") if count_limited else ""

        if count_limit_reply:
            return self._finalize_llm_tool_success(
                f"[TOOL_SUCCESS] 图片已经发给用户。{count_limit_reply} 请用自然语气随口收尾一句。")
        if total_fail > 0:
            return self._finalize_llm_tool_success(
                f"[TOOL_SUCCESS] 图片已经发给用户，成功 {total_success} 张，有 {total_fail} 张没弄好。请自然地随口带过。")
        return self._finalize_llm_tool_success(
            "[TOOL_SUCCESS] 图片已经发给用户。请用自己的语气自然收尾一句，比如“给你啦，别说我没帮你。”不要提系统、工具或生成。")

    @filter.llm_tool(name="shoubanhua_edit_image")
    async def image_edit_tool(self, event: AstrMessageEvent, prompt: str, use_message_images: bool = True,
                              task_types: str = "edit", count: int = 1, merge_multiple_images: bool = False):
        '''编辑用户发送的图片或引用的图片（图生图）。仅在用户明确要求对图片进行处理时才调用。

        调用前请判断：
        1. 用户是否明确要求处理/编辑/转换图片？
        2. 用户是否发送了图片或引用了包含图片的消息？
        3. 请求是否具体且合理？

        如果用户只是发送图片但没有明确要求处理，或者只是闲聊，请不要调用此工具。
        如果用户要求“你/Bot本人/人设角色”和图片里的人合影、同框，或要求你参考上图换装/摆姿势/拍自拍，请改用 shoubanhua_persona_photo，不要用本工具。
        如果用户明确要求“修改上面那张 / 改刚才生成的图 / 继续改 / 在这个基础上改”，即使当前消息没有图片也要调用本工具；系统会自动使用最近的用户图片或Bot生成图片作为输入，不需要让用户引用。

        【多图处理规则】当用户提供/引用了多张图片时：
        - 默认情况 (merge_multiple_images=false)：会将这多张图片拆开，【分别、独立地】生成每一张图片。适用于用户一次性发多张图片想分别转化的场景。
        - 只有当用户明确要求"把这几张图融合"、"参考第一张修改第二张"等合并需求时：将 merge_multiple_images 设置为 true。这会将多图作为一个整体发给模型。

        【批量生成不同版本的数量控制（极度重要）】
        - 除非用户明确指定了数量（如"画5张"），否则【严禁】随意设置大量 count。
        - 用户说"打包成PDF"、"发个PDF"并不等于要多张！只说打包不说数量时 count 必须保持 1。
        - 如果用户只说"多来点"、"多来几张"，请默认设置 count=3。
        - 不要为了表现热情而擅自将 count 设为 10 等大数字！

        【重要】task_types 参数选择规则（请严格遵守）：
        1. task_types="edit"（默认）：按用户要求编辑/处理图片，【不使用任何预设】。prompt 中【不要】包含"手办化"等预设名称！
        2. task_types="figurine"：将图片转换为手办/模型风格。

        Args:
            prompt(string): 图片编辑提示词。task_types="edit"时只描述编辑要求，不要加预设名；task_types="figurine"时可以是预设名+追加规则
            use_message_images(boolean): 默认 true
            task_types(string): 任务类型，"edit"=编辑模式，"figurine"=手办化
            count(int): 生成不同版本的数量，默认1张，最大10张。当用户未指定时严禁使用过大数字。
            merge_multiple_images(boolean): 如果有多张图片，是否合并为同一请求（默认为False即分别处理）。
        '''
        # 0. 检查 LLM 工具开关
        if not self._get_conf_bool("enable_llm_tool", True):
            return self._finalize_llm_tool_result(
                "[TOOL_UNAVAILABLE] 当前功能暂时不可用。请用自己平时的语气告诉用户现在不方便，别提功能禁用、指令模式等词汇，要自然。")

        persona_request_text = " ".join([str(getattr(event, "message_str", "") or ""), str(prompt or "")])
        if self._persona_mode and self._looks_like_persona_photo_request(persona_request_text):
            logger.info("shoubanhua_edit_image 命中人设请求兜底分流，改用 shoubanhua_persona_photo")
            inferred_count = self._infer_requested_count_from_text(persona_request_text, default=1, multi_default=3)
            return await self.persona_photo_tool(
                event=event,
                scene_hint="",
                extra_request=persona_request_text,
                count=inferred_count,
            )

        # 0.1 检查图片生成冷却时间
        uid = norm_id(event.get_sender_id())
        in_cooldown, remaining = self._check_image_cooldown(uid)
        if in_cooldown:
            # 返回借口让LLM用自然语言拒绝
            excuse = self._get_cooldown_excuse(remaining)
            return f"【冷却中】{excuse}\n\n请用自然的方式告诉用户现在不方便处理图片，可以稍后再试。不要直接说'冷却'这个词。"

        # 1. 根据 task_types 决定是否使用预设
        # 当 task_types 为 "edit" 时，不匹配预设，直接使用用户的 prompt
        if task_types.lower() == "edit":
            # 编辑模式：不使用预设，直接使用用户的编辑指令
            processed_prompt = prompt
            preset_name = "编辑"
            extra_rules = ""
            final_prompt = f"Edit the image according to the following instructions: {processed_prompt}"
        else:
            # 手办化或其他预设模式：尝试匹配预设
            processed_prompt, preset_name, extra_rules = self._process_prompt_and_preset(prompt)
            if preset_name == "自定义":
                # 没有匹配到任何预设，回退到编辑模式，不使用预设
                preset_name = "编辑"
                extra_rules = ""
                final_prompt = f"Edit the image according to the following instructions: {prompt}"
            else:
                # 匹配到预设，使用预设内容
                final_prompt = processed_prompt

        # 根据配置决定是否隐藏进度提示（白名单用户和普通用户使用同一开关）
        show_llm_progress = self._get_conf_bool("llm_show_progress", True)
        hide_llm_result_text = True

        # 3. 提取图片。当前消息没有图时，自动回溯用户/机器人最近图片上下文。
        images = await self._resolve_contextual_image_inputs(
            event,
            prompt=prompt,
            merge_multiple_images=merge_multiple_images,
            include_current=use_message_images,
            allow_context_fallback=True,
        )

        if not images:
            # 不要重复发送错误消息，只返回给 LLM
            return self._finalize_llm_tool_result(
                "[TOOL_FAILED] 未检测到图片。请让用户发送或引用包含图片的消息后再试。【重要】不要再次调用此工具，直接用自然语言告诉用户需要提供图片。")

        # 4. 限制批量生成数量
        count_request_text = f"{prompt} {extra_rules}".strip() if extra_rules else prompt
        raw_count = self._resolve_tool_requested_count(count_request_text, incoming_count=count, default=1, multi_default=3)
        count, count_limited = self._normalize_generation_count(raw_count, "edit")
        gid = norm_id(event.get_group_id())

        # ==== 分支：分别批量处理多张图片 ====
        if len(images) > 1 and not merge_multiple_images:
            total_images = len(images)
            total_cost = total_images * count
            deduction = await self._check_quota(event, uid, gid, total_cost)
            if not deduction["allowed"]:
                return self._finalize_llm_tool_result(
                    "[TOOL_FAILED] 用户次数不足，无法完成。"
                    "请用你自己平时的语气告诉用户现在搞不了，"
                    "别提'次数''配额''限制'等词汇，随口带过就好，可以让TA联系管理员或等等再来。"
                )

            self._update_image_cooldown(uid)

            # 发送进度提示
            if show_llm_progress:
                feedback = self._build_llm_progress_text(
                    "edit", preset_name=preset_name, count=count, total_images=total_images,
                    extra_request=extra_rules or prompt
                )
                await event.send(event.chain_result([Plain(feedback)]))

            # 多张图片分别处理时也走并发，避免 LLM 识别到多图后仍然一张张串行跑
            await self._register_pending_generation(event.unified_msg_origin, total_images * count)
            semaphore = asyncio.Semaphore(max(1, self.conf.get("batch_concurrency", 3)))

            async def process_single_source(img: bytes):
                async with semaphore:
                    if count == 1:
                        return await self._run_background_task(
                            event, [img], final_prompt, preset_name, deduction, uid, gid, count,
                            extra_rules, hide_text=hide_llm_result_text, charge_quota=False,
                            suppress_user_error=True
                        )
                    else:
                        return await self._run_batch_image_to_image(
                            event, [img], final_prompt, preset_name, deduction, uid, gid,
                            count, extra_rules, hide_llm_result_text, charge_quota=False,
                            suppress_user_error=True
                        )

            branch_results = await asyncio.gather(*(process_single_source(img) for img in images))
            total_success = 0
            total_fail = 0
            branch_errors = []
            for item in branch_results:
                if isinstance(item, tuple):
                    if item[0]:
                        total_success += 1
                    else:
                        total_fail += 1
                        branch_errors.append(item[1])
                elif isinstance(item, dict):
                    total_success += int(item.get("success", 0))
                    total_fail += int(item.get("fail", 0))
                    branch_errors.extend(item.get("errors", []))

            if total_success <= 0:
                return self._build_llm_tool_failure(branch_errors[0] if branch_errors else "这次没弄好，请稍后再试。")

            if total_fail > 0:
                return self._finalize_llm_tool_success(
                    f"[TOOL_SUCCESS] 发出去了{total_success}张，有{total_fail}张没弄好。用你自己的语气随口带过。")
            return self._finalize_llm_tool_success(
                f"[TOOL_SUCCESS] {total_images}张图都弄好发出去了。用你自己的语气随口带过，也可以什么都不说。")

        # ==== 分支：普通单次处理（单图或合并多图） ====
        total_cost = count
        deduction = await self._check_quota(event, uid, gid, total_cost)
        if not deduction["allowed"]:
            return self._finalize_llm_tool_result(
                "[TOOL_FAILED] 用户次数不足，无法完成。"
                "请用你自己平时的语气告诉用户现在搞不了，"
                "别提'次数''配额''限制'等词汇，随口带过就好，可以让TA联系管理员或等等再来。"
            )

        # 2. 根据配置决定是否发送进度提示
        if show_llm_progress:
            feedback = self._build_llm_progress_text(
                "edit", preset_name=preset_name, count=count, extra_request=extra_rules or prompt
            )
            await event.send(event.chain_result([Plain(feedback)]))

        # 6. 更新图片生成冷却时间
        self._update_image_cooldown(uid)

        # 6.1 新任务开始前，若当前会话空闲则清掉上一轮生成缓存，避免后续 PDF 混入旧图
        await self._clear_session_image_cache_if_idle(event.unified_msg_origin)

        # 7. 等待生成任务完成并确认图片已发送，再把成功结果交给二次 LLM 收尾。
        await self._register_pending_generation(event.unified_msg_origin, count)
        if count == 1:
            success, error_msg = await self._run_background_task(
                event, images, final_prompt, preset_name, deduction, uid, gid, total_cost,
                extra_rules, hide_text=hide_llm_result_text, suppress_user_error=True
            )
            total_success = 1 if success else 0
            total_fail = 0 if success else 1
            branch_errors = [error_msg] if error_msg else []
        else:
            batch_result = await self._run_batch_image_to_image(
                event, images, final_prompt, preset_name, deduction, uid, gid, count,
                extra_rules, hide_llm_result_text, suppress_user_error=True
            )
            total_success = int(batch_result.get("success", 0))
            total_fail = int(batch_result.get("fail", 0))
            branch_errors = batch_result.get("errors", [])

        if total_success <= 0:
            return self._build_llm_tool_failure(branch_errors[0] if branch_errors else "这次没弄好，请稍后再试。")

        # 图片已经发出，现在可以交给二次 LLM 说收尾话。
        count_limit_reply = self._build_count_limit_reply(count, "edit") if count_limited else ""

        if count_limit_reply:
            return self._finalize_llm_tool_success(
                f"[TOOL_SUCCESS] 图片已经处理好并发给用户。{count_limit_reply} 请自然地随口收尾一句。")
        if total_fail > 0:
            return self._finalize_llm_tool_success(
                f"[TOOL_SUCCESS] 图片已经发给用户，成功 {total_success} 张，有 {total_fail} 张没弄好。请自然地随口带过。")
        return self._finalize_llm_tool_success(
            "[TOOL_SUCCESS] 图片已经发给用户。请用自己的语气自然收尾一句，比如“给你啦，别说我没帮你。”不要提系统、工具或生成。")

    # ================= 传统指令触发 =================

    @filter.event_message_type(filter.EventMessageType.ALL, priority=5)
    async def on_figurine_request(self, event: AstrMessageEvent, ctx=None):
        if self.conf.get("prefix", True) and not event.is_at_or_wake_command:
            return

        text = event.message_str.strip()
        if not text: return

        # 消息去重检查：防止多平台重复处理同一消息
        msg_id = str(event.message_obj.message_id)
        dedup_key = f"figurine_{msg_id}"
        if self._is_message_processed(dedup_key):
            logger.debug(f"FigurinePro: 消息 {msg_id} 已被处理，跳过重复执行")
            return

        parts = text.split(maxsplit=1)
        cmd_raw = parts[0]
        base_cmd = cmd_raw

        user_prompt = ""
        preset_name = "自定义"

        extra_prefix = self.conf.get("extra_prefix", "bnn")
        is_bnn = (base_cmd == extra_prefix)

        if is_bnn:
            user_prompt = parts[1] if len(parts) > 1 else ""

            # [修改] bnn 模式下不再自动匹配预设，改为纯自定义模式
            # user_prompt, preset_name = self._process_prompt_and_preset(user_prompt)
            preset_name = "自定义"

        else:
            preset_prompt = self.data_mgr.get_prompt(base_cmd)

            if not preset_prompt: return

            user_prompt = preset_prompt
            preset_name = base_cmd

            if "%" in base_cmd: user_prompt += base_cmd.split("%", 1)[1]
            if len(parts) > 1:
                user_prompt += " " + parts[1]

            user_prompt = self._append_preset_safety_suffix(user_prompt, preset_name)

        uid = norm_id(event.get_sender_id())
        gid = norm_id(event.get_group_id())
        cost = 1

        deduction = await self._check_quota(event, uid, gid, cost)
        if deduction["allowed"] is False:
            yield event.chain_result([Plain(deduction["msg"])])
            event.stop_event()
            return

        # 立即阻止事件继续传递，防止重复触发
        event.stop_event()

        # 指令模式：先启动本处理流程，再发提示，避免被 after_message_sent 类插件截断后半段逻辑。
        _internal = {"自定义", "编辑", "edit", "custom"}
        preset_display = "" if (not preset_name or preset_name.strip().lower() in _internal) else preset_name
        template = self.conf.get("generating_msg_template", "🎨 收到请求，正在生成 [{preset}]...")
        feedback = template.replace("{preset}", preset_display) if preset_display else template.replace(" [{preset}]",
                                                                                                        "").replace(
            "[{preset}]", "")
        await event.send(event.chain_result([Plain(feedback)]))

        bot_id = self._get_bot_id(event)
        # 传递 bot_id 给 image manager 以过滤，并传入 context 支持 message_id
        images = await self.img_mgr.extract_images_from_event(event, ignore_id=bot_id, context=self.context)
        if images:
            await self._remember_session_image_context(
                event.unified_msg_origin,
                image_bytes_list=images,
                source="user_current"
            )

        if not is_bnn and user_prompt:
            urls = extract_image_urls_from_text(user_prompt)
            for u in urls:
                if b := await self.img_mgr.load_bytes(u):
                    images.append(b)
                    await self._remember_session_image_context(
                        event.unified_msg_origin,
                        image_bytes_list=[b],
                        image_urls=[u],
                        source="user_prompt_url"
                    )

        wants_context_image = bool(user_prompt and self._looks_like_context_image_reference(user_prompt))
        if not images and wants_context_image:
            images = await self._resolve_contextual_image_inputs(
                event,
                prompt=user_prompt,
                merge_multiple_images=False,
                include_current=False,
                allow_context_fallback=False,
            )

        if not images and wants_context_image:
            yield event.chain_result([Plain("没找到可继续修改的上文图片，先发图或生成一张再说。")])
            return

        if not images and not (is_bnn and user_prompt):
            yield event.chain_result([Plain("请发送图片或提供描述。")])
            return

        # 判断是否为纯文生图模式（bnn 指令且没有图片）
        if deduction["source"] == "user":
            await self.data_mgr.decrease_user_count(uid, cost)
        elif deduction["source"] == "group":
            await self.data_mgr.decrease_group_count(gid, cost)

        self._log_prompt_preview(
            f"command:{base_cmd if base_cmd else 'bnn'}",
            user_prompt
        )

        start = datetime.now()
        res = await self.api_mgr.call_api(
            images, user_prompt, "", proxy=self.img_mgr.proxy
        )

        if isinstance(res, bytes):
            res = await self._prepare_send_image_bytes(res)
            elapsed = (datetime.now() - start).total_seconds()
            await self.data_mgr.record_usage(uid, gid)
            await self._register_generation_success(event.unified_msg_origin, 1)
            await self._register_generated_image(event.unified_msg_origin, res)
            if not is_bnn: await self.data_mgr.save_preset_image(base_cmd, res)

            quota_str = self._get_quota_str(deduction, uid, gid)
            timing_text = self._format_success_timing(elapsed)
            info = f"\n✅ 生成成功 ({timing_text}) | 预设: {preset_name} | 剩余: {quota_str}"
            if self.conf.get("show_model_info", False):
                info += f" | 模型: {self.api_mgr.get_last_metrics().get('model', '默认')}"

            yield event.chain_result([Image.fromBytes(res), Plain(info)])
        else:
            yield event.chain_result([Plain(f"没弄好: {self._resolve_debug_error_message(res, '这次没弄好，请稍后再试。')}")])

    @filter.command("文生图", prefix_optional=True)
    async def on_txt2img(self, event: AstrMessageEvent, ctx=None):
        raw = event.message_str.strip()
        cmd_name = "文生图"
        prompt = raw.replace(cmd_name, "").strip()
        if not prompt: yield event.chain_result([Plain("请输入描述。")]); return

        uid = norm_id(event.get_sender_id())
        deduction = await self._check_quota(event, uid, event.get_group_id(), 1)
        if not deduction["allowed"]: yield event.chain_result([Plain(deduction["msg"])]); return

        final_prompt, preset_name, extra_rules = self._process_prompt_and_preset(prompt)

        _internal2 = {"自定义", "编辑", "edit", "custom"}
        preset_display = "" if (not preset_name or preset_name.strip().lower() in _internal2) else preset_name
        template = self.conf.get("generating_msg_template", "🎨 收到请求，正在生成 [{preset}]...")
        feedback = template.replace("{preset}", preset_display) if preset_display else template.replace(" [{preset}]",
                                                                                                        "").replace(
            "[{preset}]", "")
        await event.send(event.chain_result([Plain(feedback)]))

        if deduction["source"] == "user":
            await self.data_mgr.decrease_user_count(uid, 1)
        elif deduction["source"] == "group":
            await self.data_mgr.decrease_group_count(event.get_group_id(), 1)

        # 加载预设参考图
        images = []
        if preset_name != "自定义" and self.conf.get("enable_preset_ref_images", True):
            ref_images = await self._load_preset_ref_images(preset_name)
            if ref_images:
                images = ref_images
                logger.info(f"已加载 {len(ref_images)} 张预设参考图: {preset_name}")

        # 文生图使用专用模型
        model = ""
        self._log_prompt_preview("command:文生图", final_prompt)
        start = datetime.now()
        res = await self.api_mgr.call_api(
            images, final_prompt, model, False, self.img_mgr.proxy
        )

        if isinstance(res, bytes):
            res = await self._prepare_send_image_bytes(res)
            elapsed = (datetime.now() - start).total_seconds()
            await self.data_mgr.record_usage(uid, norm_id(event.get_group_id()))
            await self._register_generation_success(event.unified_msg_origin, 1)
            await self._register_generated_image(event.unified_msg_origin, res)
            quota_str = self._get_quota_str(deduction, uid, norm_id(event.get_group_id()))
            timing_text = self._format_success_timing(elapsed)
            info = f"\n✅ 生成成功 ({timing_text}) | 预设: {preset_name}"
            if extra_rules:
                info += f" | 规则: {extra_rules[:15]}..."
            info += f" | 剩余: {quota_str}"
            yield event.chain_result([Image.fromBytes(res), Plain(info)])
        else:
            yield event.chain_result([Plain(str(res))])

    # 辅助方法
    def _get_help_node(self, event):
        txt = self.conf.get("help_text", "帮助文档未配置")
        bot_id = self._get_bot_id(event) or "2854196310"
        return event.chain_result([Nodes(nodes=[Node(name="手办化助手", uin=bot_id, content=[Plain(txt)])])])

    # 省略 Admin指令，它们和上一版完全一致，请确保不要覆盖掉下面的代码（lm列表, lm添加, 增加次数等）

    @filter.command("lm列表", aliases={"lmlist"}, prefix_optional=True)
    async def on_preset_list(self, event: AstrMessageEvent, ctx=None):
        presets = []
        for k, v in self.data_mgr.prompt_map.items():
            presets.append((k, v == "[内置预设]"))
        presets.sort(key=lambda x: x[0])
        if not presets: yield event.chain_result([Plain("暂无预设")]); return
        img_data = await self.img_mgr.create_preset_table(presets, self.data_mgr)
        yield event.chain_result([Image.fromBytes(img_data)])

    @filter.command("lm添加", aliases={"lma"}, prefix_optional=True)
    async def on_add_preset(self, event: AstrMessageEvent, ctx=None):
        if not self.is_admin(event): return
        msg = event.message_str.replace("lm添加", "").replace("lma", "").strip()
        if ":" not in msg: yield event.chain_result([Plain("格式: 词:提示词")]); return

        k, v = msg.split(":", 1)
        k, v = k.strip(), v.strip()

        if not k or not v:
            yield event.chain_result([Plain("触发词和提示词都不能为空哦")])
            return

        # 使用 DataManager 进行持久化保存
        await self.data_mgr.add_user_prompt(k, v)

        # 同时更新到配置文件的 prompt_list 中，确保双重持久化
        prompt_list = self.conf.get("prompt_list", [])
        # 移除已存在的同名预设
        prompt_list = [item for item in prompt_list if not item.startswith(f"{k}:")]
        # 添加新预设
        prompt_list.append(f"{k}:{v}")
        self.conf["prompt_list"] = prompt_list
        self._save_config(["prompt_list"])

        yield event.chain_result([Plain(f"✅ 已添加预设: {k}\n💾 已同步保存到配置文件")])

    @filter.command("lm删除", aliases={"lmd", "lm删", "删除预设"}, prefix_optional=True)
    async def on_delete_preset(self, event: AstrMessageEvent, ctx=None):
        if not self.is_admin(event): return

        raw = event.message_str.strip()
        name = raw
        for cmd in ["lm删除", "lmd", "lm删", "删除预设"]:
            if name.startswith(cmd):
                name = name[len(cmd):]
                break
        name = name.strip().lstrip(":：").strip()

        if not name:
            yield event.chain_result([Plain("用法: #lm删除 <预设名>\n例如: #lm删除 手办化新版")])
            return

        prompt_list = self.conf.get("prompt_list", [])
        if not isinstance(prompt_list, list):
            prompt_list = []

        def _prompt_list_key(item) -> str:
            text = str(item or "")
            if ":" in text:
                return text.split(":", 1)[0].strip()
            if "：" in text:
                return text.split("：", 1)[0].strip()
            return ""

        in_user_prompts = name in getattr(self.data_mgr, "user_prompts", {})
        in_prompt_list = any(_prompt_list_key(item) == name for item in prompt_list)
        exists = name in self.data_mgr.prompt_map

        if exists and not in_user_prompts and not in_prompt_list:
            yield event.chain_result([Plain(f"预设 [{name}] 是内置或配置文件固定预设，不能用 lm删除 删除。")])
            return

        if not exists and not in_user_prompts and not in_prompt_list:
            yield event.chain_result([Plain(f"没找到预设 [{name}]")])
            return

        removed_runtime = await self.data_mgr.remove_user_prompt(name)
        new_prompt_list = [item for item in prompt_list if _prompt_list_key(item) != name]
        removed_config = len(new_prompt_list) != len(prompt_list)
        self.conf["prompt_list"] = new_prompt_list

        # 删除这个预设绑定的参考图，避免列表里没有预设但参考图索引还残留。
        ref_count = await self.data_mgr.clear_preset_ref_images(name)

        self.data_mgr.reload_prompts()
        self._save_config(["prompt_list"])

        details = []
        if removed_runtime:
            details.append("运行时记录")
        if removed_config:
            details.append("配置项")
        if ref_count:
            details.append(f"{ref_count} 张参考图")
        detail_text = "\n已清理: " + "、".join(details) if details else ""
        yield event.chain_result([Plain(f"✅ 已删除预设: {name}{detail_text}")])

    @filter.command("lm查看", aliases={"lmv", "lm预览"}, prefix_optional=True)
    async def on_view_preset(self, event: AstrMessageEvent, ctx=None):
        parts = event.message_str.split()
        if len(parts) < 2: yield event.chain_result([Plain("用法: #lm查看 <关键词>")]); return
        kw = parts[1].strip()
        prompt = self.data_mgr.get_prompt(kw)
        msg = f"🔍 [{kw}]:\n{prompt}" if prompt else f"没找到 [{kw}]"
        yield event.chain_result([Plain(msg)])

    @filter.command("手办化签到", prefix_optional=True)
    async def on_checkin(self, event: AstrMessageEvent, ctx=None):
        if not self.conf.get("enable_checkin", False): yield event.chain_result([Plain("未开启签到")]); return
        uid = norm_id(event.get_sender_id())
        msg = await self.data_mgr.process_checkin(uid)
        yield event.chain_result([Plain(msg)])

    @filter.command("手办化查询次数", prefix_optional=True)
    async def on_query_count(self, event: AstrMessageEvent, ctx=None):
        uid = norm_id(event.get_sender_id())
        if self.is_admin(event):
            bot_id = self._get_bot_id(event)
            for seg in event.message_obj.message:
                if isinstance(seg, At):
                    at_qq = str(seg.qq).strip()
                    if bot_id and at_qq == str(bot_id).strip():
                        continue
                    uid = at_qq;
                    break
            # 也支持纯数字指定用户
            parts = event.message_str.split()
            for p in parts:
                if p.isdigit() and len(p) >= 5:
                    uid = p;
                    break
        u_cnt = self.data_mgr.get_user_count(uid)
        msg = f"👤 用户 {uid} 剩余: {u_cnt}"
        if gid := event.get_group_id():
            msg += f"\n👥 本群剩余: {self.data_mgr.get_group_count(norm_id(gid))}"
        yield event.chain_result([Plain(msg)])

    @filter.command("手办化切换源", prefix_optional=True)
    async def on_switch_source(self, event: AstrMessageEvent, ctx=None):
        """#手办化切换源 <序号> - 切换当前图片信息源"""
        if not self.is_admin(event): return
        parts = event.message_str.split()
        if len(parts) < 2 or not parts[1].isdigit():
            yield event.chain_result([Plain("用法: #手办化切换源 <序号>\n使用 #手办化查看源 查看可用信息源")])
            return
        idx = int(parts[1])
        image_sources = [s for s in self.conf.get("generic_sources", []) if not str(s.get("video_model", "")).strip()]
        if idx < 1 or idx > len(image_sources):
            yield event.chain_result([Plain(f"序号无效，当前有 {len(image_sources)} 个图片源。使用 #手办化查看源 查看。")])
            return
        self.conf["generic_active_source"] = idx
        self._save_config(["generic_active_source"])
        src = image_sources[idx - 1]
        alias = str(src.get("alias", "")).strip()
        display_name = alias if alias else f"图片源{idx}"
        yield event.chain_result([Plain(f"✅ 已切换图片源至: {display_name}")])

    @filter.command("手办化切换视频源", prefix_optional=True)
    async def on_switch_video_source(self, event: AstrMessageEvent, ctx=None):
        """#手办化切换视频源 <序号> - 切换当前视频信息源"""
        if not self.is_admin(event): return
        parts = event.message_str.split()
        if len(parts) < 2 or not parts[1].isdigit():
            yield event.chain_result([Plain("用法: #手办化切换视频源 <序号>\n使用 #手办化查看源 查看视频源列表")])
            return
        idx = int(parts[1])
        video_sources = [s for s in self.conf.get("generic_sources", []) if str(s.get("video_model", "")).strip()]
        if idx < 1 or idx > len(video_sources):
            yield event.chain_result([Plain(f"序号无效，当前有 {len(video_sources)} 个视频源。使用 #手办化查看源 查看。")])
            return
        self.conf["generic_active_video_source"] = idx
        self._save_config(["generic_active_video_source"])
        src = video_sources[idx - 1]
        alias = str(src.get("alias", "")).strip()
        display_name = alias if alias else f"视频源{idx}"
        yield event.chain_result([Plain(f"✅ 已切换视频源至: {display_name}")])

    @filter.command("手办化查看源", prefix_optional=True)
    async def on_view_sources(self, event: AstrMessageEvent, ctx=None):
        """#手办化查看源 - 查看所有信息源"""
        if not self.is_admin(event): return

        sources = self.conf.get("generic_sources", [])
        if not sources:
            yield event.chain_result([Plain("未配置任何 API 信息源，请在 WebUI 配置页面中添加。")])
            return

        # 分类：图片源（有 keys）和视频源（有 video_model）
        image_sources = []
        video_sources = []
        for src in sources:
            if str(src.get("video_model", "")).strip():
                video_sources.append(src)
            else:
                image_sources.append(src)

        active_img_idx = int(self.conf.get("generic_active_source", 1))
        active_vid_idx = int(self.conf.get("generic_active_video_source", 1))

        lines = []

        # 图片源
        if image_sources:
            lines.append("🖼️ 图片源:")
            for i, src in enumerate(image_sources):
                alias = str(src.get("alias", "")).strip()
                src_model = src.get("model", "未设置")
                marker = " ⬅️ 当前" if (i + 1) == active_img_idx else ""
                name = alias if alias else f"图片源{i + 1}"
                lines.append(f"  📡 {name}{marker} | 模型: {src_model}")

        # 视频源
        if video_sources:
            lines.append("\n🎬 视频源:")
            for i, src in enumerate(video_sources):
                alias = str(src.get("alias", "")).strip()
                v_model = str(src.get("video_model", "")).strip()
                marker = " ⬅️ 当前" if (i + 1) == active_vid_idx else ""
                name = alias if alias else f"视频源{i + 1}"
                lines.append(f"  📡 {name}{marker} | 模型: {v_model}")

        if not image_sources and not video_sources:
            yield event.chain_result([Plain("未配置任何 API 信息源，请在 WebUI 配置页面中添加。")])
            return

        lines.append(f"\n------------------\n图片源: {len(image_sources)} 个, 视频源: {len(video_sources)} 个")
        lines.append("使用 #手办化切换源 <序号> 切换")
        yield event.chain_result([Plain("\n".join(lines))])

    @filter.command("预设图片清理", prefix_optional=True)
    async def on_cleanup_presets(self, event: AstrMessageEvent, ctx=None):
        if not self.is_admin(event): return
        parts = event.message_str.split()
        days = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 30
        count = await self.data_mgr.cleanup_old_presets(days)
        yield event.chain_result([Plain(f"✅ 清理了 {count} 张超过 {days} 天的图片")])

    @filter.command("预设图片统计", prefix_optional=True)
    async def on_preset_stats(self, event: AstrMessageEvent, ctx=None):
        if not self.is_admin(event): return
        cnt, size = self.data_mgr.get_preset_stats()
        yield event.chain_result([Plain(f"📊 缓存统计:\n数量: {cnt} 张\n占用: {size:.2f} MB")])

    @filter.command("手办化视频", aliases={"视频"}, prefix_optional=True)
    async def on_video_request(self, event: AstrMessageEvent, ctx=None):
        """#手办化视频 / #视频 [比例] [时长] <提示词> +图片（可选）

        示例: #视频 10 一只猫 | #视频 16:9 让画面动起来 +图片
        """
        raw_input = event.message_str.strip()
        # 移除指令前缀
        for cmd in ("手办化视频", "视频"):
            if raw_input.startswith(cmd):
                user_input = raw_input[len(cmd):].strip()
                break
        else:
            user_input = raw_input

        if not user_input:
            yield event.chain_result([Plain("用法: #视频 [比例] [时长] 提示词 +图片\n"
                "示例: #视频 10 一只猫 | #视频 16:9 让画面动起来 +图片\n"
                "比例: 1:1 / 2:3 / 3:2 / 9:16 / 16:9 | 时长: 6 / 10 / 15 秒")])
            return

        # 提取图片（图生视频）
        bot_id = self._get_bot_id(event)
        images = await self.img_mgr.extract_images_from_event(event, ignore_id=bot_id, context=self.context)

        # 解析参数
        aspect_ratio = self.api_mgr._parse_video_aspect_ratio(user_input)
        video_length = self.api_mgr._parse_video_length(user_input)

        # 从输入中移除已解析的参数词，保留纯提示词
        prompt = user_input
        for token in ["16:9", "9:16", "3:2", "2:3", "1:1"]:
            prompt = prompt.replace(token, "")
        prompt = re.sub(r'\d+\s*[秒sS]', '', prompt)
        # 移除独立的时长数字（6/10/15）
        tokens = prompt.split()
        filtered = [t for t in tokens if not (t.isdigit() and int(t) in (6, 10, 15))]
        prompt = " ".join(filtered).strip()

        if not prompt and not images:
            yield event.chain_result([Plain("请输入视频提示词或附带图片")])
            return

        mode = "图生视频" if images else "文生视频"
        yield event.chain_result([Plain(f"🎬 正在进行 [{mode}] · {video_length}秒 · {aspect_ratio} ...")])

        video_bytes, video_url, error = await self.api_mgr.call_video_api(
            images, prompt, aspect_ratio=aspect_ratio, video_length=video_length,
        )

        if error:
            yield event.chain_result([Plain(f"❌ 视频生成失败: {error}")])
            return

        # 处理结果
        try:
            if video_bytes:
                filename = f"video_{int(time.time())}.mp4"
                file_path = Path(StarTools.get_data_dir()) / "temp" / filename
                file_path.parent.mkdir(exist_ok=True, parents=True)
                async with aiofiles.open(file_path, "wb") as f:
                    await f.write(video_bytes)
                yield event.chain_result([Video.fromFileSystem(path=str(file_path), name=filename)])
                try:
                    await aiofiles.os.remove(file_path)
                except Exception:
                    pass
            elif video_url:
                downloaded = await self.api_mgr._download_video(video_url)
                if isinstance(downloaded, bytes) and len(downloaded) > 0:
                    filename = f"video_{int(time.time())}.mp4"
                    file_path = Path(StarTools.get_data_dir()) / "temp" / filename
                    file_path.parent.mkdir(exist_ok=True, parents=True)
                    async with aiofiles.open(file_path, "wb") as f:
                        await f.write(downloaded)
                    yield event.chain_result([Video.fromFileSystem(path=str(file_path), name=filename)])
                    try:
                        await aiofiles.os.remove(file_path)
                    except Exception:
                        pass
                else:
                    yield event.chain_result([Plain(f"❌ 视频下载失败: {downloaded}")])
            else:
                yield event.chain_result([Plain("❌ 未获取到视频")])
        except Exception as e:
            logger.error(f"视频发送失败: {e}")
            yield event.chain_result([Plain(f"❌ 视频发送失败: {e}")])

    @filter.command("手办化帮助", aliases={"lmh", "lm帮助"}, prefix_optional=True)
    async def on_help(self, event: AstrMessageEvent, ctx=None):
        yield self._get_help_node(event)

    # ================= 上下文记录与 LLM 智能判断 =================

    def _extract_message_info(self, event: AstrMessageEvent) -> Dict[str, Any]:
        """从事件中提取消息信息"""
        has_image = False
        image_urls = []
        content_parts = []

        for seg in event.message_obj.message:
            if isinstance(seg, Image):
                has_image = True

                seg_url = getattr(seg, "url", None)
                seg_file = getattr(seg, "file", None)

                if self.img_mgr._is_probably_valid_source(seg_url):
                    image_urls.append(seg_url)
                elif self.img_mgr._is_probably_valid_source(seg_file):
                    image_urls.append(seg_file)

                content_parts.append("[图片]")
            elif isinstance(seg, Plain) and seg.text:
                content_parts.append(seg.text)
            elif isinstance(seg, Reply):
                # 检查回复中是否有图片
                if seg.chain:
                    for s_chain in seg.chain:
                        if isinstance(s_chain, Image):
                            has_image = True
                            chain_url = getattr(s_chain, "url", None)
                            chain_file = getattr(s_chain, "file", None)

                            if self.img_mgr._is_probably_valid_source(chain_url):
                                image_urls.append(chain_url)
                            elif self.img_mgr._is_probably_valid_source(chain_file):
                                image_urls.append(chain_file)

        return {
            "content": "".join(content_parts) or event.message_str,
            "has_image": has_image,
            "image_urls": image_urls
        }

    @filter.event_message_type(filter.EventMessageType.ALL, priority=100)
    async def on_message_record(self, event: AstrMessageEvent, ctx=None):
        """记录所有消息到上下文管理器（高优先级，不阻断）"""
        try:
            session_id = event.unified_msg_origin
            msg_id = str(event.message_obj.message_id)
            sender_id = event.get_sender_id()
            sender_name = event.get_sender_name() or sender_id

            # 检查是否是 Bot 自己的消息
            bot_id = self._get_bot_id(event)
            is_bot = (sender_id == bot_id) if bot_id else False

            # 提取消息信息
            msg_info = self._extract_message_info(event)

            # 记录到上下文管理器
            await self.ctx_mgr.add_message(
                session_id=session_id,
                msg_id=msg_id,
                sender_id=sender_id,
                sender_name=sender_name,
                content=msg_info["content"],
                is_bot=is_bot,
                has_image=msg_info["has_image"],
                image_urls=msg_info["image_urls"]
            )

            if msg_info["image_urls"]:
                await self._remember_session_image_context(
                    session_id,
                    image_urls=msg_info["image_urls"],
                    source="bot_context" if is_bot else "user"
                )

        except Exception as e:
            logger.debug(f"FigurinePro: 消息记录失败: {e}")

        # 不阻断事件传递
        return

    @filter.llm_tool(name="shoubanhua_smart_generate")
    async def smart_generate_tool(self, event: AstrMessageEvent, user_request: str = ""):
        '''智能判断并生成图片。仅在用户明确表达需要生成图片时才调用。

        调用前请严格判断：
        1. 用户是否明确要求生成/画/创作/处理图片？
        2. 如果用户只是闲聊、询问问题、分享图片但没有要求处理，请不要调用此工具
        3. 如果用户的意图不明确，请先询问用户是否需要生成图片
        4. 如果用户是在索要你自己的照片、自拍、写真、长相，请不要用这个工具，必须改用 shoubanhua_persona_photo

        此工具会消耗用户的使用次数，请谨慎调用。
        如果用户明确说“修改上面那张 / 改刚才那张 / 继续改 / 在这个基础上改”，即使当前消息没带图片，也应按 image_to_image 处理，系统会自动取最近图片上下文。

        Args:
            user_request(string): 用户的请求描述（可选，如果为空则使用当前消息）
        '''
        # 0. 检查 LLM 工具开关
        if not self._get_conf_bool("enable_llm_tool", True):
            return self._finalize_llm_tool_result(
                "[TOOL_UNAVAILABLE] 当前功能暂时不可用。请用自己平时的语气告诉用户现在不方便，别提功能禁用、指令模式等词汇，要自然。")

        # 0.1 检查图片生成冷却时间
        uid = norm_id(event.get_sender_id())
        in_cooldown, remaining = self._check_image_cooldown(uid)
        if in_cooldown:
            # 返回借口让LLM用自然语言拒绝
            excuse = self._get_cooldown_excuse(remaining)
            return f"【冷却中】{excuse}\n\n请用自然的方式告诉用户现在不方便生成图片，可以稍后再试。不要直接说'冷却'这个词。"

        # 1. 获取上下文
        session_id = event.unified_msg_origin
        context_messages = await self.ctx_mgr.get_recent_messages(
            session_id,
            count=self._context_rounds
        )

        # 2. 提取当前消息信息
        msg_info = self._extract_message_info(event)
        current_message = user_request or msg_info["content"]
        has_current_image = msg_info["has_image"]
        has_recent_image_context = await self._has_recent_session_image_context(session_id)
        references_recent_image = (
                (not has_current_image)
                and has_recent_image_context
                and self._looks_like_context_image_reference(current_message)
        )

        # 2.5 对“看你本人照片/自拍/写真”做强制分流，避免 LLM 误走通用智能工具
        if self._persona_mode and (
                self._looks_like_persona_photo_request(current_message)
                or self._looks_like_persona_followup_request(current_message, context_messages)
        ):
            inferred_count = self._infer_requested_count_from_text(current_message, default=1, multi_default=3)
            return await self.persona_photo_tool(
                event=event,
                scene_hint="",
                extra_request=current_message,
                count=inferred_count,
            )

        # 3. 分析任务类型。对“修改上面那张/继续改刚才那张”优先按图生图处理。
        if references_recent_image:
            analysis = {
                "task_type": "image_to_image",
                "confidence": 0.95,
                "reason": "检测到用户明确引用最近图片上下文并要求修改",
                "suggested_prompt": current_message,
            }
        else:
            analysis = LLMTaskAnalyzer.analyze_task_type(
                current_message=current_message,
                context_messages=context_messages,
                has_current_image=has_current_image
            )

        task_type = analysis["task_type"]
        confidence = analysis["confidence"]
        reason = analysis["reason"]

        logger.info(f"FigurinePro 智能判断: {task_type} (置信度: {confidence:.2f}) - {reason}")

        # 4. 置信度检查
        if confidence < self._auto_detect_confidence:
            return f"无法确定任务类型 (置信度: {confidence:.2f})。请明确指定是文生图还是图生图。\n分析: {reason}"

        # 5. 根据任务类型执行
        if task_type == "none":
            return f"根据分析，当前请求不需要生成图片。\n分析: {reason}"

        elif task_type == "text_to_image":
            # 文生图
            prompt = analysis.get("suggested_prompt", current_message)
            final_prompt, preset_name, extra_rules = self._process_prompt_and_preset(prompt)

            # 根据配置决定是否发送进度提示
            if self._get_conf_bool("llm_show_progress", True):
                feedback = self._build_llm_progress_text(
                    "auto_text", preset_name=preset_name, extra_request=extra_rules, confidence=confidence
                )
                await event.send(event.chain_result([Plain(feedback)]))

            gid = norm_id(event.get_group_id())
            deduction = await self._check_quota(event, uid, gid, 1)
            if not deduction["allowed"]:
                return deduction["msg"]

            # 更新冷却时间
            self._update_image_cooldown(uid)

            await self._register_pending_generation(event.unified_msg_origin, 1)
            success, error_msg = await self._run_background_task(
                event, [], final_prompt, preset_name, deduction, uid, gid, 1, extra_rules,
                hide_text=True, suppress_user_error=True
            )
            if not success:
                return self._build_llm_tool_failure(error_msg)

            return self._finalize_llm_tool_success(
                f"[TOOL_SUCCESS] 文生图任务已结束，预设：{preset_name}，图片已经发送给用户。你可以按原本人设自然接话，也可以不补充收尾。")

        elif task_type == "image_to_image":
            # 图生图
            prompt = analysis.get("suggested_prompt", current_message)
            processed_prompt, preset_name, extra_rules = self._process_prompt_and_preset(prompt)

            # 根据配置决定是否发送进度提示
            if self._get_conf_bool("llm_show_progress", True):
                feedback = self._build_llm_progress_text(
                    "auto_image", preset_name=preset_name, extra_request=extra_rules, confidence=confidence
                )
                await event.send(event.chain_result([Plain(feedback)]))

            # 提取图片；当前消息没图时，会自动使用最近用户图片或 Bot 生成图。
            images = await self._resolve_contextual_image_inputs(
                event,
                prompt=prompt,
                merge_multiple_images=False,
                include_current=True,
                allow_context_fallback=True,
            )

            if not images:
                return self._finalize_llm_tool_result(
                    "[TOOL_FAILED] 未检测到图片。请直接自然告诉用户需要先发图或引用图片，不要再次调用工具。")

            gid = norm_id(event.get_group_id())
            deduction = await self._check_quota(event, uid, gid, 1)
            if not deduction["allowed"]:
                return deduction["msg"]

            # 更新冷却时间
            self._update_image_cooldown(uid)

            await self._register_pending_generation(event.unified_msg_origin, 1)
            success, error_msg = await self._run_background_task(
                event, images, processed_prompt, preset_name, deduction, uid, gid, 1,
                extra_rules, hide_text=True, suppress_user_error=True
            )
            if not success:
                return self._build_llm_tool_failure(error_msg)

            return self._finalize_llm_tool_success(
                f"[TOOL_SUCCESS] 图生图任务已结束，预设：{preset_name}，图片已经发送给用户。你可以按原本人设自然接话，也可以不补充收尾。")

        return "未知任务类型"

    @filter.command("预设参考图添加", aliases={"lmref添加", "添加参考图"}, prefix_optional=True)
    async def on_add_preset_ref(self, event: AstrMessageEvent, ctx=None):
        """为预设添加参考图（管理员）

        用法: #预设参考图添加 <预设名> [图片]
        """
        if not self.is_admin(event): return

        # 解析预设名
        parts = event.message_str.split()
        if len(parts) < 2:
            yield event.chain_result([Plain("用法: #预设参考图添加 <预设名> [图片]\n请同时发送或引用图片")])
            return

        preset_name = parts[1].strip()

        # 检查预设是否存在
        if preset_name not in self.data_mgr.prompt_map:
            yield event.chain_result([Plain(f"预设 [{preset_name}] 不存在，先用 #lm添加 创建一个吧")])
            return

        # 提取图片
        bot_id = self._get_bot_id(event)
        images = await self.img_mgr.extract_images_from_event(event, ignore_id=bot_id, context=self.context)

        if not images:
            yield event.chain_result([Plain("没检测到图片，发送或引用一下图片再试")])
            return

        # 保存参考图
        count = await self.data_mgr.add_preset_ref_images(preset_name, images)

        if count > 0:
            total = len(self.data_mgr.get_preset_ref_image_paths(preset_name))
            yield event.chain_result(
                [Plain(f"✅ 已为预设 [{preset_name}] 添加 {count} 张参考图\n当前共 {total} 张参考图")])
        else:
            yield event.chain_result([Plain("参考图保存失败了，再试试？")])

    @filter.command("预设参考图查看", aliases={"lmref查看", "查看参考图"}, prefix_optional=True)
    async def on_view_preset_ref(self, event: AstrMessageEvent, ctx=None):
        """查看预设的参考图（管理员）

        用法: #预设参考图查看 <预设名>
        """
        if not self.is_admin(event): return

        parts = event.message_str.split()
        if len(parts) < 2:
            yield event.chain_result([Plain("用法: #预设参考图查看 <预设名>")])
            return

        preset_name = parts[1].strip()

        if not self.data_mgr.has_preset_ref_images(preset_name):
            yield event.chain_result([Plain(f"预设 [{preset_name}] 没有参考图")])
            return

        # 加载参考图
        ref_images = await self.data_mgr.load_preset_ref_images_bytes(preset_name)

        if not ref_images:
            yield event.chain_result([Plain(f"预设 [{preset_name}] 的参考图加载失败")])
            return

        # 发送参考图
        result = [Plain(f"📷 预设 [{preset_name}] 的参考图 ({len(ref_images)} 张):\n")]
        for i, img_bytes in enumerate(ref_images[:5]):  # 最多显示5张
            result.append(Image.fromBytes(img_bytes))

        if len(ref_images) > 5:
            result.append(Plain(f"\n... 还有 {len(ref_images) - 5} 张未显示"))

        yield event.chain_result(result)

    @filter.command("预设参考图清除", aliases={"lmref清除", "清除参考图"}, prefix_optional=True)
    async def on_clear_preset_ref(self, event: AstrMessageEvent, ctx=None):
        """清除预设的所有参考图（管理员）

        用法: #预设参考图清除 <预设名>
        """
        if not self.is_admin(event): return

        parts = event.message_str.split()
        if len(parts) < 2:
            yield event.chain_result([Plain("用法: #预设参考图清除 <预设名>")])
            return

        preset_name = parts[1].strip()

        count = await self.data_mgr.clear_preset_ref_images(preset_name)

        if count > 0:
            yield event.chain_result([Plain(f"✅ 已清除预设 [{preset_name}] 的 {count} 张参考图")])
        else:
            yield event.chain_result([Plain(f"预设 [{preset_name}] 没有参考图")])

    @filter.command("预设参考图删除", aliases={"lmref删除", "删除参考图"}, prefix_optional=True)
    async def on_remove_preset_ref(self, event: AstrMessageEvent, ctx=None):
        """删除预设的指定参考图（管理员）

        用法: #预设参考图删除 <预设名> <序号>
        """
        if not self.is_admin(event): return

        parts = event.message_str.split()
        if len(parts) < 3:
            yield event.chain_result([Plain("用法: #预设参考图删除 <预设名> <序号>\n序号从1开始")])
            return

        preset_name = parts[1].strip()

        if not parts[2].isdigit():
            yield event.chain_result([Plain("序号得填数字啊")])
            return

        index = int(parts[2]) - 1  # 转为0开始的索引

        success = await self.data_mgr.remove_preset_ref_image(preset_name, index)

        if success:
            remaining = len(self.data_mgr.get_preset_ref_image_paths(preset_name))
            yield event.chain_result(
                [Plain(f"✅ 已删除预设 [{preset_name}] 的第 {index + 1} 张参考图\n剩余 {remaining} 张")])
        else:
            yield event.chain_result([Plain("删除失败了，检查一下预设名和序号对不对？")])

    @filter.command("预设参考图统计", aliases={"lmref统计", "参考图统计"}, prefix_optional=True)
    async def on_preset_ref_stats(self, event: AstrMessageEvent, ctx=None):
        """查看预设参考图统计（管理员）"""
        if not self.is_admin(event): return

        stats = self.data_mgr.get_preset_ref_stats()

        msg = f"📊 预设参考图统计:\n"
        msg += f"有参考图的预设: {stats['total_presets']} 个\n"
        msg += f"总图片数: {stats['total_images']} 张\n"
        msg += f"总占用: {stats['total_size_mb']:.2f} MB\n"

        if stats['details']:
            msg += f"\n详情:\n"
            for preset, count in sorted(stats['details'].items(), key=lambda x: -x[1])[:10]:
                msg += f"  {preset}: {count} 张\n"

            if len(stats['details']) > 10:
                msg += f"  ... 还有 {len(stats['details']) - 10} 个预设"

        yield event.chain_result([Plain(msg)])

    @filter.command("预设参考图列表", aliases={"lmref列表", "参考图列表"}, prefix_optional=True)
    async def on_list_preset_refs(self, event: AstrMessageEvent, ctx=None):
        """列出所有有参考图的预设（管理员）"""
        if not self.is_admin(event): return

        stats = self.data_mgr.get_preset_ref_stats()

        if not stats['details']:
            yield event.chain_result([Plain("暂无预设参考图")])
            return

        msg = f"📋 有参考图的预设列表:\n"
        for preset, count in sorted(stats['details'].items()):
            has_prompt = "✓" if preset in self.data_mgr.prompt_map else "✗"
            msg += f"  [{preset}] {count}张 (预设{has_prompt})\n"

        yield event.chain_result([Plain(msg)])

    # ================= 批量处理图片功能 =================

    async def _collect_images_from_context(self, session_id: str, count: int = 10,
                                           include_bot: bool = False,
                                           sender_id: str = "") -> List[Tuple[str, List[str]]]:
        """
        从上下文中收集图片

        Args:
            session_id: 会话ID
            count: 获取的消息数量
            include_bot: 是否包含Bot发出的图片
            sender_id: 仅收集该发送者的图片；为空时不过滤发送者

        Returns:
            [(消息ID, [图片URL列表]), ...]
        """
        messages = await self.ctx_mgr.get_recent_messages(session_id, count=count)

        result = []
        for msg in messages:
            if msg.has_image and msg.image_urls:
                if msg.is_bot:
                    if not include_bot:
                        continue
                elif sender_id and norm_id(getattr(msg, "sender_id", "")) != norm_id(sender_id):
                    continue

                filtered_urls = []
                seen_urls = set()
                for url in msg.image_urls:
                    if not self.img_mgr._is_probably_valid_source(url):
                        continue
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)
                    filtered_urls.append(url)

                if filtered_urls:
                    result.append((msg.msg_id, filtered_urls))

        return result

    def _merge_batch_image_urls(self, current_urls: List[str], context_image_sources: List[Tuple[str, List[str]]],
                                max_images: int = 10, prefer_current: bool = True) -> List[str]:
        """
        批量任务统一的图片顺序整理逻辑：
        1. 优先使用当前消息中的图片，保持用户发送顺序
        2. 当前消息没有图片时，再回退到上下文中的历史图片
        3. 全程过滤头像/无效来源并去重
        """
        all_image_urls = []
        seen_urls = set()

        def add_url(url: str):
            if not self.img_mgr._is_probably_valid_source(url):
                return
            if url in seen_urls:
                return
            seen_urls.add(url)
            all_image_urls.append(url)

        has_current_inputs = bool(current_urls) and prefer_current

        if has_current_inputs:
            for url in current_urls:
                add_url(url)
        else:
            for _, urls in reversed(context_image_sources):
                for url in urls:
                    add_url(url)

            # 上面是从最新消息开始收集，为了让最终处理顺序符合用户看到的时间线，这里反转回来
            all_image_urls.reverse()

        if max_images > 0:
            all_image_urls = all_image_urls[:max_images]

        return all_image_urls

    def _normalize_pdf_filename(self, filename: str, fallback: str = "图片合集") -> str:
        """清理并规范化 PDF 文件名。"""
        text = str(filename or "").strip()

        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[\\/:*?\"<>|\r\n\t]+", "_", text)
        text = re.sub(r"\.pdf\s*$", "", text, flags=re.IGNORECASE)
        text = text.strip(" ._，。；;、\"'“”‘’()（）[]【】<>《》")

        if len(text) > 40:
            text = text[:40].rstrip(" ._")

        return text or fallback

    def _extract_explicit_pdf_filename(self, text: str = "") -> str:
        """从用户文本中提取明确指定的 PDF 文件名。"""
        raw = str(text or "").strip()
        if not raw:
            return ""

        normalized = raw.replace("\n", " ").replace("\r", " ")
        normalized = re.sub(r"\s+", " ", normalized).strip()

        patterns = [
            r"(?:pdf|PDF)\s*(?:名字|名称|文件名)?\s*(?:叫|是|为|命名为|命名成)\s*[：: ]*\s*[\"“”']?([^\"“”'\s]+(?:\.pdf)?)",
            r"(?:文件名|名称|名字)\s*(?:叫|是|为|命名为|命名成)\s*[：: ]*\s*[\"“”']?([^\"“”'\s]+(?:\.pdf)?)",
            r"(?:输出为|保存为|导出为|打包为)\s*[\"“”']?([^\"“”'\s]+(?:\.pdf)?)",
            r"[\"“”']([^\"“”']+?\.pdf)[\"“”']",
            r"\b([A-Za-z0-9_\-\u4e00-\u9fff]+\.pdf)\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if not match:
                continue

            candidate = self._normalize_pdf_filename(match.group(1))
            if candidate:
                return candidate

        return ""

    def _guess_pdf_topic_name(self, prompt: str = "", preset_name: str = "") -> str:
        """在未指定文件名时，尽量生成简短且符合任务语义的名字，而不是直接使用整段提示词。"""
        raw_text = str(prompt or "").strip()
        normalized = raw_text.replace("\n", " ").replace("\r", " ")
        normalized = re.sub(r"\s+", " ", normalized).strip()

        preset_text = "" if str(preset_name or "").strip() in {"", "自定义", "编辑", "edit", "custom"} else str(
            preset_name).strip()

        # 1. 先按明显任务类型命名
        keyword_map = [
            (["漫画", "翻译", "日语", "汉化"], "漫画翻译"),
            (["手办", "手办化"], "手办化"),
            (["批量", "全部处理"], "批量处理"),
            (["打包", "pdf", "PDF"], "图片合集"),
            (["换装", "换衣", "穿搭"], "换装图集"),
            (["写真", "自拍"], "写真合集"),
        ]
        lowered = normalized.lower()
        for keywords, title in keyword_map:
            if any(keyword.lower() in lowered for keyword in keywords):
                if title == "批量处理" and preset_text:
                    return f"{preset_text}结果"
                return title

        # 2. 预设名优先作为简短标题
        if preset_text:
            return f"{preset_text}结果"

        # 3. 从文本里提取少量核心词，避免直接整句当文件名
        cleanup_patterns = [
            r"Additional requirements:\s*",
            r"Edit the image according to the following instructions:\s*",
            r"(?:帮我|给我|把|将|麻烦|请)\s*",
            r"(?:打包成?|合成|整理成?)\s*pdf",
            r"(?:名字|名称|文件名).*$",
        ]
        candidate = normalized
        for pattern in cleanup_patterns:
            candidate = re.sub(pattern, "", candidate, flags=re.IGNORECASE).strip()

        chunks = re.split(r"[，。,.、；;：:\-—_\s]+", candidate)
        stop_words = {
            "的", "了", "呢", "啊", "呀", "吧", "一下", "一个", "一些", "这个", "那个",
            "图片", "图", "照片", "处理", "生成", "结果", "合集", "pdf", "PDF"
        }

        picked = []
        for chunk in chunks:
            chunk = self._normalize_pdf_filename(chunk, fallback="")
            if not chunk:
                continue
            if chunk in stop_words:
                continue
            if len(chunk) <= 1:
                continue
            picked.append(chunk)
            if len(picked) >= 2:
                break

        if picked:
            return "".join(picked)[:16]

        return "图片合集"

    def _build_pdf_filename_hint(self, prompt: str = "", preset_name: str = "", count: int = 0) -> str:
        """根据用户批量请求生成更自然的 PDF 文件名提示，优先使用明确指定的名字。"""
        explicit_name = self._extract_explicit_pdf_filename(prompt)
        if explicit_name:
            return f"{explicit_name}.pdf"

        text = self._guess_pdf_topic_name(prompt=prompt, preset_name=preset_name)
        text = self._normalize_pdf_filename(text)

        if count > 0 and not re.search(r"\d+\s*张", text) and text not in {"漫画翻译", "图片合集"}:
            text = f"{text}{count}张"

        return f"{text}.pdf"

    async def _gather_images_for_pdf(self, event: AstrMessageEvent, max_images: int = 10,
                                     wait_for_generation: bool = True) -> List[bytes]:
        """
        收集用于打包 PDF 的图片；当图片可能仍在后台生成时，等待一段时间直到图片数量稳定。

        Args:
            event: 消息事件
            max_images: 最多收集的图片数量，<=0 表示不限制
            wait_for_generation: 是否等待后台图片生成完成

        Returns:
            图片字节列表
        """

        async def collect_once() -> List[bytes]:
            images_bytes = []
            session_id = event.unified_msg_origin
            sender_id = norm_id(event.get_sender_id())

            # 0. 如果当前会话里已经有本轮成功生成结果，优先只打包这些结果，
            # 避免把源图、历史图或群里其他人中途发送的图片混进来。
            # 1. 先从上下文获取（包含 Bot 发出的图片）
            image_sources = await self._collect_images_from_context(
                session_id,
                count=30,
                include_bot=True,
                sender_id=sender_id
            )
            all_urls = []
            seen_urls = set()

            for _, urls in reversed(image_sources):
                for url in urls:
                    if not self.img_mgr._is_probably_valid_source(url):
                        continue
                    if url not in seen_urls:
                        all_urls.append(url)
                        seen_urls.add(url)

            if max_images > 0:
                all_urls = all_urls[:max_images]

            all_urls.reverse()

            for url in all_urls:
                try:
                    img_b = await self.img_mgr.load_bytes(url)
                    if img_b and len(img_b) > 0:
                        images_bytes.append(img_b)
                except Exception as e:
                    logger.error(f"收集 PDF 图片时下载失败 {url[:20]}: {e}")

            # 2. 再提取当前消息中的图片（包括引用）
            bot_id = self._get_bot_id(event)
            current_images = await self.img_mgr.extract_images_from_event(event, ignore_id=bot_id, context=self.context)
            images_bytes.extend([b for b in current_images if b and len(b) > 0])

            # 按内容去重（从后往前保留最新）
            unique_images = []
            seen_hashes = set()
            for img in reversed(images_bytes):
                img_hash = hash(img)
                if img_hash not in seen_hashes:
                    unique_images.append(img)
                    seen_hashes.add(img_hash)

            unique_images.reverse()

            # 最终再次严格按 max_images 截断
            if max_images > 0:
                unique_images = unique_images[-max_images:]

            return unique_images

        if not wait_for_generation:
            return await collect_once()

        max_wait_seconds = max(0, self.conf.get("pdf_wait_timeout", 120))
        poll_interval = max(1, self.conf.get("pdf_wait_poll_interval", 2))
        stable_rounds_required = max(1, self.conf.get("pdf_wait_stable_rounds", 2))

        waited = 0
        last_count = -1
        stable_rounds = 0
        latest_images = []

        while waited <= max_wait_seconds:
            latest_images = await collect_once()
            current_count = len(latest_images)

            pending_count = await self._get_pending_generation_count(event.unified_msg_origin)

            # 达到数量上限时，只有确认后台已经全部完成才结束等待
            if max_images > 0 and current_count >= max_images and pending_count == 0:
                break

            # 只有在后台没有待完成生成任务时，才允许按“数量稳定”判断结束
            if pending_count == 0 and current_count > 0 and current_count == last_count:
                stable_rounds += 1
                if stable_rounds >= stable_rounds_required:
                    break
            else:
                stable_rounds = 0

            last_count = current_count

            if waited >= max_wait_seconds:
                break

            await asyncio.sleep(poll_interval)
            waited += poll_interval

        return latest_images

    def _mask_llm_error(self, error: Any, default_msg: str = "操作失败，请稍后再试。") -> str:
        """屏蔽返回给 LLM/用户的内部异常细节"""
        try:
            err_text = str(error).strip() if error is not None else ""
        except Exception:
            err_text = ""

        if not err_text:
            return default_msg

        translated = self._translate_error_to_chinese(err_text, default_msg=default_msg)

        unsafe_keywords = [
            "traceback", "exception", "stack", "context", "attributeerror",
            "keyerror", "indexerror", "typeerror", "valueerror", "runtimeerror",
            "http", "https", "file not found", "no such file", "invalid", "none"
        ]
        lower_text = err_text.lower()
        if any(k in lower_text for k in unsafe_keywords):
            return default_msg

        if translated.startswith("未知错误:"):
            return default_msg

        return translated or default_msg

    async def _pack_and_send_pdf(self, event: AstrMessageEvent, images_bytes: List[bytes],
                                 success_prefix: str = "✅ 成功将图片打包为 PDF",
                                 filename_hint: str = "") -> Tuple[bool, str]:
        """独立执行图片打包为 PDF 并发送，不依赖大模型处理逻辑"""
        tmp_path = ""
        try:
            valid_images = [img for img in images_bytes if isinstance(img, bytes) and len(img) > 0]
            if not valid_images:
                return False, "没检测到有效的图片。"

            pdf_bytes = self.img_mgr.images_to_pdf(valid_images)
            if not pdf_bytes:
                return False, "打包 PDF 失败了，可能图片格式不支持。"

            import uuid
            import os
            from astrbot.core.message.components import File

            filename = (filename_hint or "").strip()
            if not filename:
                filename = f"images_packed_{uuid.uuid4().hex[:6]}.pdf"
            elif not filename.lower().endswith(".pdf"):
                filename = f"{filename}.pdf"

            tmp_path = os.path.join(self.data_mgr.data_dir, filename)
            with open(tmp_path, "wb") as f:
                f.write(pdf_bytes)

            await event.send(event.chain_result(
                [File(name=filename, file=tmp_path), Plain(f"\n{success_prefix}（共 {len(valid_images)} 张）")]))
            return True, f"成功将 {len(valid_images)} 张图片打包为 PDF"
        except Exception as e:
            logger.error(f"独立打包 PDF 异常: {e}", exc_info=True)
            return False, f"{self._mask_llm_error(e, '打包 PDF 失败了，请稍后再试。')}"
        finally:
            await self._cleanup_temp_file(tmp_path)

    def _translate_error_to_chinese(self, error: str, default_msg: str = "这边刚才有点小问题，稍后再试试。") -> str:
        """将错误信息翻译为中文"""
        error_lower = str(error).lower()

        # 网络相关错误
        if "timeout" in error_lower or "timed out" in error_lower:
            return "请求超时，API响应时间过长"
        if "connection" in error_lower and ("refused" in error_lower or "reset" in error_lower):
            return "连接被拒绝或重置，网络不稳定"
        if "connection" in error_lower:
            return "网络连接异常"
        if "ssl" in error_lower or "certificate" in error_lower:
            return "SSL证书验证失败"
        if "dns" in error_lower or "resolve" in error_lower:
            return "DNS解析失败，无法访问服务器"

        # API相关错误
        if "rate limit" in error_lower or "429" in error_lower:
            return "API请求频率过高，触发限流"
        if "quota" in error_lower or "exceeded" in error_lower:
            return "API配额已用尽"
        if "unauthorized" in error_lower or "401" in error_lower:
            return "API密钥无效或已过期"
        if "forbidden" in error_lower or "403" in error_lower:
            return "API访问被禁止"
        if "not found" in error_lower or "404" in error_lower:
            return "API接口不存在"
        if "500" in error_lower or "internal server" in error_lower:
            return "API服务器内部错误"
        if "502" in error_lower or "bad gateway" in error_lower:
            return "API网关错误"
        if "503" in error_lower or "service unavailable" in error_lower:
            return "API服务暂时不可用"
        if "524" in error_lower:
            return "Cloudflare超时，请求时间过长"

        # 图片相关错误
        if "image" in error_lower and ("invalid" in error_lower or "corrupt" in error_lower):
            return "图片格式无效或已损坏"
        if "image" in error_lower and "size" in error_lower:
            return "图片尺寸不符合要求"
        if "download" in error_lower:
            return "图片下载失败"
        if "base64" in error_lower:
            return "图片编码失败"

        # 内容相关错误
        if "content" in error_lower and ("policy" in error_lower or "filter" in error_lower):
            return "内容被安全策略过滤"
        if "nsfw" in error_lower or "inappropriate" in error_lower:
            return "内容不符合安全规范"

        # JSON相关错误
        if "json" in error_lower:
            return "API返回数据格式异常"

        return default_msg

    async def _run_single_batch_task(self, event: AstrMessageEvent, image_bytes: bytes,
                                     prompt: str, preset_name: str, task_index: int, total_tasks: int,
                                     uid: str, gid: str, extra_rules: str = "",
                                     image_source: str = "", hide_text: bool = False) -> Tuple[bool, str]:
        """
        执行单个批量任务（不扣费，由调用方统一扣费）

        Returns:
            (是否成功, 错误信息)
        """
        try:
            # 加载预设参考图（如果有）
            images = [image_bytes]
            if preset_name != "自定义" and self.conf.get("enable_preset_ref_images", True):
                ref_images = await self._load_preset_ref_images(preset_name)
                if ref_images:
                    images = ref_images + images

            # 调用 API
            model = ""
            start_time = datetime.now()

            res = await self.api_mgr.call_api(images, prompt, model, False, self.img_mgr.proxy)

            # 处理结果
            if isinstance(res, bytes):
                res = await self._prepare_send_image_bytes(res)
                elapsed = (datetime.now() - start_time).total_seconds()
                await self.data_mgr.record_usage(uid, gid)
                await self._register_generation_success(event.unified_msg_origin, 1)
                await self._register_generated_image(event.unified_msg_origin, res)

                # PDF 暂存模式：不发送单张图片，用 task_index 保证顺序
                if await self._maybe_stage_after_grace(event.unified_msg_origin, res, staging_index=task_index):
                    return True, ""

                chain_nodes = [Image.fromBytes(res)]
                if not hide_text:
                    # 构建成功文案
                    timing_text = self._format_success_timing(elapsed)
                    info_text = f"\n✅ [{task_index}/{total_tasks}] 生成成功 ({timing_text}) | 预设: {preset_name}"
                    if extra_rules:
                        info_text += f" | 规则: {extra_rules[:15]}..."
                    chain_nodes.append(Plain(info_text))

                # 发送结果
                chain = event.chain_result(chain_nodes)
                await event.send(chain)
                return True, ""
            else:
                # API返回错误
                error_msg = self._translate_error_to_chinese(res)
                logger.error(f"Batch task {task_index} API error: {res}")
                return False, error_msg

        except Exception as e:
            # 系统异常
            error_msg = self._translate_error_to_chinese(str(e))
            logger.error(f"Batch task {task_index} exception: {e}", exc_info=True)
            return False, error_msg

    @filter.command("打包PDF", aliases={"图片转PDF", "合成PDF"}, prefix_optional=True)
    async def on_pack_pdf_cmd(self, event: AstrMessageEvent, ctx=None):
        """将上下文或当前消息中的图片打包为PDF（不生成新图，纯打包）"""

        can_pack, block_msg = await self._can_pack_pdf_now(event)
        if not can_pack:
            yield event.chain_result([Plain(block_msg)])
            return

        yield event.chain_result([Plain("📦 正在检测可用图片，请稍候...")])

        valid_images_bytes = await self._gather_images_for_pdf(event, max_images=0, wait_for_generation=True)

        can_pack, block_msg = await self._can_pack_pdf_now(event, valid_images_bytes)
        if not can_pack:
            yield event.chain_result([Plain(block_msg)])
            return

        if not valid_images_bytes:
            yield event.chain_result([Plain("没检测到有效的图片，如果还在弄的话稍等一下再试~")])
            return

        yield event.chain_result([Plain(f"📦 已检测到 {len(valid_images_bytes)} 张图片，正在打包为 PDF，请稍候...")])

        success, msg = await self._pack_and_send_pdf(
            event,
            valid_images_bytes,
            success_prefix="✅ 成功将图片打包为 PDF",
            filename_hint=self._build_pdf_filename_hint(event.message_str, count=len(valid_images_bytes))
        )
        if not success:
            yield event.chain_result([Plain(msg)])

    @filter.llm_tool(name="shoubanhua_pack_images_to_pdf")
    async def pack_images_to_pdf_tool(self, event: AstrMessageEvent, max_images: int = 10):
        '''将上下文中的图片（包括Bot正在生成或已经生成的图片）直接打包成一个PDF文件发送给用户。注意：此工具【不会】修改图片，【不会】生成新图，仅仅是把已有的图片原样打包。

        【调用规则（必须严格遵守）】
        1. 用户必须明确要求"把刚才的图打包成PDF"、"合成PDF"、"把这些图片做成PDF"时，才可以调用。
        2. 如果用户要求"把这些图片手办化/处理后再打包"，请使用 `shoubanhua_batch_process` 工具并设置 output_as_pdf=True。
        3. 这个工具只负责【纯打包】，会自动等待正在生成的图片全部完成后再打包。
        4. 如果前面有图片生成任务正在进行，此工具会自动等待它们完成，无需担心时序问题。
        5. 如果用户说的是"生成/处理 N 张后再打包 PDF"，其中 N 是生成数量，不是本工具的 max_images；不要把数量错误地传给本工具。

        Args:
            max_images(int): 最多打包的图片数量，默认10张，可以根据用户需求调整。
        '''
        session_id = event.unified_msg_origin

        # 0. 基本前置检查
        can_pack, block_msg = await self._can_pack_pdf_now(event)
        if not can_pack:
            return block_msg

        # 1. 立即进入 PDF 暂存模式
        #    后续的后台生成任务将不再发送单张图片，而是写入暂存列表
        await self._enter_pdf_staging_mode(session_id)

        await event.send(event.chain_result([Plain("📦 正在等待所有图片就绪，请稍候...")]))

        try:
            # 2. 等待所有后台生成任务完成，并收集暂存的图片
            pdf_wait_timeout = max(30, self.conf.get("pdf_wait_timeout", 120))
            valid_images_bytes = await self._wait_for_all_generations_and_collect(
                session_id, timeout=pdf_wait_timeout
            )

            # 3. 如果暂存+缓存都没有图片，回退到旧的收集逻辑（兼容直接打包上下文图片的场景）
            if not valid_images_bytes:
                try:
                    valid_images_bytes = await self._gather_images_for_pdf(
                        event, max_images=max_images, wait_for_generation=False
                    )
                except Exception as e:
                    logger.error(f"收集 PDF 图片异常: {e}", exc_info=True)

            if not valid_images_bytes:
                return self._finalize_llm_tool_result(
                    "[TOOL_FAILED] 等了一会儿还是没找到可用的图片。\n"
                    "请用你自己的语气告诉用户现在没有图可以打包，稍后再来。"
                )

            # 4. 按 max_images 截断
            if max_images > 0 and len(valid_images_bytes) > max_images:
                valid_images_bytes = valid_images_bytes[-max_images:]

            await event.send(event.chain_result([
                Plain(f"📦 已收集到 {len(valid_images_bytes)} 张图片，正在打包为 PDF，请稍候...")
            ]))

            # 5. 打包并发送 PDF
            success, msg = await self._pack_and_send_pdf(
                event,
                valid_images_bytes,
                success_prefix="✅ 成功将图片打包为 PDF",
                filename_hint=self._build_pdf_filename_hint(
                    event.message_str, count=len(valid_images_bytes)
                )
            )

            if success:
                # 打包成功后清除本次会话的图片缓存，防止旧图污染下次打包
                await self._clear_session_image_cache(session_id)
                return self._finalize_llm_tool_success(
                    "[TOOL_SUCCESS] 图片已经打包成 PDF 并发送给用户。"
                    "你可以按原本人设自然接话，也可以不补充收尾。"
                )
            return msg

        except Exception as e:
            logger.error(f"PDF 打包异常: {e}", exc_info=True)
            return self._finalize_llm_tool_result(
                f"[TOOL_FAILED] {self._mask_llm_error(e, '打包 PDF 失败了')}\n"
                "请用你自己的语气告诉用户现在搞不了，稍后再试试。"
            )
        finally:
            # 6. 无论成功失败，都退出暂存模式
            await self._exit_pdf_staging_mode(session_id)

    @filter.llm_tool(name="shoubanhua_batch_process")
    async def batch_process_tool(self, event: AstrMessageEvent, prompt: str, max_images: int = 10,
                                 output_as_pdf: bool = False):
        '''批量处理上下文中的多张图片，也可以将结果打包为PDF输出。仅在用户明确要求批量处理多张图片时才调用。

        调用前请严格判断：
        1. 用户是否明确要求批量处理/全部处理多张图片？或者用户是否要求将图片打包成PDF输出？
        2. 上下文中是否确实有多张图片或PDF需要处理？
        3. 如果用户只是发送了多张图片但没有要求处理，请不要调用此工具

        此工具会消耗用户大量使用次数（每张图片消耗1次），请谨慎调用。

        Args:
            prompt(string): 图片处理的提示词，可以是预设名+追加规则，如"手办化 皮肤白一点"
            max_images(int): 最多处理的图片数量，默认10张
            output_as_pdf(boolean): 用户是否明确要求将生成的多张图片打包成PDF输出，默认False
        '''
        # 统一转发到并发版本，避免 LLM 误调用本工具时退化成串行处理
        return await self.batch_process_concurrent_tool(
            event=event,
            prompt=prompt,
            max_images=max_images,
            concurrency=self.conf.get("batch_concurrency", 3),
            output_as_pdf=output_as_pdf
        )

        # 0. 读取配置中的限制
        conf_max_images = self.conf.get("batch_max_images", 10)
        max_images = min(max_images, conf_max_images) if max_images > 0 else conf_max_images

        # 0. 检查 LLM 工具开关
        if not self._get_conf_bool("enable_llm_tool", True):
            return self._finalize_llm_tool_result(
                "[TOOL_UNAVAILABLE] 当前功能暂时不可用。请用自己平时的语气告诉用户现在不方便，别提功能禁用、指令模式等词汇，要自然。")

        active_task = await self._get_active_session_task(event.unified_msg_origin)
        if active_task:
            return self._finalize_llm_tool_result(f"[TOOL_SUCCESS] {self._build_active_task_reply(active_task)}")

        # 0.1 检查图片生成冷却时间
        uid = norm_id(event.get_sender_id())
        in_cooldown, remaining = self._check_image_cooldown(uid)
        if in_cooldown:
            # 返回借口让LLM用自然语言拒绝
            excuse = self._get_cooldown_excuse(remaining)
            return f"【冷却中】{excuse}\n\n请用自然的方式告诉用户现在不方便处理图片，可以稍后再试。不要直接说'冷却'这个词。"

        # 1. 提取当前消息的图片 URL（包括引用消息中的图片）
        msg_info = self._extract_message_info(event)
        current_urls = msg_info.get("image_urls", [])

        # 1.5 提取当前消息的 PDF（转化为图片）
        current_pdf_bytes_list = await self.img_mgr.extract_pdfs_from_event(event)
        pdf_extracted_urls = []
        for pdf_bytes in current_pdf_bytes_list:
            try:
                extracted_images = self.img_mgr.pdf_to_images(pdf_bytes)
                # 使用 base64 存储提取出的图片，以便和其他 URL 格式统一处理
                import base64
                for img in extracted_images:
                    b64 = base64.b64encode(img).decode()
                    pdf_extracted_urls.append(f"base64://{b64}")
            except Exception as e:
                logger.error(f"Failed to extract images from PDF: {e}")

        uid = norm_id(event.get_sender_id())
        gid = norm_id(event.get_group_id())

        uid = norm_id(event.get_sender_id())
        gid = norm_id(event.get_group_id())

        # 2. 获取上下文中的图片
        session_id = event.unified_msg_origin
        image_sources = await self._collect_images_from_context(
            session_id,
            count=self._context_rounds,
            include_bot=True,
            sender_id=uid
        )
        cached_image_sources = await self._get_recent_generated_image_sources(session_id, max_images=max_images)

        if not image_sources and not current_urls and not pdf_extracted_urls and not cached_image_sources:
            return self._finalize_llm_tool_result(
                "[TOOL_FAILED] 上下文里没找到图片或PDF。\n请用你自己的语气让用户先发图再试。")

        # 3. 收集所有图片URL（统一过滤头像、去重并按顺序整理）
        # 修复：当前消息里已经明确带图时，不再混入会话缓存图片，否则会把历史图一起算进去
        merged_current_urls = list(pdf_extracted_urls) + list(current_urls)
        if not merged_current_urls:
            merged_current_urls = list(cached_image_sources)
        all_image_urls = self._merge_batch_image_urls(merged_current_urls, image_sources, max_images=max_images)
        total_images = len(all_image_urls)
        if total_images == 0:
            return self._finalize_llm_tool_result(
                "[TOOL_FAILED] 没找到有效的图片。\n请用你自己的语气告诉用户没找到图，发几张再来。")

        # 3. 计算预设和追加规则
        final_prompt, preset_name, extra_rules = self._process_prompt_and_preset(prompt)

        # 4. 检查配额（批量任务需要足够的次数）
        gid = norm_id(event.get_group_id())
        total_cost = total_images

        deduction = await self._check_quota(event, uid, gid, total_cost)
        if not deduction["allowed"]:
            return self._finalize_llm_tool_result(
                "[TOOL_FAILED] 用户次数不足，无法完成。"
                "请用你自己平时的语气告诉用户现在搞不了，"
                "别提'次数''配额''限制'等词汇，随口带过就好，可以让TA联系管理员或等等再来。"
            )

        # 4.1 更新冷却时间
        self._update_image_cooldown(uid)

        # 根据配置决定是否隐藏进度提示（白名单用户和普通用户使用同一开关）
        show_llm_progress = self._get_conf_bool("llm_show_progress", True)
        hide_llm_result_text = True

        # 5. 发送开始提示
        if show_llm_progress:
            feedback = self._build_llm_progress_text(
                "batch", preset_name=preset_name, total_images=total_images, extra_request=extra_rules
            )
            await event.send(event.chain_result([Plain(feedback)]))

        # 6. 扣费
        if deduction["source"] == "user":
            await self.data_mgr.decrease_user_count(uid, total_cost)
        elif deduction["source"] == "group":
            await self.data_mgr.decrease_group_count(gid, total_cost)

        await self._begin_session_task(event.unified_msg_origin, "批量处理任务", total_images)

        # 7. 启动批量处理任务
        async def process_all():
            success_count = 0
            fail_count = 0
            failed_details = []  # 记录失败详情
            max_retries = self.conf.get("batch_retries", 2)
            pdf_result_images = []  # 如果要打PDF，这里存最后生成的 bytes

            for i, url in enumerate(all_image_urls, 1):
                try:
                    # 下载图片
                    img_bytes = await self.img_mgr.load_bytes(url)
                    if not img_bytes:
                        error_msg = "图片下载失败，可能是链接已过期或网络问题"
                        logger.error(f"Batch process image {i} download failed: {url}")
                        failed_details.append({
                            "index": i,
                            "reason": error_msg,
                            "url_preview": url[:50] + "..." if len(url) > 50 else url
                        })
                        fail_count += 1
                        await self._update_session_task_progress(
                            event.unified_msg_origin, current=i, success=success_count, fail=fail_count
                        )
                        # 发送单条失败提示（仅调试模式显示）
                        if self._should_show_debug_errors():
                            await event.send(event.chain_result([
                                Plain(f"第 {i}/{total_images} 张没弄好: {error_msg}")
                            ]))
                        continue

                    # 处理单张图片（带重试机制）
                    retry_count = 0
                    success = False
                    error_msg = ""

                    while retry_count <= max_retries:
                        if output_as_pdf:
                            # 为 PDF 输出时，隐藏每张的成功提示，只存字节
                            # 因此我们需要在这里单独写获取单图生成的逻辑，或者让 single_batch_task 返回 bytes
                            # 考虑到原有架构，这里写个临时内部函数以复用生成逻辑
                            try:
                                images = [img_bytes]
                                if preset_name != "自定义" and self.conf.get("enable_preset_ref_images", True):
                                    ref_images = await self._load_preset_ref_images(preset_name)
                                    if ref_images:
                                        images = ref_images + images
                                model = ""
                                res = await self.api_mgr.call_api(images, final_prompt, model, False,
                                                                  self.img_mgr.proxy)
                                if isinstance(res, bytes):
                                    res = await self._prepare_send_image_bytes(res)
                                    pdf_result_images.append(res)
                                    await self.data_mgr.record_usage(uid, gid)
                                    await self._register_generation_success(event.unified_msg_origin, 1)
                                    await self._register_generated_image(event.unified_msg_origin, res)
                                    success = True
                                    error_msg = ""
                                else:
                                    success = False
                                    error_msg = self._translate_error_to_chinese(res)
                            except Exception as e:
                                success = False
                                error_msg = self._resolve_debug_error_message(e, "图片生成失败")
                        else:
                            success, error_msg = await self._run_single_batch_task(
                                event=event,
                                image_bytes=img_bytes,
                                prompt=final_prompt,
                                preset_name=preset_name,
                                task_index=i,
                                total_tasks=total_images,
                                uid=uid,
                                gid=gid,
                                extra_rules=extra_rules,
                                image_source=url,
                                hide_text=hide_llm_result_text
                            )

                        if success:
                            break

                        retry_count += 1
                        if retry_count <= max_retries:
                            if self._should_show_debug_errors():
                                await event.send(event.chain_result([
                                    Plain(
                                        f"⚠️ 第 {i}/{total_images} 张图片生成失败 ({error_msg})\n⏳ 正在进行第 {retry_count} 次重试...")
                                ]))
                            await asyncio.sleep(1.5)

                    if success:
                        success_count += 1
                    else:
                        fail_count += 1
                        failed_details.append({
                            "index": i,
                            "reason": error_msg,
                            "url_preview": url[:50] + "..." if len(url) > 50 else url
                        })
                        # 发送单条失败提示（仅调试模式显示）
                        if self._should_show_debug_errors():
                            await event.send(event.chain_result([
                                Plain(f"第 {i}/{total_images} 张最终没弄好: {error_msg}")
                            ]))

                    await self._update_session_task_progress(
                        event.unified_msg_origin, current=i, success=success_count, fail=fail_count
                    )

                    # 添加短暂延迟，避免API限流
                    if i < total_images:
                        await asyncio.sleep(0.5)

                except Exception as e:
                    error_msg = self._resolve_debug_error_message(e, "这次没弄好，请稍后再试。")
                    logger.error(f"Batch process image {i} exception: {e}", exc_info=True)
                    failed_details.append({
                        "index": i,
                        "reason": error_msg,
                        "url_preview": url[:50] + "..." if len(url) > 50 else url
                    })
                    fail_count += 1
                    await self._update_session_task_progress(
                        event.unified_msg_origin, current=i, success=success_count, fail=fail_count
                    )
                    if self._should_show_debug_errors():
                        await event.send(event.chain_result([
                            Plain(f"第 {i}/{total_images} 张出了点问题: {error_msg}")
                        ]))

            if output_as_pdf:
                if pdf_result_images:
                    try:
                        pdf_bytes_result = self.img_mgr.images_to_pdf(pdf_result_images)
                        if pdf_bytes_result:
                            import uuid
                            filename = f"batch_output_{uuid.uuid4().hex[:6]}.pdf"
                            # astrbot 的文件组件支持直接发字节(通过File类)，但在标准组件中未必完善，
                            # 这里我们将其写入到临时目录或者缓存里再发送
                            import os
                            from astrbot.core.message.components import File
                            tmp_path = os.path.join(self.data_mgr.data_dir, filename)
                            with open(tmp_path, "wb") as f:
                                f.write(pdf_bytes_result)
                            await event.send(event.chain_result([File(name=filename, file=tmp_path)]))
                            await event.send(event.chain_result([Plain("这批图我已经先整理成册发你了。")]))
                            # 打包成功后清除缓存，防止旧图污染下次打包
                            await self._clear_session_image_cache(event.unified_msg_origin)
                    except Exception as e:
                        logger.error(f"打包 PDF 失败: {e}")
                        if self._should_show_debug_errors():
                            await event.send(event.chain_result([Plain(f"PDF 打包出了点问题: {e}")]))
                else:
                    if self._should_show_debug_errors():
                        await event.send(event.chain_result([Plain("抱歉，这批图都没弄好，打包不了 PDF。")]))
            elif show_llm_progress:
                summary = "这批图我先整理完了，能发出来的都已经给你了。"
                await event.send(event.chain_result([Plain(summary)]))

        # 启动异步任务
        async def wrapped_process_all():
            try:
                await process_all()
            finally:
                await self._finish_session_task(event.unified_msg_origin)

        asyncio.create_task(wrapped_process_all())

        return self._finalize_llm_tool_success(
            f"[TOOL_SUCCESS] 批量处理任务已启动，共 {total_images} 张图片，预设：{preset_name}。图片将陆续发出，请用自然语言告知用户稍等即可，不要用'生成'等机械词汇。")

    @filter.llm_tool(name="shoubanhua_batch_process_concurrent")
    async def batch_process_concurrent_tool(self, event: AstrMessageEvent, prompt: str, max_images: int = 10,
                                            concurrency: int = 3, output_as_pdf: bool = False):
        '''并发批量处理上下文中的多张图片，也可以将结果打包为PDF输出。仅在用户明确要求快速批量处理时才调用。

        调用前请严格判断：
        1. 用户是否明确要求批量处理/全部处理多张图片？或者用户是否要求将图片打包成PDF输出？
        2. 上下文中是否确实有多张图片或PDF需要处理？
        3. 如果用户只是发送了多张图片但没有要求处理，请不要调用此工具

        此工具会消耗用户大量使用次数（每张图片消耗1次），请谨慎调用。

        Args:
            prompt(string): 图片处理的提示词，可以是预设名+追加规则
            max_images(int): 最多处理的图片数量，默认10张
            concurrency(int): 并发数量，默认3（同时处理3张图片）
            output_as_pdf(boolean): 用户是否明确要求将生成的多张图片打包成PDF输出，默认False
        '''
        # 读取配置中的限制，强制覆盖 LLM 参数
        conf_max_images = self.conf.get("batch_max_images", 10)
        conf_concurrency = self.conf.get("batch_concurrency", 3)
        max_images = min(max_images, conf_max_images) if max_images > 0 else conf_max_images
        concurrency = max(1, conf_concurrency)

        # 0. 检查 LLM 工具开关
        if not self._get_conf_bool("enable_llm_tool", True):
            return self._finalize_llm_tool_result(
                "[TOOL_UNAVAILABLE] 当前功能暂时不可用。请用自己平时的语气告诉用户现在不方便，别提功能禁用、指令模式等词汇，要自然。")

        active_task = await self._get_active_session_task(event.unified_msg_origin)
        if active_task:
            return self._finalize_llm_tool_result(f"[TOOL_SUCCESS] {self._build_active_task_reply(active_task)}")

        # 0.1 检查图片生成冷却时间
        uid = norm_id(event.get_sender_id())
        in_cooldown, remaining = self._check_image_cooldown(uid)
        if in_cooldown:
            # 返回借口让LLM用自然语言拒绝
            excuse = self._get_cooldown_excuse(remaining)
            return f"【冷却中】{excuse}\n\n请用自然的方式告诉用户现在不方便处理图片，可以稍后再试。不要直接说'冷却'这个词。"

        # 1. 提取当前消息的图片 URL（包括引用消息中的图片）
        msg_info = self._extract_message_info(event)
        current_urls = msg_info.get("image_urls", [])

        # 1.5 提取当前消息的 PDF（转化为图片）
        current_pdf_bytes_list = await self.img_mgr.extract_pdfs_from_event(event)
        pdf_extracted_urls = []
        for pdf_bytes in current_pdf_bytes_list:
            try:
                extracted_images = self.img_mgr.pdf_to_images(pdf_bytes)
                import base64
                for img in extracted_images:
                    b64 = base64.b64encode(img).decode()
                    pdf_extracted_urls.append(f"base64://{b64}")
            except Exception as e:
                logger.error(f"Failed to extract images from PDF: {e}")

        # 2. 获取上下文中的图片
        session_id = event.unified_msg_origin
        image_sources = await self._collect_images_from_context(
            session_id,
            count=self._context_rounds,
            include_bot=True,
            sender_id=uid
        )
        cached_image_sources = await self._get_recent_generated_image_sources(session_id, max_images=max_images)

        if not image_sources and not current_urls and not pdf_extracted_urls and not cached_image_sources:
            return self._finalize_llm_tool_result(
                "[TOOL_FAILED] 上下文里没找到图片或PDF。\n请用你自己的语气让用户先发图再试。")

        # 3. 收集所有图片URL（统一过滤头像、去重并按顺序整理）
        # 修复：当前消息里已经明确带图时，不再混入会话缓存图片，否则会把历史图一起算进去
        merged_current_urls = list(pdf_extracted_urls) + list(current_urls)
        if not merged_current_urls:
            merged_current_urls = list(cached_image_sources)
        all_image_urls = self._merge_batch_image_urls(merged_current_urls, image_sources, max_images=max_images)
        total_images = len(all_image_urls)
        if total_images == 0:
            return self._finalize_llm_tool_result(
                "[TOOL_FAILED] 没找到有效的图片。\n请用你自己的语气告诉用户没找到图，发几张再来。")

        # 3. 计算预设和追加规则
        final_prompt, preset_name, extra_rules = self._process_prompt_and_preset(prompt)

        # 4. 检查配额
        gid = norm_id(event.get_group_id())
        total_cost = total_images

        deduction = await self._check_quota(event, uid, gid, total_cost)
        if not deduction["allowed"]:
            return self._finalize_llm_tool_result(
                "[TOOL_FAILED] 用户次数不足，无法完成。"
                "请用你自己平时的语气告诉用户现在搞不了，"
                "别提'次数''配额''限制'等词汇，随口带过就好，可以让TA联系管理员或等等再来。"
            )

        # 4.1 更新冷却时间
        self._update_image_cooldown(uid)

        # 根据配置决定是否隐藏进度提示（白名单用户和普通用户使用同一开关）
        show_llm_progress = self._get_conf_bool("llm_show_progress", True)
        hide_llm_result_text = True

        # 5. 发送开始提示
        if show_llm_progress:
            feedback = self._build_llm_progress_text(
                "batch", preset_name=preset_name, total_images=total_images,
                extra_request=extra_rules, concurrency=concurrency
            )
            await event.send(event.chain_result([Plain(feedback)]))

        # 6. 扣费
        if deduction["source"] == "user":
            await self.data_mgr.decrease_user_count(uid, total_cost)
        elif deduction["source"] == "group":
            await self.data_mgr.decrease_group_count(gid, total_cost)

        await self._begin_session_task(event.unified_msg_origin, "并发批量处理任务", total_images)

        # 7. 使用信号量控制并发
        semaphore = asyncio.Semaphore(concurrency)
        results = {"success": 0, "fail": 0}
        failed_details = []
        pdf_result_images_dict = {}  # 用于保证并发生成的图片顺序
        results_lock = asyncio.Lock()
        max_retries = self.conf.get("batch_retries", 2)

        async def process_single(index: int, url: str):
            async with semaphore:
                try:
                    # 下载图片
                    img_bytes = await self.img_mgr.load_bytes(url)
                    if not img_bytes:
                        error_msg = "图片下载失败，可能是链接已过期或网络问题"
                        logger.error(f"Concurrent batch process image {index} download failed: {url}")
                        async with results_lock:
                            results["fail"] += 1
                            failed_details.append({
                                "index": index,
                                "reason": error_msg,
                                "url_preview": url[:50] + "..." if len(url) > 50 else url
                            })
                        await self._update_session_task_progress(
                            event.unified_msg_origin, current=index, success=results["success"], fail=results["fail"]
                        )
                        if self._should_show_debug_errors():
                            await event.send(event.chain_result([
                                Plain(f"第 {index}/{total_images} 张没弄好: {error_msg}")
                            ]))
                        return

                    # 处理单张图片（带重试机制）
                    retry_count = 0
                    success = False
                    error_msg = ""

                    while retry_count <= max_retries:
                        if output_as_pdf:
                            try:
                                images = [img_bytes]
                                if preset_name != "自定义" and self.conf.get("enable_preset_ref_images", True):
                                    ref_images = await self._load_preset_ref_images(preset_name)
                                    if ref_images:
                                        images = ref_images + images
                                model = ""
                                res = await self.api_mgr.call_api(images, final_prompt, model, False,
                                                                  self.img_mgr.proxy)
                                if isinstance(res, bytes):
                                    res = await self._prepare_send_image_bytes(res)
                                    async with results_lock:
                                        pdf_result_images_dict[index] = res
                                    await self.data_mgr.record_usage(uid, gid)
                                    await self._register_generation_success(event.unified_msg_origin, 1)
                                    await self._register_generated_image(event.unified_msg_origin, res)
                                    success = True
                                    error_msg = ""
                                else:
                                    success = False
                                    error_msg = self._translate_error_to_chinese(res)
                            except Exception as e:
                                success = False
                                error_msg = self._resolve_debug_error_message(e,self._translate_error_to_chinese(str(e)))
                        else:
                            success, error_msg = await self._run_single_batch_task(
                                event=event,
                                image_bytes=img_bytes,
                                prompt=final_prompt,
                                preset_name=preset_name,
                                task_index=index,
                                total_tasks=total_images,
                                uid=uid,
                                gid=gid,
                                extra_rules=extra_rules,
                                image_source=url,
                                hide_text=hide_llm_result_text
                            )

                        if success:
                            break

                        retry_count += 1
                        if retry_count <= max_retries:
                            if self._should_show_debug_errors():
                                await event.send(event.chain_result([
                                    Plain(
                                        f"⚠️ 第 {index}/{total_images} 张图片生成失败 ({error_msg})\n⏳ 正在进行第 {retry_count} 次重试...")
                                ]))
                            await asyncio.sleep(1.5)

                    async with results_lock:
                        if success:
                            results["success"] += 1
                        else:
                            results["fail"] += 1
                            failed_details.append({
                                "index": index,
                                "reason": error_msg,
                                "url_preview": url[:50] + "..." if len(url) > 50 else url
                            })
                            if self._should_show_debug_errors():
                                await event.send(event.chain_result([
                                    Plain(f"第 {index}/{total_images} 张最终没弄好: {error_msg}")
                                ]))
                        await self._update_session_task_progress(
                            event.unified_msg_origin, current=index, success=results["success"], fail=results["fail"]
                        )

                except Exception as e:
                    error_msg = self._translate_error_to_chinese(str(e))
                    logger.error(f"Concurrent batch process image {index} exception: {e}", exc_info=True)
                    async with results_lock:
                        results["fail"] += 1
                        failed_details.append({
                            "index": index,
                            "reason": error_msg,
                            "url_preview": url[:50] + "..." if len(url) > 50 else url
                        })
                    await self._update_session_task_progress(
                        event.unified_msg_origin, current=index, success=results["success"], fail=results["fail"]
                    )
                    if self._should_show_debug_errors():
                        await event.send(event.chain_result([
                            Plain(f"第 {index}/{total_images} 张出了点问题: {error_msg}")
                        ]))

        async def process_all():
            # 创建所有任务
            tasks = [
                process_single(i, url)
                for i, url in enumerate(all_image_urls, 1)
            ]

            # 等待所有任务完成
            await asyncio.gather(*tasks)

            if output_as_pdf:
                if pdf_result_images_dict:
                    try:
                        # 按原有顺序重组图片
                        ordered_images = [pdf_result_images_dict[k] for k in sorted(pdf_result_images_dict.keys())]
                        pdf_bytes_result = self.img_mgr.images_to_pdf(ordered_images)
                        if pdf_bytes_result:
                            import uuid
                            import os
                            from astrbot.core.message.components import File
                            filename = self._build_pdf_filename_hint(
                                prompt=extra_rules or prompt,
                                preset_name=preset_name,
                                count=results["success"]
                            )
                            tmp_path = os.path.join(self.data_mgr.data_dir, filename)
                            with open(tmp_path, "wb") as f:
                                f.write(pdf_bytes_result)
                            await event.send(event.chain_result([File(name=filename, file=tmp_path)]))

                            await event.send(event.chain_result([
                                Plain(f"这批图我已经整理成册发你了，共 {results['success']} 张。")
                            ]))
                            # 打包成功后清除缓存，防止旧图污染下次打包
                            await self._clear_session_image_cache(event.unified_msg_origin)
                    except Exception as e:
                        logger.error(f"打包 PDF 失败: {e}")
                        if self._should_show_debug_errors():
                            await event.send(event.chain_result([Plain(f"PDF 打包出了点问题: {e}")]))
                else:
                    if self._should_show_debug_errors():
                        await event.send(event.chain_result([Plain("抱歉，这批图都没弄好，打包不了 PDF。")]))
            elif show_llm_progress:
                # 发送完成汇总
                summary = "这批图我先整理完了，能发出来的都已经给你了。"
                await event.send(event.chain_result([Plain(summary)]))

        # 启动异步任务
        async def wrapped_process_all():
            try:
                await process_all()
            finally:
                await self._finish_session_task(event.unified_msg_origin)

        asyncio.create_task(wrapped_process_all())

        if output_as_pdf:
            return self._finalize_llm_tool_success(
                f"[TOOL_SUCCESS] 并发批量处理任务已启动，共 {total_images} 张图片，预设：{preset_name}。"
                "这次不会一张张发出，我会等整批都完成后直接整理成一个 PDF 发给用户。"
                "请用自然语言告诉用户稍等即可，不要用'生成'等机械词汇。"
            )

        return self._finalize_llm_tool_success(
            f"[TOOL_SUCCESS] 并发批量处理任务已启动，共 {total_images} 张图片，预设：{preset_name}。"
            "图片将陆续发出，请用自然语言告知用户稍等即可，不要用'生成'等机械词汇。"
        )

    # ================= 日常人设功能 =================

    @filter.llm_tool(name="shoubanhua_persona_photo")
    async def persona_photo_tool(self, event: AstrMessageEvent, scene_hint: str = "", extra_request: str = "",
                                 count: int = 1):
        '''生成Bot人设角色（你自己）的日常照片或写真。

        【唯一指定用途】
        只要用户是要求看**你的 / Bot本人 / 人设角色**的新照片、自拍、写真、露脸、外貌、穿搭、换装、cos、动作展示或与参考图合影，
        无论要求多少张，都【必须且只能】使用此工具，绝对不能使用 shoubanhua_draw_image。

        【常见人设拍照意图（不要死记原句，要按语义判断）】
        - 自拍/拍照：自拍、拍个照、来张照片、发张近照、拍几张、营业一下、出镜看看、拍给我看
        - 看本人/外貌：看看你、想看你、让我看看你、你长什么样、露脸、看脸、看穿搭、看今天的样子
        - 写真/日常：写真、写真集、生活照、日常照、私房照、全身照、半身照、头像照、证件照
        - 换装/cos：要求你/Bot本人/人设角色穿某件衣服、换一套、试穿、cos某角色、扮演某角色等
        - 动作/道具/场景：抱着猫、拿着花、戴帽子、比心、wink、坐在咖啡店、公园、卧室、夜景等
        - 参考图/合影：和上图的人合影、跟这张图同框、照着上面图片的衣服/姿势/氛围来，但人设身份必须保持 Bot 本人

        【不要调用本工具的情况】
        - 用户只是闲聊、问你在干嘛，但没有明确表示要看照片/样子/自拍/出镜
        - 用户明确要求画别人、生成一个角色/物体、处理用户图片，且没有要求 Bot 本人入镜
        - 用户只说“某角色cos某角色给我看”“画一个cos某角色的人”等第三方人物创作，应该使用 shoubanhua_draw_image

        【重要】调用条件（请严格遵守）：
        1. 用户明确要求看 Bot 本人照片、自拍、外貌、写真、换装、cos、动作展示或与参考图合影时才调用
        2. 用户只是问"你在干嘛"、"你在做什么"、"闲聊" → 用文字回答即可，不要发照片
        3. 没有明确表达想看照片意愿 → 不要主动发照片

        【数量控制（极度重要！）】
        - 默认只生成1张，除非用户明确说出了具体数字（如"来5张""拍3张"）。
        - 用户说"打包成PDF"、"发个PDF"并不等于要多张！只说打包不说数量时 count 必须保持 1。
        - 只有"写真集"、"多来点"、"多拍几张"等明确要多张的表述才设置 count=3。
        - 坚决不要为了表现热情而擅自设置大量 count！

        【上下文连续性（非常重要！）】
        当用户在同一次对话中连续请求照片时，你必须结合之前的上下文判断：
        - 用户说"再来一张"、"换一个"、"换个姿势"、"继续" → 继承上一次的场景和主题，只修改用户提到的变化部分
        - 例如：上一轮是"睡觉"，用户说"换个姿势再来一张" → scene_hint 保持"卧室"，extra_request 写"换一个不同的睡姿"
        - 例如：上一轮是"在咖啡店"，用户说"换成公园" → scene_hint 改为"公园"，保留之前的额外要求
        - 例如：上一轮是"穿红裙子"，用户说"换成蓝裙子" → extra_request 改为"穿蓝裙子"，场景不变
        - 绝对不要丢失上下文！如果用户没有明确改变场景，就保持上次的场景

        Args:
            scene_hint(string): 场景提示（可选），如"咖啡店"、"公园"等，用于匹配预设场景
            extra_request(string): 用户的额外要求（可选），如"穿红色衣服"、"微笑"等
            count(int): 生成图片的数量，默认1张，最大10张。只有用户明确要求多张时才增大。
        '''
        # 0. 检查功能开关
        if not self._persona_mode:
            return self._finalize_llm_tool_result(
                "[TOOL_UNAVAILABLE] 人设功能当前未启用。请用自己平时的语气告诉用户现在不方便，别提配置开关等技术词汇。")

        if not self._get_conf_bool("enable_llm_tool", True):
            return self._finalize_llm_tool_result(
                "[TOOL_UNAVAILABLE] 当前功能暂时不可用。请用自己平时的语气告诉用户现在不方便，别提功能禁用、指令模式等词汇，要自然。")

        # 0.1 检查图片生成冷却时间
        uid = norm_id(event.get_sender_id())
        persona_id = int(self.conf.get("current_persona", 1))
        user_text = event.message_str.strip()

        # 0.2 人设名自动匹配：用户消息中包含某个人设名时，自动切换到该人设
        # 例如 "mm,看看你的跑步照片" → 自动使用名为 mm 的人设参考图
        persona_list = self.conf.get("persona_list", [])
        if persona_list:
            for idx, p in enumerate(persona_list):
                p_name = str(p.get("persona_name", "")).strip()
                if p_name and p_name in user_text:
                    detected_id = idx + 1
                    if detected_id != persona_id:
                        logger.info(f"[人设拍照] 🔀 人设名匹配: 检测到 [{p_name}]，自动从人设#{persona_id} 切换到人设#{detected_id}")
                        persona_id = detected_id
                        self.conf["current_persona"] = persona_id
                    break

        logger.info(f"[人设拍照] 触发 | 人设#{persona_id} | 用户: {uid} | 消息: {user_text[:80]} | LLM参数: scene_hint={scene_hint}, extra_request={extra_request[:50] if extra_request else ''}, count={count}")

        in_cooldown, remaining = self._check_image_cooldown(uid)
        if in_cooldown:
            excuse = self._get_cooldown_excuse(remaining)
            logger.info(f"[人设拍照] 冷却中 | 剩余: {remaining}秒")
            return f"【冷却中】{excuse}\n\n请用自然的方式告诉用户现在不方便拍照，可以稍后再试。不要直接说'冷却'这个词。"

        # 1. 加载人设参考图
        ref_images = await self._load_persona_ref_images()
        if not ref_images:
            logger.info(f"[人设拍照] 人设#{persona_id} 无参考图，终止")
            return self._finalize_llm_tool_result(
                "[TOOL_UNAVAILABLE] 人设参考图还没配置好。请用自己平时的语气告诉用户现在还没办法，不要提指令或命令。")
        logger.info(f"[人设拍照] 人设#{persona_id} 加载 {len(ref_images)} 张参考图")

        # 1.2 预设命中检查
        preset_parts = user_text.split(maxsplit=1)
        preset_cmd = preset_parts[0].lstrip("#/＃")
        matched_preset_prompt = self.data_mgr.get_prompt(preset_cmd)
        if matched_preset_prompt:
            logger.info(f"[人设拍照] ✅ 命中预设 [{preset_cmd}] | 人设#{persona_id} | 图片: {len(ref_images)}张参考图 + 用户图 | 路径: 预设直出")
            final_prompt = matched_preset_prompt
            if len(preset_parts) > 1:
                final_prompt += " " + preset_parts[1]
            final_prompt = self._append_preset_safety_suffix(final_prompt, preset_cmd)

            bot_id = self._get_bot_id(event)
            extra_images = await self.img_mgr.extract_images_from_event(
                event, ignore_id=bot_id, context=None, include_at_avatar=False
            )
            all_images = ref_images + extra_images

            count_match = re.search(r'(\d+)\s*[张个份]', user_text)
            req_count = int(count_match.group(1)) if count_match else 1
            req_count = max(1, min(req_count, self._get_active_persona().get("persona_photo_max_count", 5)))

            # 配额检查
            uid = norm_id(event.get_sender_id())
            gid = norm_id(event.get_group_id())
            deduction = await self._check_quota(event, uid, gid, req_count)
            if not deduction["allowed"]:
                return self._finalize_llm_tool_result(
                    "[TOOL_FAILED] 配额不足。用自己的话告诉用户今天的额度用完了。")
            self._update_image_cooldown(uid)

            self._log_prompt_preview(f"persona_preset:{preset_cmd}", final_prompt)
            await self._register_pending_generation(event.unified_msg_origin, req_count)

            if req_count == 1:
                success, error_msg = await self._run_background_task(
                    event, all_images, final_prompt, preset_cmd, deduction, uid, gid, 1,
                    suppress_user_error=True
                )
                if success:
                    logger.info(f"[人设拍照] ✅ 成功 | 人设#{persona_id} | 预设=[{preset_cmd}] | 1张")
                    return self._finalize_llm_tool_success()
                logger.info(f"[人设拍照] ❌ 失败 | 人设#{persona_id} | 预设=[{preset_cmd}] | 错误: {error_msg[:100]}")
                return self._build_llm_tool_failure(error_msg)
            else:
                batch_result = await self._run_batch_image_to_image(
                    event, all_images, final_prompt, preset_cmd, deduction, uid, gid, req_count,
                    suppress_user_error=True
                )
                total_ok = batch_result.get("success", 0)
                if total_ok > 0:
                    return self._finalize_llm_tool_success()
                return self._build_llm_tool_failure(f"{req_count}张生成失败")

        logger.info(f"[人设拍照] 未命中预设 [{preset_cmd}] | 路径: LLM优化 | scene_hint={scene_hint}")

        # 1.5 提取用户可能提供的参考图片（如衣服款式、姿势参考等）
        # 默认只取当前消息/已展开的引用图片，避免把上一轮生成图误当成“用户补充参考图”。
        user_images = []
        bot_id = self._get_bot_id(event)
        requested_text = " ".join([str(scene_hint or ""), str(extra_request or "")]).strip()
        _ref_intent_keywords = [
            "同款", "这件", "那件", "这套", "那套", "这身", "那身",
            "穿这个", "穿那个", "照着", "参考", "参考图", "模仿",
            "一样的", "跟这个", "按这个", "合影", "同框", "一起拍",
            "上面", "上图", "上一张", "刚才", "刚刚", "这张", "这图",
            "这个图", "图片里的人", "照片里的人", "图里的人", "这个人", "那个人",
        ]
        _has_ref_intent = any(kw in requested_text for kw in _ref_intent_keywords)
        user_images = await self.img_mgr.extract_images_from_event(
            event, ignore_id=bot_id, context=None, include_at_avatar=False
        )
        user_ref_source = "current_or_expanded_reply" if user_images else ""

        # 当前消息没有图片时，仅在用户明确说“上图/这张/参考/合影/同款”等引用意图时才回溯上下文。
        # “cos 某角色给我看”不等于用户补充了参考图，否则会误捞上一轮生成图。
        if not user_images and _has_ref_intent:
            user_images = await self.img_mgr.extract_images_from_event(
                event, ignore_id=bot_id, context=self.context, include_at_avatar=False
            )
            if user_images:
                user_ref_source = "reply_fetch_by_explicit_reference"

        if not user_images and _has_ref_intent:
            session_id = event.unified_msg_origin
            context_messages_full = await self.ctx_mgr.get_recent_messages(session_id, count=self._context_rounds)
            if context_messages_full:
                last_img_msg = self.ctx_mgr.get_last_image_message(context_messages_full)
                if last_img_msg and last_img_msg.image_urls:
                    for url in last_img_msg.image_urls:
                        img_bytes = await self.img_mgr.load_bytes(url)
                        if img_bytes:
                            user_images.append(img_bytes)
                if user_images:
                    user_ref_source = "recent_context_by_explicit_reference"

        # 合并图片：人设参考图在前，用户参考图在后
        final_images = ref_images + user_images
        if user_images:
            logger.info(f"[人设拍照] 用户补充参考图 {len(user_images)} 张 (source={user_ref_source})")

        # 2. 匹配场景
        user_input_text = event.message_str.strip()
        llm_text = " ".join([str(scene_hint or ""), str(extra_request or "")]).strip()
        match_text = f"{user_input_text} {llm_text}"
        scene_name, scene_prompt = self._match_persona_scene(match_text)
        scene_matched = bool(scene_name)

        # 未命中场景时：LLM 的 scene_hint 优先，再用默认提示词补充风格
        if not scene_matched:
            persona = self._get_active_persona()
            default_prompt = (persona.get("persona_default_prompt", "") or "").strip()
            if scene_hint:
                scene_prompt = scene_hint
                if default_prompt:
                    scene_prompt += f" {default_prompt}"
            else:
                scene_prompt = default_prompt

        logger.info(f"[人设拍照] 场景匹配: {'命中 [' + scene_name + ']' if scene_matched else '未命中 → 兜底'} | 人设#{persona_id} | 图片: {len(final_images)}张 ({len(ref_images)}参考+{len(user_images)}用户)")

        # 3. 构建完整提示词
        effective_extra = "" if scene_matched else extra_request
        full_prompt = self._build_persona_prompt(scene_prompt, effective_extra, scene_matched=scene_matched)
        full_prompt += " " + self._build_current_time_persona_hint()
        if user_images:
            persona_ref_text = " ".join([str(scene_hint or ""), str(extra_request or "")])
            if "合影" in persona_ref_text:
                full_prompt += (
                    " The persona reference image is the main character and must keep identity priority."
                    " The additional user reference image shows the other person to appear in the photo."
                    " Create a natural group photo with both the persona character and the referenced person."
                    " Do NOT replace the persona character's face, hairstyle, body identity, or overall persona with the user reference image."
                )
            else:
                full_prompt += (
                    " Use the additional user reference image only for outfit, accessories, pose, composition, or atmosphere reference."
                    " If the user asks for same outfit, reproduce the same clothing design and matching style as closely as possible."
                    " The persona reference has absolute priority for identity consistency."
                    " Do NOT replace the character's face, hairstyle, body identity, or overall persona with the user reference image."
                    " In short: keep the persona character unchanged, only borrow the requested clothing or styling details from the user reference."
                )
            self._log_prompt_preview(f"persona:{scene_name}", full_prompt)

        # 5. 数量决策：
        # - 如果用户文本里明确写了数量，优先按用户文本；
        # - 否则若 LLM 工具已经显式传入 count>1，则信任这次工具调用；
        # - 最后才根据“写真集/多来几张”等自然语言补推断。
        # 这样可以避免因为 extra_request 只保留了部分内容（如只剩“穿着睡衣”）而把
        # 已经由 LLM 明确决定好的批量参数错误降回 1 张。
        explicit_count = self._extract_explicit_requested_count_from_text(requested_text)
        incoming_count = max(1, int(count or 1))

        if explicit_count is not None:
            requested_count = explicit_count
            if incoming_count != explicit_count:
                logger.info(
                    f"人设拍照：工具传入 count={incoming_count} 与用户显式数量 {explicit_count} 不一致，已按用户文本修正"
                )
        elif incoming_count > 1:
            requested_count = incoming_count
            logger.info(f"人设拍照：用户文本未显式写数量，采用工具传入 count={incoming_count}")
        else:
            requested_count = self._resolve_tool_requested_count(
                requested_text,
                incoming_count=1,
                default=1,
                multi_default=3
            )
            if requested_count > 1:
                logger.info(f"人设拍照：根据用户文本语义推断为多张输出，count={requested_count}")

        # 6. 限制数量（必须先做，避免错误批量参数影响配额检查和分支选择）
        raw_count = requested_count
        count, count_limited = self._normalize_generation_count(requested_count, "persona")

        # 7. 根据配置决定是否发送进度提示
        if self._get_conf_bool("llm_show_progress", True):
            feedback = self._build_llm_progress_text(
                "persona", count=count, scene_name=scene_name,
                extra_request=extra_request, has_user_images=bool(user_images)
            )
            await event.send(event.chain_result([Plain(feedback)]))

        # 8. 检查配额
        gid = norm_id(event.get_group_id())
        deduction = await self._check_quota(event, uid, gid, count)
        if not deduction["allowed"]:
            return self._finalize_llm_tool_result(
                "[TOOL_FAILED] 用户次数不足，无法完成。"
                "请用你自己平时的语气告诉用户现在搞不了，随口带过就好。"
            )

        # 8. 更新冷却时间
        self._update_image_cooldown(uid)

        # 8.1 新任务开始前，若当前会话空闲则清掉上一轮生成缓存，避免后续 PDF 混入旧图
        await self._clear_session_image_cache_if_idle(event.unified_msg_origin)

        # 9. 计算是否隐藏输出文本（白名单用户和普通用户使用同一开关）
        hide_llm_result_text = True

        # 10. 等待拍照任务完成并确认图片已发送，再把成功结果交给二次 LLM 收尾。
        await self._register_pending_generation(event.unified_msg_origin, count)
        if count == 1:
            success, error_msg = await self._run_background_task(
                event=event,
                images=final_images,
                prompt=full_prompt,
                preset_name=f"人设-{scene_name}",
                deduction=deduction,
                uid=uid,
                gid=gid,
                cost=1,
                extra_rules=extra_request,
                hide_text=hide_llm_result_text,
                suppress_user_error=True
            )
            if not success:
                logger.info(f"[人设拍照] ❌ 失败 | 人设#{persona_id} | 场景={scene_name or '兜底'} | 错误: {error_msg[:100]}")
                return self._build_llm_tool_failure(error_msg)
            logger.info(f"[人设拍照] ✅ 成功 | 人设#{persona_id} | 场景={scene_name or '兜底'} | 1张")
            if count_limited:
                return self._finalize_llm_tool_success(
                    f"[TOOL_SUCCESS] 照片已经发给用户。{self._build_count_limit_reply(count, 'persona')} 请自然收尾一句。")
            return self._finalize_llm_tool_success(
                "[TOOL_SUCCESS] 照片已经发给用户。请用你自己的语气自然收尾一句，比如“给你啦，才不是特意拍给你的呢。”不要提系统、工具或生成。")
        else:
            # 对于人设的多张生成，因为传递的是最终合并好的图片（包含人设参考+用户参考），
            # 所以使用 _run_batch_image_to_image。但是要防止该函数再次从数据库读取 "_persona_" 导致图片翻倍。
            # _run_batch_image_to_image 中有逻辑：如果不是“自定义”，就去取预设图片叠加。
            # "人设-xxx" 是不会在预设里查到的，所以不会产生重复！这是完美的。
            batch_result = await self._run_batch_image_to_image(
                event=event,
                images=final_images,
                prompt=full_prompt,
                preset_name=f"人设-{scene_name}",
                deduction=deduction,
                uid=uid,
                gid=gid,
                count=count,
                extra_rules=extra_request,
                hide_text=hide_llm_result_text,
                suppress_user_error=True
            )
            total_success = int(batch_result.get("success", 0))
            total_fail = int(batch_result.get("fail", 0))
            branch_errors = batch_result.get("errors", [])
            if total_success <= 0:
                logger.info(f"[人设拍照] ❌ 全部失败 | 人设#{persona_id} | 场景={scene_name or '兜底'} | {total_fail}张")
                return self._build_llm_tool_failure(branch_errors[0] if branch_errors else "这次没弄好，请稍后再试。")
            logger.info(f"[人设拍照] ✅ 完成 | 人设#{persona_id} | 场景={scene_name or '兜底'} | 成功{total_success}张 失败{total_fail}张")
            if count_limited:
                return self._finalize_llm_tool_success(
                    f"[TOOL_SUCCESS] 照片已经发给用户，成功 {total_success} 张。{self._build_count_limit_reply(count, 'persona')} 请自然收尾一句。")
            if total_fail > 0:
                return self._finalize_llm_tool_success(
                    f"[TOOL_SUCCESS] 照片已经发给用户，成功 {total_success} 张，有 {total_fail} 张没弄好。请自然地随口带过。")
            return self._finalize_llm_tool_success(
                f"[TOOL_SUCCESS] {total_success} 张照片已经发给用户。请用你自己的语气自然收尾一句，比如“给你啦，才不是特意拍给你的呢。”不要提系统、工具或生成。")

    @filter.command("人设拍照", prefix_optional=True)
    async def on_persona_photo_cmd(self, event: AstrMessageEvent, ctx=None):
        """生成人设角色的日常照片（指令模式）

        用法: #人设拍照 [场景] [额外要求]
        示例: #人设拍照 咖啡店 穿白色连衣裙
        """
        if not self._persona_mode:
            yield event.chain_result([Plain("人设功能还没开，联系管理员设置一下吧")])
            return

        # 加载人设参考图
        ref_images = await self._load_persona_ref_images()
        if not ref_images:
            yield event.chain_result([Plain("人设参考图还没配置好，先用 #人设参考图添加 添加几张吧")])
            return

        # 解析参数
        parts = event.message_str.split(maxsplit=2)
        scene_hint = parts[1] if len(parts) > 1 else ""
        extra_request = parts[2] if len(parts) > 2 else ""

        user_images = await self.img_mgr.extract_images_from_event(
            event, ignore_id=self._get_bot_id(event), context=self.context
        )
        ref_text = " ".join([scene_hint, extra_request])
        if not user_images and any(kw in ref_text for kw in ["合影", "上面", "上图", "这张", "这图", "参考", "照着"]):
            context_messages_full = await self.ctx_mgr.get_recent_messages(event.unified_msg_origin, count=self._context_rounds)
            last_img_msg = self.ctx_mgr.get_last_image_message(context_messages_full)
            if last_img_msg and last_img_msg.image_urls:
                for url in last_img_msg.image_urls:
                    img_bytes = await self.img_mgr.load_bytes(url)
                    if img_bytes:
                        user_images.append(img_bytes)
        final_images = ref_images + user_images

        # 获取上下文用于场景匹配
        session_id = event.unified_msg_origin
        context_messages = await self.ctx_mgr.get_recent_messages(session_id, count=10)

        context_text = scene_hint
        if context_messages:
            for msg in context_messages[-5:]:
                if msg.is_bot:
                    context_text += " " + msg.content

        # 匹配场景
        scene_name, scene_prompt = self._match_persona_scene(context_text)

        # 构建提示词
        full_prompt = self._build_persona_prompt(scene_prompt, extra_request)
        full_prompt += " " + self._build_current_time_persona_hint()
        if user_images:
            if "合影" in ref_text:
                full_prompt += (
                    " The persona reference image is the main character and must keep identity priority."
                    " The additional user reference image shows the other person to appear in the photo."
                    " Create a natural group photo with both the persona character and the referenced person."
                )
            else:
                full_prompt += (
                    " Use the additional user reference image only as requested while keeping the persona reference identity unchanged."
                )

        # 检查配额
        uid = norm_id(event.get_sender_id())
        gid = norm_id(event.get_group_id())
        deduction = await self._check_quota(event, uid, gid, 1)
        if not deduction["allowed"]:
            yield event.chain_result([Plain(deduction["msg"])])
            return

        # 发送反馈
        persona_name = self._get_active_persona().get("persona_name", "小助手")
        feedback = f"📸 正在生成 {persona_name} 的照片"
        if scene_name:
            feedback += f"\n🎬 场景: {scene_name}"
        if extra_request:
            feedback += f"\n📝 要求: {extra_request[:30]}..."
        feedback += "\n⏳ 请稍候..."
        yield event.chain_result([Plain(feedback)])

        # 扣费
        if deduction["source"] == "user":
            await self.data_mgr.decrease_user_count(uid, 1)
        elif deduction["source"] == "group":
            await self.data_mgr.decrease_group_count(gid, 1)

        # 调用 API
        model = ""
        start = datetime.now()
        res = await self.api_mgr.call_api(final_images, full_prompt, model, False, self.img_mgr.proxy)

        if isinstance(res, bytes):
            res = await self._prepare_send_image_bytes(res)
            elapsed = (datetime.now() - start).total_seconds()
            await self.data_mgr.record_usage(uid, gid)
            await self._register_generation_success(event.unified_msg_origin, 1)
            await self._register_generated_image(event.unified_msg_origin, res)

            quota_str = self._get_quota_str(deduction, uid, gid)
            timing_text = self._format_success_timing(elapsed)
            info = f"\n✅ 生成成功 ({timing_text})"
            if scene_name:
                info += f" | 场景: {scene_name}"
            info += f" | 剩余: {quota_str}"
            yield event.chain_result([Image.fromBytes(res), Plain(info)])
        else:
            yield event.chain_result([Plain(f"没弄好: {self._resolve_debug_error_message(res, '这次没弄好，请稍后再试。')}")])

    @filter.command("人设参考图添加", aliases={"添加人设图"}, prefix_optional=True)
    async def on_add_persona_ref(self, event: AstrMessageEvent, ctx=None):
        """添加人设参考图（管理员）

        用法: #人设参考图添加 [图片]
        """
        if not self.is_admin(event): return

        # 提取图片
        bot_id = self._get_bot_id(event)
        images = await self.img_mgr.extract_images_from_event(event, ignore_id=bot_id, context=self.context)

        if not images:
            yield event.chain_result([Plain("没检测到图片，发送或引用一下图片再试")])
            return

        # 保存到当前人设
        key = self._get_persona_key()
        count = await self.data_mgr.add_preset_ref_images(key, images)

        if count > 0:
            total = len(self.data_mgr.get_preset_ref_image_paths(key))
            persona_id = int(self.conf.get("current_persona", 1))
            yield event.chain_result([Plain(f"✅ 已添加 {count} 张参考图到人设 {persona_id}\n当前人设共 {total} 张参考图")])
        else:
            yield event.chain_result([Plain("参考图保存失败了，再试试？")])

    @filter.command("人设参考图查看", aliases={"查看人设图"}, prefix_optional=True)
    async def on_view_persona_ref(self, event: AstrMessageEvent, ctx=None):
        """查看人设参考图（管理员）"""
        if not self.is_admin(event): return

        key = self._get_persona_key()
        if not self.data_mgr.has_preset_ref_images(key):
            # 兼容旧版
            if not self.data_mgr.has_preset_ref_images("_persona_"):
                yield event.chain_result([Plain("暂无人设参考图")])
                return

        ref_images = await self._load_persona_ref_images()
        persona_id = int(self.conf.get("current_persona", 1))

        if not ref_images:
            yield event.chain_result([Plain("人设参考图加载失败")])
            return

        result = [Plain(f"📷 人设 {persona_id} 参考图 ({len(ref_images)} 张):\n")]
        for img_bytes in ref_images[:5]:
            result.append(Image.fromBytes(img_bytes))

        if len(ref_images) > 5:
            result.append(Plain(f"\n... 还有 {len(ref_images) - 5} 张未显示"))

        yield event.chain_result(result)

    @filter.command("人设参考图清除", aliases={"清除人设图"}, prefix_optional=True)
    async def on_clear_persona_ref(self, event: AstrMessageEvent, ctx=None):
        """清除所有人设参考图（管理员）"""
        if not self.is_admin(event): return

        key = self._get_persona_key()
        count = await self.data_mgr.clear_preset_ref_images(key)
        # 兼容旧版
        if count == 0:
            count = await self.data_mgr.clear_preset_ref_images("_persona_")

        persona_id = int(self.conf.get("current_persona", 1))
        if count > 0:
            yield event.chain_result([Plain(f"✅ 已清除人设 {persona_id} 的 {count} 张参考图")])
        else:
            yield event.chain_result([Plain("暂无人设参考图")])

    @filter.command("人设场景列表", aliases={"场景列表"}, prefix_optional=True)
    async def on_list_persona_scenes(self, event: AstrMessageEvent, ctx=None):
        """查看所有人设场景"""
        if not self._persona_scene_map:
            yield event.chain_result([Plain("暂无场景配置")])
            return

        msg = f"🎬 人设场景列表 ({len(self._persona_scene_map)} 个):\n"
        for scene_name, prompt in sorted(self._persona_scene_map.items()):
            prompt_preview = prompt[:40] + "..." if len(prompt) > 40 else prompt
            msg += f"\n• {scene_name}: {prompt_preview}"

        default_prompt = (self._get_active_persona().get("persona_default_prompt", "") or "").strip()
        if default_prompt:
            msg += f"\n\n📌 默认场景: {default_prompt[:40]}..."

        yield event.chain_result([Plain(msg)])

    @filter.command("人设状态", prefix_optional=True)
    async def on_persona_status(self, event: AstrMessageEvent, ctx=None):
        """查看人设功能状态（管理员）"""
        if not self.is_admin(event): return

        persona = self._get_active_persona()
        persona_name = persona.get("persona_name", "小助手")
        persona_desc = persona.get("persona_description", "一个可爱的二次元女孩")
        photo_style = persona.get("persona_photo_style", "")
        trigger_keywords = persona.get("persona_trigger_keywords", [])
        current_id = int(self.conf.get("current_persona", 1))

        ref_images = await self._load_persona_ref_images()
        ref_count = len(ref_images) if ref_images else 0

        msg = f"👤 人设功能状态:\n"
        msg += f"启用状态: {'✅ 已启用' if self._persona_mode else '❌ 未启用'}\n"
        msg += f"当前人设: #{current_id}\n"
        msg += f"人设名称: {persona_name}\n"
        msg += f"人设描述: {persona_desc[:50]}{'...' if len(persona_desc) > 50 else ''}\n"
        msg += f"照片风格: {photo_style[:30] if photo_style else '自动跟随参考图'}{'...' if len(photo_style) > 30 else ''}\n"
        msg += f"参考图: {ref_count} 张\n"
        msg += f"场景数: {len(self._persona_scene_map)} 个\n"
        msg += f"触发词: {', '.join(trigger_keywords[:5])}{'...' if len(trigger_keywords) > 5 else ''}"

        yield event.chain_result([Plain(msg)])

    @filter.command("手办化查看人设", prefix_optional=True)
    async def on_list_personas(self, event: AstrMessageEvent, ctx=None):
        """查看所有人设及其参考图数量"""
        if not self.is_admin(event): return

        current_id = int(self.conf.get("current_persona", 1))
        persona_name = self._get_active_persona().get("persona_name", "小助手")

        # 扫描所有人设（_persona_N_ 格式）
        personas = []
        for i in range(1, 20):
            key = f"_persona_{i}_"
            if self.data_mgr.has_preset_ref_images(key):
                count = len(self.data_mgr.get_preset_ref_image_paths(key))
                personas.append((i, count))
        # 兼容旧版 _persona_
        if self.data_mgr.has_preset_ref_images("_persona_"):
            count = len(self.data_mgr.get_preset_ref_image_paths("_persona_"))
            if not any(p[0] == 1 for p in personas):
                personas.insert(0, (1, count))

        if not personas:
            yield event.chain_result([Plain("暂无人设参考图。使用 #人设参考图添加 上传图片。")])
            return

        lines = [f"👤 人设列表 (当前: #{current_id} {persona_name})"]
        for pid, count in personas:
            marker = " ⬅️ 当前" if pid == current_id else ""
            lines.append(f"  人设 {pid}: {count} 张参考图{marker}")
        lines.append(f"\n使用 #手办化切换人设 <序号> 切换")

        yield event.chain_result([Plain("\n".join(lines))])

    @filter.command("手办化切换人设", prefix_optional=True)
    async def on_switch_persona(self, event: AstrMessageEvent, ctx=None):
        """切换当前人设"""
        if not self.is_admin(event): return
        parts = event.message_str.split()
        if len(parts) < 2 or not parts[1].isdigit():
            yield event.chain_result([Plain("用法: #手办化切换人设 <序号>\n使用 #手办化查看人设 查看可用人设")])
            return
        idx = int(parts[1])

        # 检查该人设是否有参考图
        key = f"_persona_{idx}_"
        has_ref = self.data_mgr.has_preset_ref_images(key)
        # 兼容旧版 _persona_ 无编号存储
        if not has_ref and idx == 1:
            has_ref = self.data_mgr.has_preset_ref_images("_persona_")

        self.conf["current_persona"] = idx
        self._save_config(["current_persona"])
        self._load_persona_scenes()  # 切换人设后重新加载场景提示词

        if has_ref:
            ref_count = len(await self._load_persona_ref_images(idx))
            persona_name = self._get_active_persona().get("persona_name", "小助手")
            yield event.chain_result([Plain(f"✅ 已切换至人设 {idx} ({persona_name})，共 {ref_count} 张参考图")])
        else:
            yield event.chain_result([Plain(f"✅ 已切换至人设 {idx}（暂无参考图）\n使用 #人设参考图添加 上传图片")])

    @filter.event_message_type(filter.EventMessageType.ALL, priority=4)
    async def on_batch_process_cmd(self, event: AstrMessageEvent, ctx=None):
        """批量处理上下文中的图片（指令模式）

        用法: #批量<预设名> [追加规则]
        示例: #批量手办化 皮肤白一点
        """
        if self.conf.get("prefix", True) and not event.is_at_or_wake_command:
            return

        text = event.message_str.strip()
        if not text: return

        # 消息去重检查：防止多平台重复处理同一消息
        msg_id = str(event.message_obj.message_id)
        dedup_key = f"batch_{msg_id}"
        if self._is_message_processed(dedup_key):
            logger.debug(f"FigurinePro: 批量处理消息 {msg_id} 已被处理，跳过重复执行")
            return

        # 匹配 "批量xxx" 或 "全部xxx"
        match = re.match(r"^(?:#|/)?(批量|全部)(.+)$", text)
        if not match:
            # 单独的 "批量" 或 "全部"
            if re.match(r"^(?:#|/)?(批量|全部)$", text):
                yield event.chain_result([Plain("用法: #批量<预设名> [追加规则]\n示例: #批量手办化 皮肤白一点")])
                event.stop_event()
            return

        preset_and_rules = match.group(2).strip()

        if preset_and_rules.startswith("处理"):
            # 兼容旧版的 "批量处理 xxx"
            parts = text.split(maxsplit=1)
            if len(parts) < 2:
                yield event.chain_result([Plain("用法: #批量<预设名> [追加规则]\n示例: #批量手办化 皮肤白一点")])
                event.stop_event()
                return
            prompt = parts[1].strip()
        else:
            prompt = preset_and_rules

        # 阻止事件继续传递给 on_figurine_request
        event.stop_event()

        active_task = await self._get_active_session_task(event.unified_msg_origin)
        if active_task:
            yield event.chain_result([Plain(self._build_active_task_reply(active_task))])
            return

        # 1. 提取当前消息的图片 URL（包括引用消息中的图片）
        msg_info = self._extract_message_info(event)
        current_urls = msg_info.get("image_urls", [])
        max_images = self.conf.get("batch_max_images", 10)

        # 2. 获取上下文中的图片
        uid = norm_id(event.get_sender_id())
        gid = norm_id(event.get_group_id())
        session_id = event.unified_msg_origin
        image_sources = await self._collect_images_from_context(
            session_id,
            count=self._context_rounds,
            include_bot=True,
            sender_id=uid
        )
        cached_image_sources = await self._get_recent_generated_image_sources(session_id, max_images=max_images)

        if not image_sources and not current_urls and not cached_image_sources:
            yield event.chain_result([Plain("上下文里没找到图片，先发点图再来批量处理吧~")])
            return

        # 3. 收集所有图片URL（统一过滤头像、去重并按顺序整理）
        merged_current_urls = list(current_urls)
        all_image_urls = self._merge_batch_image_urls(merged_current_urls, image_sources, max_images=max_images)
        total_images = len(all_image_urls)
        if total_images == 0:
            yield event.chain_result([Plain("没找到有效的图片。")])
            return

        # 计算预设和追加规则
        final_prompt, preset_name, extra_rules = self._process_prompt_and_preset(prompt)

        # 检查配额
        total_cost = total_images

        deduction = await self._check_quota(event, uid, gid, total_cost)
        if not deduction["allowed"]:
            yield event.chain_result(
                [Plain("这会儿先帮不了你啦，今天有点忙不过来。你可以稍后再来，或者联系管理员看下配额设置。")])
            return

        concurrency = max(1, self.conf.get("batch_concurrency", 3))

        # 发送开始提示
        _internal3 = {"自定义", "编辑", "edit", "custom"}
        preset_display = "" if (not preset_name or preset_name.strip().lower() in _internal3) else preset_name
        feedback = f"📦 批量处理任务开始\n"
        feedback += f"📷 共 {total_images} 张图片 | 并发: {concurrency}\n"
        if preset_display:
            feedback += f"🎨 预设: {preset_display}\n"
        feedback += f"⏳ 图片将并发处理，请耐心等待..."
        yield event.chain_result([Plain(feedback)])

        # 扣费
        if deduction["source"] == "user":
            await self.data_mgr.decrease_user_count(uid, total_cost)
        elif deduction["source"] == "group":
            await self.data_mgr.decrease_group_count(gid, total_cost)

        # 使用信号量控制并发
        semaphore = asyncio.Semaphore(concurrency)
        results = {"success": 0, "fail": 0}
        failed_details = []
        results_lock = asyncio.Lock()
        max_retries = self.conf.get("batch_retries", 2)

        async def process_single(index: int, url: str):
            async with semaphore:
                try:
                    # 下载图片
                    img_bytes = await self.img_mgr.load_bytes(url)
                    if not img_bytes:
                        error_msg = "图片下载失败，可能是链接已过期或网络问题"
                        async with results_lock:
                            results["fail"] += 1
                            failed_details.append({
                                "index": index,
                                "reason": error_msg,
                                "url_preview": url[:50] + "..." if len(url) > 50 else url
                            })
                        await event.send(event.chain_result([
                            Plain(f"第 {index}/{total_images} 张没弄好: {error_msg}")
                        ]))
                        return

                    # 处理单张图片（带重试机制）
                    retry_count = 0
                    success = False
                    error_msg = ""

                    while retry_count <= max_retries:
                        success, error_msg = await self._run_single_batch_task(
                            event=event,
                            image_bytes=img_bytes,
                            prompt=final_prompt,
                            preset_name=preset_name,
                            task_index=index,
                            total_tasks=total_images,
                            uid=uid,
                            gid=gid,
                            extra_rules=extra_rules,
                            image_source=url,
                            hide_text=False
                        )

                        if success:
                            break

                        retry_count += 1
                        if retry_count <= max_retries:
                            await event.send(event.chain_result([
                                Plain(
                                    f"⚠️ 第 {index}/{total_images} 张图片生成失败 ({error_msg})\n⏳ 正在进行第 {retry_count} 次重试...")
                            ]))
                            await asyncio.sleep(1.5)

                    async with results_lock:
                        if success:
                            results["success"] += 1
                        else:
                            results["fail"] += 1
                            failed_details.append({
                                "index": index,
                                "reason": error_msg,
                                "url_preview": url[:50] + "..." if len(url) > 50 else url
                            })
                            await event.send(event.chain_result([
                                Plain(f"第 {index}/{total_images} 张最终没弄好: {error_msg}")
                            ]))

                except Exception as e:
                    error_msg = self._resolve_debug_error_message(e, "这次没弄好，请稍后再试。")
                    logger.error(f"Batch process image {index} exception: {e}", exc_info=True)
                    async with results_lock:
                        results["fail"] += 1
                        failed_details.append({
                            "index": index,
                            "reason": error_msg,
                            "url_preview": url[:50] + "..." if len(url) > 50 else url
                        })
                    await event.send(event.chain_result([
                        Plain(f"第 {index}/{total_images} 张出了点问题: {error_msg}")
                    ]))

        async def process_all():
            tasks = [process_single(i, url) for i, url in enumerate(all_image_urls, 1)]
            await asyncio.gather(*tasks)

            # 发送完成汇总
            quota_str = self._get_quota_str(deduction, uid, gid)
            summary = f"\n📊 批量处理完成\n"
            summary += f"✅ 成功: {results['success']} 张\n"
            summary += f"失败: {results['fail']} 张\n"
            summary += f"💰 剩余次数: {quota_str}"

            if failed_details:
                summary += f"\n\n📋 失败图片汇总:"
                for detail in sorted(failed_details, key=lambda x: x['index'])[:5]:
                    summary += f"\n  • 第{detail['index']}张: {detail['reason']}"
                if len(failed_details) > 5:
                    summary += f"\n  ... 还有 {len(failed_details) - 5} 张失败"

            await event.send(event.chain_result([Plain(summary)]))

        # 启动异步任务
        asyncio.create_task(process_all())
