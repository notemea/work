
import os, time
from collections import deque, defaultdict
from pathlib import Path as _Path

# -------- 字体解析与中文绘制 --------
# -------- 字体解析与中文绘制 --------
# 给一个默认值（后面会被配置区覆盖）
CN_FONT_PATH = ""
_CN_FONT_FALLBACKS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
]
_cn_font_cache = {"key": None, "font": None, "path": None}
def _resolve_cn_font_path():
    for p in ([CN_FONT_PATH] if CN_FONT_PATH else []) + _CN_FONT_FALLBACKS:
        if not p:
            continue
        try:
            if _Path(p).exists():
                return p
        except Exception:
            pass
    return None

# Log resolved CN font once at import
try:
    _fp = _resolve_cn_font_path()
    log.info(f"[font] CN font path: {_fp}")
except Exception:
    pass


def _get_cn_font(px: int):
    try:
        from PIL import ImageFont
        path = _resolve_cn_font_path()
        key = (path, px)
        if _cn_font_cache.get("key") != key:
            if path is None:
                _cn_font_cache["font"] = None
            else:
                _cn_font_cache["font"] = ImageFont.truetype(path, px)
            _cn_font_cache["path"] = path
            _cn_font_cache["key"] = key
        return _cn_font_cache["font"]
    except Exception:
        return None


def draw_texts_cn(bgr_img, items, shadow=True):
    """Draw a batch of Chinese/ASCII texts.
    - Try PIL for high-quality Chinese rendering;
    - Per-item fallback to cv2 when the font for that px is unavailable;
    - If PIL is not installed at all, fallback all to cv2.
    items: [(text, (x,y), font_px, (b,g,r))...]
    """
    # If nothing to draw, return early
    if not items:
        return bgr_img

    # Try PIL import first
    try:
        from PIL import Image, ImageDraw
    except Exception:
        # Hard fallback to OpenCV
        for t,(x,y),px,col in items:
            try:
                import cv2
                scale = max(px/24.0, 0.6)
                cv2.putText(bgr_img, t, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2, cv2.LINE_AA)
            except Exception:
                pass
        return bgr_img

    # Use PIL canvas
    rgb = bgr_img[:, :, ::-1]
    im = Image.fromarray(rgb)
    draw = ImageDraw.Draw(im)

    any_pil = False
    for t,(x,y),px,col in items:
        font = _get_cn_font(px)
        if font is None:
            # Per-item fallback to cv2
            try:
                import cv2
                scale = max(px/24.0, 0.6)
                cv2.putText(bgr_img, t, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2, cv2.LINE_AA)
            except Exception:
                pass
            continue
        # Draw with PIL
        if shadow:
            draw.text((int(x)+1, int(y)+1), t, font=font, fill=(0,0,0))
        draw.text((int(x), int(y)), t, font=font, fill=(int(col[2]), int(col[1]), int(col[0])))
        any_pil = True

    if any_pil:
        out = np.array(im)[:, :, ::-1]
        bgr_img[:,:,:] = out
    return bgr_img
def measure_text_cn(text, font_px):
    font = _get_cn_font(font_px)
    if font is None:
        return int(len(text) * font_px * 0.6), font_px
    try:
        bbox = font.getbbox(text)
        return int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1])
    except Exception:
        return int(len(text) * font_px * 0.6), font_px

# -------- 抓拍去重器（IOU + 栅格冷却） --------
class _Snapper:
    def __init__(self, save_dir, iou_thres=0.35, cooldown=6.0, cell=72, cell_cd=10.0):
        self.save_dir = _Path(save_dir); self.save_dir.mkdir(parents=True, exist_ok=True)
        self.iou_thres = float(iou_thres); self.cooldown = float(cooldown)
        self.cell = int(cell); self.cell_cd = float(cell_cd)
        self.recent = deque(maxlen=128)  # (t, cls, x1,y1,x2,y2)
        self.cell_last = {}  # (cls, gx, gy) -> t

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
        iw = max(0.0, inter_x2 - inter_x1); ih = max(0.0, inter_y2 - inter_y1)
        inter = iw * ih
        if inter <= 0: return 0.0
        area_a = (ax2-ax1)*(ay2-ay1) + 1e-9; area_b = (bx2-bx1)*(by2-by1) + 1e-9
        return inter / (area_a + area_b - inter)

    def should_snap(self, cls_id, bbox, frame_shape):
        now = time.time()
        # IOU 冷却
        for t,c,*bb in list(self.recent)[::-1]:
            if c == cls_id and (now - t) < self.cooldown:
                if self._iou(bbox, bb) >= self.iou_thres:
                    return False
        # 栅格冷却
        h, w = frame_shape[:2]
        cx = int((bbox[0]+bbox[2])/2); cy = int((bbox[1]+bbox[3])/2)
        gx, gy = cx // self.cell, cy // self.cell
        k = (int(cls_id), int(gx), int(gy))
        if k in self.cell_last and (now - self.cell_last[k]) < self.cell_cd:
            return False
        return True

    def mark(self, cls_id, bbox, frame_shape):
        now = time.time()
        self.recent.append((now, int(cls_id), *bbox))
        h, w = frame_shape[:2]
        cx = int((bbox[0]+bbox[2])/2); cy = int((bbox[1]+bbox[3])/2)
        gx, gy = cx // self.cell, cy // self.cell
        self.cell_last[(int(cls_id), int(gx), int(gy))] = now

    def save(self, img_bgr, filename):
        day = time.strftime("%Y%m%d")
        day_dir = self.save_dir / day; day_dir.mkdir(parents=True, exist_ok=True)
        fp = day_dir / filename
        try:
            import cv2
            cv2.imwrite(str(fp), img_bgr)
            return str(fp)
        except Exception:
            return None
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GStreamer RTSP 拉流 → RKNN 推理 → GStreamer RTMP 推流（单文件，三套预设可切换）

▶ 用法：
    python3 infer_gst_rknn_single.py

本文件把所有参数集中在顶部，并提供多套「预设」：
  - default：均衡（默认）
  - low_latency：超低延迟优先
  - high_stability：链路更稳、抗抖动
  - conservative_detection：检测更保守，尽量少误报

仅显示 PPE（安全帽/无安全帽/安全背心/无安全背心），且必须在人框内；人框不绘制。
无任何“低分候选兜底”逻辑。
"""

# ========================== 预设选择（改这里） ==========================
PRESET = "default"  # 可选："default" / "low_latency" / "high_stability" / "conservative_detection"
# =====================================================================

# ========================== 参数区（基础配置） ==========================
BASE_CONFIG = dict(
    # 输入与输出
    INPUT_URL  = "rtsp://192.168.10.5:554/id=1&type=0",  # RTSP 拉流地址（摄像头）
    OUTPUT_URL = "rtmp://115.190.15.47:1935/live/250001?sign=41db35390ddad33f83944f44b8b75ded",  # RTMP 推流地址

    # 模型与类别
    MODEL_PATH = "best-rk3588.rknn",   # RKNN 模型（与脚本同目录）
    META_PATH  = "metadata.yaml",      # 元数据（names/img_size 等）；可置 None

    # 推理输入尺寸（若 metadata.yaml 里有 img_size/imgsz/input_size，将覆盖此项）
    IN_SIZE = (640, 640),

    # 检测/NMS/上屏阈值（严格模式：不会保留低分兜底）
    CONF_THRES = 0.40,   # 候选筛选（低于此分数不参与 NMS）
    DRAW_CONF  = 0.40,   # 上屏阈值（低于此分数不画）
    IOU_THRES  = 0.45,   # NMS IOU 阈值

    # PPE 与人框的关联（满足一条即可认为“在人身上”）
    PPE_NEED_CENTER_IN_PERSON = True,  # PPE 框中心点必须在人框内
    PPE_PERSON_MIN_IOU        = 0.10,  # 或与人框 IOU ≥ 此阈值（设 0 代表不启用 IOU 判定）

    # GStreamer 拉流
    RTSP_PROTOCOL   = "tcp",  # "udp" / "tcp"（udp 更低延迟，弱网更抖）
    RTSP_LATENCY_MS = 50,     # rtspsrc latency 毫秒（常试 50/30/20）

    # 解码器优先级（找到第一个可用的）
    DECODER_PRIORITY = ["mppvideodec", "v4l2h264dec", "openh264dec", "avdec_h264"],

    # 推流编码
    USE_HARDWARE_ENCODER = True,   # 尝试 mpph264enc 硬编
    FORCE_X264           = False,  # 强制 x264 软件编码（排障/更稳）
    BITRATE_KBPS         = 3000,   # 码率 kbps
    GOP                  = 50,     # 关键帧间隔（帧）

    # 输出尺寸/FPS（None = 跟随输入）
    OUT_WIDTH  = None,
    OUT_HEIGHT = None,
    DEFAULT_FPS_NUM = 25,  # 输入 caps 缺失时默认 FPS（分子）
    DEFAULT_FPS_DEN = 1,   # 输入 caps 缺失时默认 FPS（分母）

    # OSD
    SHOW_FPS_OSD = True,   # 是否显示 FPS/PPE 数
    FONT_SCALE   = 0.7,
    FONT_THICK   = 2,
    COLOR_OK     = (0, 255, 0),    # 合规（Hardhat / Safety Vest）
    COLOR_NG     = (0, 128, 255),  # 不合规（NO-*）

    # --- 中文字体与顶部安全边距（避免遮挡时间） ---
    CN_FONT_PATH = "",
    OSD_TOP_SAFE_MARGIN = 70,
    # 左侧中文列表（仅显示违规），自动换列
    LEFT_PANEL_ENABLE = True,
    LEFT_PANEL_SHOW_ALL = True,
    LEFT_PANEL_X = 10,
    LEFT_PANEL_Y = 100,
    LEFT_PANEL_SAFE_W = 320,  #  左侧列表安全宽度，避免编号覆盖0,
    LEFT_PANEL_LINE_H = 26,
    LEFT_PANEL_FONT_SCALE = 0.8,
    LEFT_PANEL_FONT_THICK = 2,
    LEFT_PANEL_SHADOW = True,
    LEFT_PANEL_WRAP = True,
    LEFT_PANEL_MAX_COLS = 3,
    LEFT_PANEL_COL_GAP = 24,
    LEFT_PANEL_BG = True,
    LEFT_PANEL_BG_ALPHA = 0.35,
    LEFT_PANEL_BG_PAD = (6,6,6,6),

    # --- 只画违规开关 ---
    DRAW_ONLY_VIOLATIONS = True,

    # --- 抓拍（仅全景）与去重冷却 ---
    SNAP_ENABLE = True,
    SNAP_DIR = "snapshots",
    SNAP_MIN_SCORE = 0.50,
    SNAP_COOLDOWN_SEC = 6.0,
    SNAP_IOU_DEDUP = 0.35,
    SNAP_CELL_SIZE = 72,
    SNAP_CELL_COOLDOWN_SEC = 10.0,
    SNAP_TEXT_TOP_MARGIN = 70,
    SNAP_MERGE_PER_PERSON = True,
    SNAP_DRAW_TEXT = False,
)

# ========================== 预设覆盖（按需改数值） ==========================
PROFILES = {
    # 均衡：默认就用 BASE_CONFIG
    "default": {},

    # 低延迟优先：UDP 拉流 + 更小 latency + 更短 GOP
    "low_latency": dict(
        RTSP_PROTOCOL = "udp",
        RTSP_LATENCY_MS = 20,   # 20~30 之间按现场调
        GOP = 25,               # 更频繁关键帧，首开更快
        BITRATE_KBPS = 2500,    # 适当降码率减压
        # 检测阈值可稍微放宽，以减少重试绘制时间
        CONF_THRES = 0.35,
        DRAW_CONF  = 0.40,
    ),

    # 高稳定优先：TCP + 大一点 latency + 软件编码（CPU 允许时）
    "high_stability": dict(
        RTSP_PROTOCOL = "tcp",
        RTSP_LATENCY_MS = 120,  # 大缓冲提升抗抖动能力
        USE_HARDWARE_ENCODER = False,
        FORCE_X264 = True,
        BITRATE_KBPS = 3000,
        GOP = 50,
        CONF_THRES = 0.40,
        DRAW_CONF  = 0.45,      # 上屏稍严，减少抖动时误报
    ),

    # 检测更保守（少误报）：阈值更高、人与 PPE 关联更严格
    "conservative_detection": dict(
        CONF_THRES = 0.55,
        DRAW_CONF  = 0.55,
        PPE_NEED_CENTER_IN_PERSON = True,
        PPE_PERSON_MIN_IOU = 0.20,
        RTSP_PROTOCOL = "tcp",
        RTSP_LATENCY_MS = 60,
    ),
}
# ==========================================================================

# 应用预设（BASE_CONFIG + PROFILES[PRESET]）
_CFG = BASE_CONFIG.copy()
_CFG.update(PROFILES.get(PRESET, {}))

# 将配置展开为常量（后续代码全部用这些常量）
INPUT_URL  = _CFG["INPUT_URL"]
OUTPUT_URL = _CFG["OUTPUT_URL"]
MODEL_PATH = _CFG["MODEL_PATH"]
META_PATH  = _CFG["META_PATH"]
IN_SIZE = tuple(_CFG["IN_SIZE"])
CONF_THRES = float(_CFG["CONF_THRES"])
DRAW_CONF  = float(_CFG["DRAW_CONF"])
IOU_THRES  = float(_CFG["IOU_THRES"])
PPE_NEED_CENTER_IN_PERSON = bool(_CFG["PPE_NEED_CENTER_IN_PERSON"])
PPE_PERSON_MIN_IOU        = float(_CFG["PPE_PERSON_MIN_IOU"])
RTSP_PROTOCOL   = _CFG["RTSP_PROTOCOL"].lower().strip()
RTSP_LATENCY_MS = int(_CFG["RTSP_LATENCY_MS"])
DECODER_PRIORITY = list(_CFG["DECODER_PRIORITY"])
USE_HARDWARE_ENCODER = bool(_CFG["USE_HARDWARE_ENCODER"])
FORCE_X264 = bool(_CFG["FORCE_X264"])
BITRATE_KBPS = int(_CFG["BITRATE_KBPS"])
GOP = int(_CFG["GOP"])
OUT_WIDTH  = _CFG["OUT_WIDTH"]
OUT_HEIGHT = _CFG["OUT_HEIGHT"]
DEFAULT_FPS_NUM = int(_CFG["DEFAULT_FPS_NUM"])
DEFAULT_FPS_DEN = int(_CFG["DEFAULT_FPS_DEN"])
SHOW_FPS_OSD = bool(_CFG["SHOW_FPS_OSD"])
FONT_SCALE = float(_CFG["FONT_SCALE"])
FONT_THICK = int(_CFG["FONT_THICK"])
COLOR_OK   = tuple(_CFG["COLOR_OK"])
COLOR_NG   = tuple(_CFG["COLOR_NG"])



# --- 提取新增配置 ---
CN_FONT_PATH = str(_CFG.get("CN_FONT_PATH", "") or "").strip()
OSD_TOP_SAFE_MARGIN = int(_CFG.get("OSD_TOP_SAFE_MARGIN", 70))

LEFT_PANEL_ENABLE = bool(_CFG.get("LEFT_PANEL_ENABLE", True))
LEFT_PANEL_SHOW_ALL = bool(_CFG.get("LEFT_PANEL_SHOW_ALL", False))
LEFT_PANEL_X = int(_CFG.get("LEFT_PANEL_X", 10))
LEFT_PANEL_Y = int(_CFG.get("LEFT_PANEL_Y", 100))
LEFT_PANEL_LINE_H = int(_CFG.get("LEFT_PANEL_LINE_H", 26))
LEFT_PANEL_FONT_SCALE = float(_CFG.get("LEFT_PANEL_FONT_SCALE", 0.8))
LEFT_PANEL_FONT_THICK = int(_CFG.get("LEFT_PANEL_FONT_THICK", 2))
LEFT_PANEL_SHADOW = bool(_CFG.get("LEFT_PANEL_SHADOW", True))
LEFT_PANEL_WRAP = bool(_CFG.get("LEFT_PANEL_WRAP", True))
LEFT_PANEL_MAX_COLS = int(_CFG.get("LEFT_PANEL_MAX_COLS", 3))
LEFT_PANEL_COL_GAP = int(_CFG.get("LEFT_PANEL_COL_GAP", 24))

DRAW_ONLY_VIOLATIONS = bool(_CFG.get("DRAW_ONLY_VIOLATIONS", True))

SNAP_ENABLE = bool(_CFG.get("SNAP_ENABLE", True))
SNAP_DIR = str(_CFG.get("SNAP_DIR", "snapshots"))
SNAP_MIN_SCORE = float(_CFG.get("SNAP_MIN_SCORE", 0.50))
SNAP_COOLDOWN_SEC = float(_CFG.get("SNAP_COOLDOWN_SEC", 6.0))
SNAP_IOU_DEDUP = float(_CFG.get("SNAP_IOU_DEDUP", 0.35))
SNAP_CELL_SIZE = int(_CFG.get("SNAP_CELL_SIZE", 72))
SNAP_CELL_COOLDOWN_SEC = float(_CFG.get("SNAP_CELL_COOLDOWN_SEC", 10.0))
SNAP_TEXT_TOP_MARGIN = int(_CFG.get("SNAP_TEXT_TOP_MARGIN", 70))
SNAP_MERGE_PER_PERSON = bool(_CFG.get("SNAP_MERGE_PER_PERSON", True))
SNAP_DRAW_TEXT = bool(_CFG.get("SNAP_DRAW_TEXT", False))
# ======================== 下面是实现（无需改动） =========================
import os, time, logging
from typing import Optional
import numpy as np, cv2

# GI / GStreamer
import gi
gi.require_version("Gst", "1.0")
gi.require_version("GObject", "2.0")
gi.require_version("GLib", "2.0")
from gi.repository import Gst, GObject, GLib
Gst.init(None)

# 可选：读取 metadata（若无则跳过）
try:
    import yaml
except Exception:
    yaml = None

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s [%(levelname)s] [preset:{PRESET}] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rknn-gst")

# --------------------- 工具函数 ---------------------
def letterbox_bgr(img, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / w, new_shape[1] / h)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]
    dw //= 2; dh //= 2
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = dh, new_shape[1]-new_unpad[1]-dh
    left, right = dw, new_shape[0]-new_unpad[0]-dw
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def xywh2xyxy(x):
    y = x.copy()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def nms_numpy(boxes, scores, iou_thres=0.45):
    x1 = boxes[:,0]; y1 = boxes[:,1]; x2 = boxes[:,2]; y2 = boxes[:,3]
    areas = (x2-x1+1)*(y2-y1+1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)
        inter = w*h
        iou = inter/(areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds+1]
    return keep

# --------------------- RKNN ---------------------
class RKNNWrapper:
    def __init__(self, model_path: str, meta_path: Optional[str] = None):
        self.ok = False
        self.in_size = tuple(IN_SIZE)
        self.classes = None
        self.anchors = None
        self.strides = None
        self.conf_thres = float(CONF_THRES)
        self.iou_thres = float(IOU_THRES)
        self.draw_conf = float(DRAW_CONF)

        # 仅解析 names/img_size（阈值以顶部参数为准）
        if yaml and meta_path and os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = yaml.safe_load(f) or {}
                self.classes = meta.get("names") or meta.get("classes")
                img_size = meta.get("img_size") or meta.get("imgsz") or meta.get("input_size")
                if isinstance(img_size, (list, tuple)) and len(img_size) >= 2:
                    self.in_size = (int(img_size[0]), int(img_size[1]))
                elif isinstance(img_size, int):
                    self.in_size = (img_size, img_size)
                self.anchors = meta.get("anchors")
                self.strides = meta.get("stride") or meta.get("strides")
                log.info(f"metadata 解析：classes={self.classes}, in_size={self.in_size}, strides={self.strides}")
            except Exception as e:
                log.warning(f"解析 metadata.yaml 失败：{e}")

        # 载入 RKNN
        self.rknn = None
        try:
            from rknnlite.api import RKNNLite
            self.rknn = RKNNLite()
        except Exception:
            try:
                from rknn.api import RKNN as RKNNLite
                self.rknn = RKNNLite()
            except Exception:
                self.rknn = None

        if self.rknn is None:
            log.warning("未检测到 rknn-toolkit(-lite)，将直通(不推理)。")
        else:
            try:
                ret = self.rknn.load_rknn(model_path);  assert ret == 0
                ret = self.rknn.init_runtime();          assert ret == 0
                self.ok = True
                log.info("RKNN 模型已加载")
            except Exception as e:
                log.error(f"初始化 RKNN 失败：{e}")
                self.rknn = None

        self.t0 = time.time(); self.frame_cnt = 0; self.fps = 0.0
        self._build_class_ids()

    # ---------- 类别/几何 ----------
    def _names_list(self):
        if isinstance(self.classes, dict):
            idxs = sorted(self.classes.keys(), key=lambda x: int(x))
            return [self.classes[i] for i in idxs]
        elif isinstance(self.classes, (list, tuple)):
            return list(self.classes)
        return None

    def _class_id(self, target_name, default=None):
        names = self._names_list()
        if names:
            low = [str(n).lower() for n in names]
            tn = target_name.lower()
            if tn in low:
                return low.index(tn)
        mapping = {'person':5,'hardhat':0,'no-hardhat':2,'safety vest':7,'no-safety vest':4}
        return mapping.get(target_name.lower(), default)

    def _build_class_ids(self):
        self.ID_PERSON     = self._class_id('Person')
        self.ID_HARDHAT    = self._class_id('Hardhat')
        # 只用这四个（与需求一致）
        self.ID_NO_HARDHAT = self._class_id('NO-Hardhat')
        self.ID_VEST       = self._class_id('Safety Vest')
        self.ID_NO_VEST    = self._class_id('NO-Safety Vest')
        self._ppe_ids = set([i for i in [self.ID_HARDHAT, self.ID_NO_HARDHAT, self.ID_VEST, self.ID_NO_VEST] if i is not None])

    def _cn_name(self, cls_id:int):
        m = {}
        if getattr(self, "ID_PERSON", None) is not None: m[self.ID_PERSON] = "人"
        if getattr(self, "ID_HARDHAT", None) is not None: m[self.ID_HARDHAT] = "已戴安全帽"
        if getattr(self, "ID_NO_HARDHAT", None) is not None: m[self.ID_NO_HARDHAT] = "未戴安全帽"
        if getattr(self, "ID_VEST", None) is not None: m[self.ID_VEST] = "已穿背心"
        if getattr(self, "ID_NO_VEST", None) is not None: m[self.ID_NO_VEST] = "未穿背心"
        return m.get(int(cls_id), None)

    def _iou(self, a, b):
        a = a[:4]; b = b[:4]
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
        iw = max(0.0, inter_x2 - inter_x1); ih = max(0.0, inter_y2 - inter_y1)
        inter = iw * ih
        if inter <= 0: return 0.0
        area_a = max(0.0, (ax2-ax1)) * max(0.0, (ay2-ay1))
        area_b = max(0.0, (bx2-bx1)) * max(0.0, (by2-by1))
        union = area_a + area_b - inter + 1e-6
        return inter / union

    def _center_inside(self, box, person):
        box = box[:4]; person = person[:4]
        cx = 0.5*(box[0]+box[2]); cy = 0.5*(box[1]+box[3])
        return (person[0] <= cx <= person[2]) and (person[1] <= cy <= person[3])

    # ---------- 主流程 ----------
    def infer_and_draw(self, bgr_frame):
        self.frame_cnt += 1
        if self.frame_cnt % 10 == 0:
            now = time.time(); dt = now - self.t0
            if dt > 0: self.fps = 10.0 / dt
            self.t0 = now

        if not self.ok:
            if SHOW_FPS_OSD:
                cv2.putText(bgr_frame, f"FPS:{self.fps:.1f} | passthrough", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), FONT_THICK, cv2.LINE_AA)
            return bgr_frame

        in_w, in_h = self.in_size
        img, r, (dw, dh) = letterbox_bgr(bgr_frame, (in_w, in_h))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.uint8)[None, ...]  # 1xHxWxC

        try:
            outputs = self.rknn.inference(inputs=[blob])
            if self.frame_cnt == 1:
                try:
                    shapes = [getattr(o, 'shape', None) for o in outputs]
                    log.info(f'RKNN outputs shapes: {shapes}')
                except Exception:
                    pass
        except Exception as e:
            cv2.putText(bgr_frame, f"Infer error:{e}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), FONT_THICK)
            return bgr_frame

        dets = self._postprocess(outputs, in_w, in_h)

        # 映射回原图坐标
        mapped = []
        for x1,y1,x2,y2,score,cls in dets:
            x1 = (x1 - dw) / r; y1 = (y1 - dh) / r
            x2 = (x2 - dw) / r; y2 = (y2 - dh) / r
            x1 = max(0, min(bgr_frame.shape[1]-1, x1))
            y1 = max(0, min(bgr_frame.shape[0]-1, y1))
            x2 = max(0, min(bgr_frame.shape[1]-1, x2))
            y2 = max(0, min(bgr_frame.shape[0]-1, y2))
            if x2<=x1 or y2<=y1: continue
            mapped.append([x1,y1,x2,y2,float(score),int(cls)])

        # 分成人与 PPE，仅在人区域内显示 PPE（人框不画）
        persons = [b for b in mapped if self.ID_PERSON is not None and b[5]==self.ID_PERSON]
        ppe_all = [b for b in mapped if b[5] in self._ppe_ids]

        kept = []
        for bb in ppe_all:
            ok = False
            for pp in persons:
                cond_center = (PPE_NEED_CENTER_IN_PERSON and self._center_inside(bb, pp))
                cond_iou    = (self._iou(bb, pp) >= PPE_PERSON_MIN_IOU) if (PPE_PERSON_MIN_IOU > 0) else False
                # 若中心&IOU都关（不建议），则不过滤
                if cond_center or cond_iou or (not PPE_NEED_CENTER_IN_PERSON and PPE_PERSON_MIN_IOU <= 0):
                    ok = True; break
            if ok and bb[4] >= self.draw_conf:
                kept.append(bb)

        # 绘制（仅 PPE）
        count = 0
        # ---- 仅 PPE 的绘制 + 左侧中文列表 + 抓拍（全景、按人合并） ----
        display = []
        for x1,y1,x2,y2,score,cls in kept:
            compliant = (cls == self.ID_HARDHAT) or (cls == self.ID_VEST)
            if DRAW_ONLY_VIOLATIONS and compliant:
                continue
            display.append([x1,y1,x2,y2,score,cls,compliant])

        # 按 y 排序，编号稳定
        display.sort(key=lambda b: b[1])

        left_lines = []       # [(text,color)]
        text_items = []       # for PIL rendering
        persons_map = []      # [(ppe_bbox, person_bbox or None, label_str, score, cls, idx)]
        left_safe_right = LEFT_PANEL_X + int(globals().get('LEFT_PANEL_SAFE_W', 0))

        # --- 绘制框 + 仅编号徽标 ---
        for idx, (x1,y1,x2,y2,score,cls,compliant) in enumerate(display, 1):
            color = COLOR_OK if compliant else COLOR_NG
            # 框
            cv2.rectangle(bgr_frame, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
            # 仅编号（无底色），优先画到框上方，避免遮挡时间
            badge = f"[{idx}]"
            scale = max(FONT_SCALE, 0.7)
            thick = max(int(FONT_THICK), 2)
            # 计算位置
            label_y = int(y1) - 8
            if label_y < max(6, OSD_TOP_SAFE_MARGIN):
                label_y = int(y1) + 18
            cv2.putText(bgr_frame, badge, (int(x1) if int(x1) >= left_safe_right else left_safe_right, int(label_y)),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

            # 左侧列表（仅违规）
            if LEFT_PANEL_SHOW_ALL or (not compliant):
                name = self._cn_name(cls)
                if not name:
                    names = self._names_list()
                    name = names[int(cls)] if (names and int(cls) < len(names)) else str(int(cls))
                left_lines.append((f"[{idx}] {name}:{float(score):.2f}", color))

            # 为抓拍分组准备（关联最近人框）
            # persons: list of [px1,py1,px2,py2,pscore,pcls]
            best_pp = None; best_iou = 0.0
            for pp in persons:
                # person entry shape depends on original; assume [:4] bbox
                px1,py1,px2,py2 = map(int, pp[:4])
                # IOU with PPE box
                iw = max(0, min(int(x2),px2)-max(int(x1),px1)); ih = max(0, min(int(y2),py2)-max(int(y1),py1))
                inter = iw*ih
                if inter>0:
                    area_a = (int(x2)-int(x1))*(int(y2)-int(y1)) + 1e-9
                    area_b = (px2-px1)*(py2-py1) + 1e-9
                    iou = inter/(area_a+area_b-inter)
                    if iou>best_iou:
                        best_iou=iou; best_pp=(px1,py1,px2,py2)
            persons_map.append(((int(x1),int(y1),int(x2),int(y2)), best_pp, f"[{idx}] {name if 'name' in locals() else ''}:{float(score):.2f}", float(score), int(cls), idx))

        # --- 左侧中文列表：去重 + 自动换列 ---
        if LEFT_PANEL_ENABLE and left_lines:
            
            try:
                log.info(f"[left-panel] lines={len(left_lines)}")
            except Exception:
                pass
            # 去重
            uniq = []
            seen = set()
            for t,c in left_lines:
                if t not in seen:
                    seen.add(t); uniq.append((t,c))
            # 渲染
            y = LEFT_PANEL_Y
            x = LEFT_PANEL_X
            font_px = int(24*LEFT_PANEL_FONT_SCALE)
            # 动态行高（避免中文字体度量导致的重叠）：取测量高度的1.2倍，并与配置中的最小行高取最大
            _w0, _h0 = measure_text_cn("国", font_px)
            line_h = max(LEFT_PANEL_LINE_H, int(_h0 * 1.2))
            max_h = bgr_frame.shape[0] - 10
            # 动态列宽（根据最宽文本来确定宽度），避免换列后挤在一起
            _col_w = 0
            for _t,_c in uniq:
                _w,_h = measure_text_cn(_t, font_px)
                if _w > _col_w:
                    _col_w = _w
            if _col_w <= 0:
                _col_w = measure_text_cn("MMMMMMMMMMMM", font_px)[0]
            col = 0
            for t,c in uniq:
                # 换列
                if y + line_h > max_h and LEFT_PANEL_WRAP and (col+1) < LEFT_PANEL_MAX_COLS:
                    col += 1
                    x = LEFT_PANEL_X + col * (_col_w + LEFT_PANEL_COL_GAP)
                    y = LEFT_PANEL_Y
                text_items.append((t, (x, y), font_px, c))
                y += line_h
                        # 背景遮罩（防止与其它文字重叠）
            if globals().get('LEFT_PANEL_BG', False) and text_items:
                try:
                    pad = globals().get('LEFT_PANEL_BG_PAD', (6,6,6,6))
                    x0 = min([xy for _,(xy,_y),_px,_c in text_items])
                    y0 = min([yy for _,(xx,yy),_px,_c in text_items]) - int(line_h*0.9)
                    x1 = max([xx + measure_text_cn(t, px)[0] for t,(xx,yy),px,_c in text_items])
                    y1 = max([yy + int(line_h*0.2) for _,(xx,yy),px,_c in text_items])
                    x0 -= int(pad[0]); y0 -= int(pad[1]); x1 += int(pad[2]); y1 += int(pad[3])
                    _draw_filled_rect_alpha(bgr_frame, (x0,y0), (x1,y1), (0,0,0), float(globals().get('LEFT_PANEL_BG_ALPHA',0.35)))
                except Exception:
                    pass
            
            # 批量中文渲染
            draw_texts_cn(bgr_frame, text_items, shadow=LEFT_PANEL_SHADOW)

        # --- 抓拍（仅全景、按人合并）---
        if SNAP_ENABLE:
            # 按 person_bbox 分组
            groups = defaultdict(list)  # person_bbox(tuple or None) -> list of (label, score, cls, ppe_bbox)
            for ppe_bbox, person_bbox, lab, sc, cl, idx in persons_map:
                compliant = (cl == self.ID_HARDHAT) or (cl == self.ID_VEST)
                if compliant: 
                    continue
                if float(sc) < SNAP_MIN_SCORE:
                    continue
                key = tuple(person_bbox) if person_bbox else tuple(ppe_bbox)
                groups[key].append((lab, sc, cl, ppe_bbox))

            if groups:
                if not hasattr(self, "_snapper"):
                    self._snapper = _Snapper(SNAP_DIR, SNAP_IOU_DEDUP, SNAP_COOLDOWN_SEC, SNAP_CELL_SIZE, SNAP_CELL_COOLDOWN_SEC)
                for key, items in groups.items():
                    # 以 person 框为判定重复的 bbox
                    bx1,by1,bx2,by2 = key
                    bbox_key = [int(bx1),int(by1),int(bx2),int(by2)]
                    # 同一人触发条件：类别取 NO-*（任一即可），冷却/去重判定使用 NO-* 的 cls
                    snap_cls = items[0][2] if items else 9999
                    if self._snapper.should_snap(snap_cls, bbox_key, bgr_frame.shape):
                        snap = bgr_frame.copy()
                        # 排序：先帽再背心
                        def _cmp(it):
                            lab, sc, cl, pb = it
                            pri = 0 if cl==self.ID_NO_HARDHAT else (1 if cl==self.ID_NO_VEST else 2)
                            return (pri, -float(sc))
                        items.sort(key=_cmp)
                        # 组装多行中文
                        if SNAP_DRAW_TEXT:
                            multi = []
                            yy = max(SNAP_TEXT_TOP_MARGIN, 30)
                            font_px2 = 26
                            for lab, sc, cl, pb in items:
                                # lab 已包含 [idx] 中文:score
                                multi.append((lab, (10, yy), font_px2, (0,255,255)))
                                yy += int(font_px2*1.1)
                            draw_texts_cn(snap, multi, shadow=True)
                            # 文件名使用第一条类别中文
                        first_name = self._cn_name(items[0][2]) or "违规"
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        fname = f"{ts}_{first_name}_{max([it[1] for it in items]):.2f}.jpg"
                        if self._snapper.save(snap, fname):
                            self._snapper.mark(snap_cls, bbox_key, bgr_frame.shape)

        if SHOW_FPS_OSD:
            cv2.putText(bgr_frame, f"FPS:{self.fps:.1f} | PPE:{count}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2, cv2.LINE_AA)
        return bgr_frame

    # ---------- 后处理（YOLOv8 单头优先，严格：无低分兜底） ----------
    def _postprocess(self, outputs, in_w, in_h):
        # YOLOv8-like single head: (1, 4+nc, N) / (1, N, 4+nc)
        try:
            out = outputs[0]
            shape = getattr(out, 'shape', None)
            if shape is not None and len(shape) == 3 and shape[0] == 1:
                o = out[0]
                C1, C2 = o.shape
                pred = o.T.astype(np.float32) if C1 < C2 else o.astype(np.float32)  # (N, 4+nc)
                box = pred[:, :4].astype(np.float32)
                cls = pred[:, 4:].astype(np.float32)
                if cls.size == 0: return []
                if cls.max() > 1.0 or cls.min() < 0.0:
                    cls = 1.0/(1.0+np.exp(-cls))  # sigmoid
                if float(np.max(box)) <= 2.0:
                    box[:, [0,2]] *= float(in_w)
                    box[:, [1,3]] *= float(in_h)
                conf = cls.max(axis=1); cls_id = np.argmax(cls, axis=1)
                mask = conf >= self.conf_thres
                if not np.any(mask): return []
                box = box[mask]; conf = conf[mask]; cls_id = cls_id[mask]
                # 假设 xywh → xyxy（若已是 xyxy 也无害）
                box_xyxy = xywh2xyxy(box)
                keep = nms_numpy(box_xyxy, conf, self.iou_thres)
                res = []
                for i in keep:
                    x1,y1,x2,y2 = box_xyxy[i]
                    res.append([x1,y1,x2,y2, float(conf[i]), int(cls_id[i])])
                return res
        except Exception:
            pass

        # 兼容：若不是 YOLOv8 格式，这里可按需扩展（省略）
        return []

# --------------------- GStreamer App ---------------------
class GSTApp:
    def __init__(self, input_url: str, output_url: str, rknn: RKNNWrapper):
        self.input_url = input_url
        self.output_url = output_url
        self.rknn = rknn
        self.in_pipeline = None; self.out_pipeline = None
        self.appsink = None; self.appsrc = None
        self.out_caps_str = None; self.running = True
        self.last_pts = 0; self.fps_num = DEFAULT_FPS_NUM; self.fps_den = DEFAULT_FPS_DEN

    def build_input_pipeline(self):
        chosen = next((n for n in DECODER_PRIORITY if Gst.ElementFactory.find(n) is not None), None)
        if not chosen:
            raise RuntimeError("未找到可用 H.264 解码器（%s）" % "/".join(DECODER_PRIORITY))
        proto = RTSP_PROTOCOL if RTSP_PROTOCOL in ("udp","tcp") else "tcp"
        lat_ms = int(RTSP_LATENCY_MS)

        desc = (
            f'rtspsrc location="{self.input_url}" protocols={proto} latency={lat_ms} ! '
            'rtph264depay ! h264parse ! ' + chosen + ' ! '
            'queue leaky=2 max-size-buffers=1 ! '
            'videoconvert ! video/x-raw,format=BGR ! '
            'queue leaky=2 max-size-buffers=1 ! '
            'appsink name=mysink emit-signals=true max-buffers=1 drop=true sync=false'
        )
        self.in_pipeline = Gst.parse_launch(desc)
        self.appsink = self.in_pipeline.get_by_name("mysink")
        self.appsink.connect("new-sample", self.on_new_sample)

        bus = self.in_pipeline.get_bus(); bus.add_signal_watch()
        bus.connect("message", self.on_bus_message, "input")

    def build_output_pipeline(self, width: int, height: int, fps_num: int, fps_den: int):
        if OUT_WIDTH  is not None: width  = int(OUT_WIDTH)
        if OUT_HEIGHT is not None: height = int(OUT_HEIGHT)

        self.fps_num, self.fps_den = fps_num, fps_den
        have_mpp = Gst.ElementFactory.find("mpph264enc") is not None
        use_mpp = USE_HARDWARE_ENCODER and have_mpp and (not FORCE_X264)

        if use_mpp:
            enc = f"queue ! mpph264enc bps={BITRATE_KBPS*1000} gop={GOP} ! h264parse config-interval=1"
            log.info("使用 mpph264enc 硬件编码")
            colorspace = "videoconvert ! video/x-raw,format=NV12"
        else:
            enc = f"queue ! x264enc tune=zerolatency speed-preset=ultrafast bitrate={BITRATE_KBPS} key-int-max={GOP} ! h264parse config-interval=1"
            if USE_HARDWARE_ENCODER and not have_mpp:
                log.warning("未检测到 mpph264enc，回退到 x264enc")
            else:
                log.info("使用 x264enc 软件编码")
            colorspace = "videoconvert"

        self.out_caps_str = f"video/x-raw,format=BGR,width={width},height={height},framerate={fps_num}/{fps_den}"
        desc = (
            f"appsrc name=mysrc is-live=true format=time do-timestamp=true caps={self.out_caps_str} ! "
            "queue leaky=2 max-size-buffers=1 ! "
            f"{colorspace} ! {enc} ! flvmux streamable=true ! "
            f'rtmpsink location="{self.output_url}" sync=false'
        )
        self.out_pipeline = Gst.parse_launch(desc)
        self.appsrc = self.out_pipeline.get_by_name("mysrc")

        bus = self.out_pipeline.get_bus(); bus.add_signal_watch()
        bus.connect("message", self.on_bus_message, "output")

        self.out_pipeline.set_state(Gst.State.PLAYING)
        log.info("输出管道已启动")

    def on_bus_message(self, bus, msg, tag):
        t = msg.type
        if t == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            log.error(f"[{tag}] GStreamer ERROR: {err} | {debug}")
            self.stop()
        elif t == Gst.MessageType.EOS:
            log.info(f"[{tag}] GStreamer EOS"); self.stop()

    def on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None: return Gst.FlowReturn.ERROR
        buf = sample.get_buffer()
        caps = sample.get_caps(); s = caps.get_structure(0)
        width  = int(s.get_value("width")); height = int(s.get_value("height"))
        try:
            fps = s.get_value("framerate"); fps_num = fps.numerator; fps_den = fps.denominator
        except Exception:
            fps_num, fps_den = DEFAULT_FPS_NUM, DEFAULT_FPS_DEN
        self.fps_num, self.fps_den = fps_num, fps_den

        if self.out_pipeline is None:
            self.build_output_pipeline(width, height, fps_num, fps_den)

        success, mapinfo = buf.map(Gst.MapFlags.READ)
        if not success: return Gst.FlowReturn.ERROR
        try:
            frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((height, width, 3)).copy()
            frame = self.rknn.infer_and_draw(frame)
            if frame is not None:
                self.push_frame(frame, buf)
        finally:
            buf.unmap(mapinfo)
        return Gst.FlowReturn.OK

    def push_frame(self, frame_bgr, inbuf: 'Gst.Buffer'):
        if frame_bgr is None or getattr(frame_bgr, "ndim", 0) != 3:
            return
        h, w, c = frame_bgr.shape
        data = frame_bgr.tobytes()
        outbuf = Gst.Buffer.new_allocate(None, len(data), None)
        outbuf.fill(0, data)

        pts = inbuf.pts; dts = inbuf.dts; duration = inbuf.duration
        if pts == Gst.CLOCK_TIME_NONE:
            if duration and duration != Gst.CLOCK_TIME_NONE:
                pts = self.last_pts + duration
            else:
                frame_time = int(Gst.SECOND * (self.fps_den / float(self.fps_num)))
                pts = self.last_pts + frame_time
        if dts == Gst.CLOCK_TIME_NONE:
            dts = pts
        if not duration or duration == Gst.CLOCK_TIME_NONE:
            duration = int(Gst.SECOND * (self.fps_den / float(self.fps_num)))
        outbuf.pts = pts; outbuf.dts = dts; outbuf.duration = duration
        self.last_pts = pts

        ret = self.appsrc.emit("push-buffer", outbuf)
        if ret != Gst.FlowReturn.OK:
            log.warning(f"appsrc push-buffer 返回：{ret}")

    def run(self):
        self.build_input_pipeline()
        self.in_pipeline.set_state(Gst.State.PLAYING)
        log.info("输入管道已启动")
        loop = GLib.MainLoop()
        try:
            loop.run()
        except KeyboardInterrupt:
            log.info("收到中断信号，准备退出...")
        finally:
            self.stop()
            try: loop.quit()
            except Exception: pass

    def stop(self):
        if not self.running: return
        self.running = False
        if self.in_pipeline:
            self.in_pipeline.set_state(Gst.State.NULL); log.info("输入管道已停止")
        if self.out_pipeline:
            try: self.appsrc.emit("end-of-stream")
            except Exception: pass
            self.out_pipeline.set_state(Gst.State.NULL); log.info("输出管道已停止")

# --------------------- 主入口 ---------------------
def main():
    model_path = os.path.abspath(MODEL_PATH)
    meta_path  = os.path.abspath(META_PATH) if META_PATH else None
    if not os.path.exists(model_path):
        log.warning(f"未找到模型：{model_path}")
    if meta_path and not os.path.exists(meta_path):
        log.warning(f"未找到 metadata：{meta_path}")

    rknn = RKNNWrapper(model_path, meta_path)
    app = GSTApp(INPUT_URL, OUTPUT_URL, rknn)
    app.run()

if __name__ == "__main__":
    main()
