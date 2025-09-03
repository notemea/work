#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, io, time, base64, json, logging, argparse, importlib.util
from copy import deepcopy

import cv2
import yaml
import numpy as np

# 独立上传模块
import upload_json as uploader

log = logging.getLogger("detect-and-post")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def b64_jpeg(img_bgr, quality=90):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buf = cv2.imencode(".jpg", img_bgr, encode_param)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")

def load_infer_module(py_path: str):
    spec = importlib.util.spec_from_file_location("infer_mod", py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def apply_overrides(mod, overrides: dict):
    if not overrides: return
    for k, v in overrides.items():
        if hasattr(mod, k):
            setattr(mod, k, v)
            log.info(f"override {k} = {v}")

def build_nn_output(items, frame_shape, cls_map, order="xyxy"):
    """items: list of (label, score, cls_id, bbox_xyxy)  bbox: (x1,y1,x2,y2) in pixels"""
    h, w = frame_shape[:2]
    out = []
    for lab, sc, cid_local, (x1,y1,x2,y2) in items:
        x1n, y1n, x2n, y2n = x1 / w, y1 / h, x2 / w, y2 / h
        # 选择映射：no-hardhat / no-vest / hardhat / vest / person
        key = None
        # 通过中文关键词/类别ID做一次鲁棒匹配
        ln = lab.replace("[","").replace("]","")
        if "未戴" in ln or "no-hat" in ln.lower(): key = "no-hardhat"
        elif "未穿" in ln or "no-vest" in ln.lower(): key = "no-vest"
        elif "安全帽" in ln and "未" not in ln: key = "hardhat"
        elif "背心" in ln and "未" not in ln: key = "vest"
        m = cls_map.get(key) or {}
        entry = {
            "conf": float(sc),
            "cid": int(m.get("cid", cid_local if cid_local is not None else 0)),
            "gcid": int(m.get("gcid", 0)),
            "aid": 0,
            "class_name": str(m.get("class_name", key or "")),
        }
        if order == "legacy":
            # x1=LT(w), x2=LT(h), y1=RB(w), y2=RB(h)
            entry.update({"x1": x1n, "x2": y1n, "y1": x2n, "y2": y2n})
        else:
            # 标准：x1=LT(w), y1=LT(h), x2=RB(w), y2=RB(h)
            entry.update({"x1": x1n, "y1": y1n, "x2": x2n, "y2": y2n})
        out.append(entry)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        C = yaml.safe_load(f)

    # 载入你的推理脚本模块（GStreamer/RKNN 封装都在里面）  ← 复用你上传的脚本
    infer_py = C["infer_module_path"]
    mod = load_infer_module(infer_py)   # <= 关键
    apply_overrides(mod, (C.get("infer_overrides") or {}))

    # 读配置
    CH = C["channel"]; WH = C["webhook"]
    CLS = C.get("classes", {})
    BOX_ORDER = (C.get("box_order") or "xyxy").lower().strip()
    SPol = C.get("snap_policy") or {}
    MIN_SCORE = float(SPol.get("min_score", 0.50))
    # 复用去重器（与原脚本同逻辑/参数）
    Snapper = getattr(mod, "_Snapper")
    snapper = Snapper(save_dir=getattr(mod, "SNAP_DIR", "snapshots"),
                      iou_thres=float(getattr(mod, "SNAP_IOU_DEDUP", 0.35)),
                      cooldown=float(getattr(mod, "SNAP_COOLDOWN_SEC", 6.0)),
                      cell=int(getattr(mod, "SNAP_CELL_SIZE", 72)),
                      cell_cd=float(getattr(mod, "SNAP_CELL_COOLDOWN_SEC", 10.0)))

    # —— 继承并“植入上报”逻辑 —— #
    class Reporter(mod.RKNNWrapper):
        def infer_and_draw(self, bgr_frame):
            # 复制原逻辑的精简版：推理→筛 PPE→只保留违规→按人合并
            self.frame_cnt += 1
            if self.frame_cnt % 10 == 0:
                now = time.time(); dt = now - self.t0
                if dt > 0: self.fps = 10.0 / dt
                self.t0 = now

            if not self.ok:
                if getattr(mod, "SHOW_FPS_OSD", True):
                    cv2.putText(bgr_frame, f"FPS:{self.fps:.1f} | passthrough", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), getattr(mod, "FONT_THICK", 2), cv2.LINE_AA)
                return bgr_frame

            in_w, in_h = self.in_size
            # letterbox
            def letterbox_bgr(img, new_shape, color=(114,114,114)):
                h, w = img.shape[:2]
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

            img, r, (dw, dh) = letterbox_bgr(bgr_frame, (in_w, in_h))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            blob = rgb.astype(np.uint8)[None, ...]

            try:
                outputs = self.rknn.inference(inputs=[blob])
            except Exception as e:
                cv2.putText(bgr_frame, f"Infer error:{e}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                return bgr_frame

            dets = self._postprocess(outputs, in_w, in_h)

            # 映回原图
            mapped = []
            H, W = bgr_frame.shape[:2]
            for x1,y1,x2,y2,score,cls in dets:
                x1 = (x1 - dw) / r; y1 = (y1 - dh) / r
                x2 = (x2 - dw) / r; y2 = (y2 - dh) / r
                x1 = max(0, min(W-1, x1)); y1 = max(0, min(H-1, y1))
                x2 = max(0, min(W-1, x2)); y2 = max(0, min(H-1, y2))
                if x2<=x1 or y2<=y1: continue
                mapped.append([x1,y1,x2,y2,float(score),int(cls)])

            persons = [b for b in mapped if getattr(self, "ID_PERSON", None) is not None and b[5]==self.ID_PERSON]
            ppe_all = [b for b in mapped if b[5] in self._ppe_ids]

            kept = []
            for bb in ppe_all:
                ok = False
                for pp in persons:
                    # 两个条件满足其一即可
                    cond_center = (getattr(mod, "PPE_NEED_CENTER_IN_PERSON", True) and self._center_inside(bb, pp))
                    cond_iou    = (self._iou(bb, pp) >= getattr(mod, "PPE_PERSON_MIN_IOU", 0.10)) if (getattr(mod, "PPE_PERSON_MIN_IOU", 0.10) > 0) else False
                    if cond_center or cond_iou or (not getattr(mod, "PPE_NEED_CENTER_IN_PERSON", True) and getattr(mod, "PPE_PERSON_MIN_IOU", 0.10) <= 0):
                        ok = True; break
                if ok and bb[4] >= self.draw_conf:
                    kept.append(bb)

            display = []
            for x1,y1,x2,y2,score,cls in kept:
                compliant = (cls == getattr(self, "ID_HARDHAT", -999)) or (cls == getattr(self, "ID_VEST", -999))
                # 只上传违规
                if compliant and getattr(mod, "DRAW_ONLY_VIOLATIONS", True):
                    continue
                display.append([x1,y1,x2,y2,score,cls,compliant])
            display.sort(key=lambda b: b[1])

            # 组人→多违规项
            from collections import defaultdict
            groups = defaultdict(list)  # key: person_bbox or ppe_bbox
            for x1,y1,x2,y2,score,cls,compliant in display:
                # 寻找关联的人框(最大IOU)
                best_pp = None; best_iou = 0.0
                for pp in persons:
                    px1,py1,px2,py2 = map(int, pp[:4])
                    iw = max(0, min(int(x2),px2)-max(int(x1),px1))
                    ih = max(0, min(int(y2),py2)-max(int(y1),py1))
                    inter = iw*ih
                    if inter>0:
                        area_a = (int(x2)-int(x1))*(int(y2)-int(y1)) + 1e-9
                        area_b = (px2-px1)*(py2-py1) + 1e-9
                        iou = inter/(area_a+area_b-inter)
                        if iou>best_iou:
                            best_iou=iou; best_pp=(px1,py1,px2,py2)
                # label 中文
                name = self._cn_name(cls) or str(int(cls))
                label = f"{name}"
                key = tuple(best_pp) if best_pp else (int(x1),int(y1),int(x2),int(y2))
                groups[key].append((label, float(score), int(cls), (int(x1),int(y1),int(x2),int(y2))))

            # —— 上传逻辑（去重+冷却 与原抓拍一致） ——
            for key, items in groups.items():
                bx1,by1,bx2,by2 = key
                bbox_key = [int(bx1),int(by1),int(bx2),int(by2)]
                # 取 NO-* 作为去重类别
                snap_cls = items[0][2] if items else 9999
                if not snapper.should_snap(snap_cls, bbox_key, bgr_frame.shape):
                    continue

                # 置信度阈值/排序（先帽后背心）
                items = [it for it in items if it[1] >= MIN_SCORE]
                if not items:
                    continue
                def _pri(it):
                    name = it[0]
                    pri = 0 if ("未戴" in name or "no" in name.lower()) else 1
                    return (pri, -it[1])
                items.sort(key=_pri)

                # 组 JSON
                now_ts = int(time.time())
                frame_h, frame_w = bgr_frame.shape[:2]
                # 图片：可选择“整帧”或“人框裁剪”；此处发整帧（更稳）
                pic_b64 = b64_jpeg(bgr_frame)
                nn_output = build_nn_output(items, bgr_frame.shape, CLS, order=BOX_ORDER)

                payload = {
                    "chid": CH["chid"],
                    "ncid": 0,
                    "ip": CH["ip"],
                    "geid": CH["geid"],
                    "sn": CH["sn"],
                    "sn32": CH["sn32"],
                    "location": CH["location"],
                    "width": CH.get("width") or frame_w,
                    "height": CH.get("height") or frame_h,
                    "desc": CH.get("desc", ""),
                    "snap_max_num": CH.get("snap_max_num", 1),
                    "pnum": CH.get("pnum", 1),
                    "record": CH.get("record", {"is_record": True, "is_send_video": False, "pre_time": 5, "after_time": 10}),
                    "pic_data": pic_b64,       # 抓拍图片（整帧）
                    "spic_data": pic_b64,      # 原始图片（此处同整帧；如需可换成人框裁剪）
                    "timestamp": now_ts,
                    "video": "",               # 如需推送小视频，可在此填入base64
                    "nn_output": nn_output,
                }

                # POST
                try:
                    uploader.post_event(payload, WH)
                    log.info(f"POST ok, {len(nn_output)} objs @ {now_ts}")
                    # 标记去重
                    snapper.mark(snap_cls, bbox_key, bgr_frame.shape)
                except Exception as e:
                    log.error(f"POST failed: {e}")

            # 仍然返回图像用于原推流（保持通路）
            if getattr(mod, "SHOW_FPS_OSD", True):
                cv2.putText(bgr_frame, f"FPS:{self.fps:.1f}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2, cv2.LINE_AA)
            return bgr_frame

    # 初始化并运行（完全复用你脚本里的 GStreamer 管道）
    rknn = Reporter(os.path.abspath(getattr(mod, "MODEL_PATH")))
    app = mod.GSTApp(getattr(mod, "INPUT_URL"), getattr(mod, "OUTPUT_URL"), rknn)
    app.run()

if __name__ == "__main__":
    main()
