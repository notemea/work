# work
好的，这个项目包含以下文件：

```
.
├── 20250823_160457.jpg
├── README.md
├── config.yaml
├── detect_and_post.py
├── infer_gst_rknn_single_CAPTURE_CN_FULL_MERGE_AUTOFONT_fix_leftpanel_fontfix_overlapfixed_notext.py
├── sample.json
├── sample_225a.json
└── upload_json.py
```

这是 `README.md` 文件草稿：

-----

# 个人防护设备 (PPE) 检测与上报系统

## 工程目录

```
.
├── 20250823_160457.jpg
├── README.md
├── config.yaml
├── detect_and_post.py
├── infer_gst_rknn_single_CAPTURE_CN_FULL_MERGE_AUTOFONT_fix_leftpanel_fontfix_overlapfixed_notext.py
├── sample.json
├── sample_225a.json
└── upload_json.py
```

## 文件作用

  * `infer_gst_rknn_single_CAPTURE_CN_FULL_MERGE_AUTOFONT_fix_leftpanel_fontfix_overlapfixed_notext.py`: GStreamer RTSP 拉流 -\> RKNN 推理 -\> GStreamer RTMP 推流（单文件，三套预设可切换）。
  * `detect_and_post.py`: 独立上传模块。
  * `upload_json.py`: 负责将检测结果以 JSON 格式上传。
  * `config.yaml`: 配置文件，用于配置摄像头、webhook、类别映射等信息。
  * `sample.json`, `sample_225a.json`: JSON 示例文件，用于展示 `upload_json.py` 上传的数据格式。
  * `20250823_160457.jpg`: 抓拍图片示例。
  * `README.md`: 本文件，项目说明。
