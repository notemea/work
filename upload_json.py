# upload_json.py
import json, time, requests

def _bool(v): return bool(v) if isinstance(v, bool) else str(v).lower() in ("1","true","yes","on")

def post_event(payload: dict, webhook_cfg: dict):
    url = webhook_cfg["url"]
    timeout = float(webhook_cfg.get("timeout_sec", 5))
    verify = _bool(webhook_cfg.get("verify_tls", True))
    headers = {"Content-Type": "application/json"}
    headers.update(webhook_cfg.get("headers") or {})
    resp = requests.post(url, headers=headers, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                         timeout=timeout, verify=verify)
    if resp.status_code >= 300:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    return resp

# 命令行：python upload_json.py sample.json http://host/path
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python upload_json.py <json_file> [webhook_url]")
        sys.exit(1)
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)
    if len(sys.argv) >= 3:
        data["_override_url"] = sys.argv[2]
        resp = requests.post(sys.argv[2], headers={"Content-Type":"application/json"},
                             data=json.dumps(data, ensure_ascii=False).encode("utf-8"), timeout=5, verify=False)
    else:
        raise SystemExit("当作模块使用时请从 detect_and_post.py 调用 post_event；命令行测试请带上 webhook_url")
    print("done:", resp.status_code)
