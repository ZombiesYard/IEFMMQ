"""Minimal tunnel test: httpx vs Windows SSH tunnel to vLLM."""
import httpx, json, time, os, base64

URL = "http://localhost:16324/v1/chat/completions"
MODEL = os.getenv("SIMTUTOR_MODEL_NAME", "simtutor")
TIMEOUT = 60

def test(name: str, payload: dict) -> bool:
    c = httpx.Client(timeout=TIMEOUT)
    try:
        body = json.dumps(payload)
        size_mb = len(body) / 1024 / 1024
        t0 = time.time()
        r = c.post(URL, content=body, headers={"Content-Type": "application/json"})
        elapsed = time.time() - t0
        ok = r.status_code == 200
        print(f"  {name}: HTTP {r.status_code} ({size_mb:.1f}MB, {elapsed:.2f}s) {'OK' if ok else 'FAIL'}")
        if not ok:
            print(f"    {r.text[:200]}")
        return ok
    except Exception as e:
        print(f"  {name}: {type(e).__name__}: {e}")
        return False
    finally:
        c.close()

print("=== Tunnel Test ===\n")

# 1-3: basic tests
big_text = "请根据座舱信息分析" * 1500
ok1 = test("text-small", {"model": MODEL, "messages": [{"role":"user","content":"Hi"}], "max_tokens": 5})
ok2 = test("text-20KB", {"model": MODEL, "messages": [{"role":"user","content": big_text}], "max_tokens": 30})
real_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
ok3 = test("mm-tiny", {"model": MODEL, "messages": [{"role":"user", "content": [
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{real_png}"}},
    {"type": "text", "text": "Say hello"}
]}], "max_tokens": 10})

# 4. ~5MB fake image (matches real screenshot size)
print("\n--- 5MB multimodal (simulates real DCS screenshot) ---")
fake_img = base64.b64encode(os.urandom(3800000)).decode()
big_payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{fake_img}"}},
        {"type": "text", "text": big_text}
    ]}],
    "max_tokens": 30,
}
ok4 = test("mm-5MB-fresh", big_payload)  # vLLM will reject fake image, we just care about transport

# 5. Connection reuse (mimics simtutor's BaseHelpModel._client)
print("\n--- Connection reuse x5 (mimics simtutor) ---")
print("  mm-5MB-reuse: ", end="", flush=True)
c = httpx.Client(timeout=TIMEOUT)
reuse_ok = 0
reuse_fail = 0
for i in range(5):
    try:
        body = json.dumps(big_payload)
        t0 = time.time()
        r = c.post(URL, content=body, headers={"Content-Type": "application/json"})
        elapsed = time.time() - t0
        print(f"#{i+1}:{r.status_code}({elapsed:.1f}s) ", end="", flush=True)
        reuse_ok += 1
    except Exception as e:
        print(f"#{i+1}:{type(e).__name__} ", end="", flush=True)
        reuse_fail += 1
c.close()
print(f"\n  reuse result: {reuse_ok} ok, {reuse_fail} fail")

print(f"\n{'='*50}")
print(f"Basic: {sum([ok1,ok2,ok3])}/3  5MB: {'OK' if ok4 else 'FAIL'}  Reuse: {reuse_ok}/5")
print(f"RemoteProtocolError count: {reuse_fail}")
