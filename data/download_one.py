                      
"""单仓库分块并发下载（辅助脚本）。

和 models/download_models.py 同一套思路，区别只有三点：
  1. 只下一个仓库，仓库名/目标目录从命令行环境变量给，不改动那个脚本的 MODELS 列表；
  2. 走 hf-mirror 而不是 huggingface.co；
  3. 支持把 `hf download` 留下的 .incomplete 前缀接续过来——hf 是顺序写的，
     所以前 N 个完整块可以直接标记为已完成，不用重下。

用法：
  DL_REPO=Qwen/Qwen2.5-1.5B DL_DIR=/path/to/models/Qwen2.5-1.5B python download_one.py
"""
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("http_proxy", "http://agent.baidu.com:8891")
os.environ.setdefault("https_proxy", "http://agent.baidu.com:8891")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import requests

REPO = os.environ["DL_REPO"]
LOCAL_DIR = os.environ["DL_DIR"]
ENDPOINT = os.environ.get("DL_ENDPOINT", "https://hf-mirror.com")

CHUNK = 16 * 1024 * 1024
WORKERS = int(os.environ.get("DL_WORKERS", "16"))
CHUNK_DEADLINE = 60
CHUNK_RETRY = 20
SESSION = requests.Session()
SESSION.mount("https://", requests.adapters.HTTPAdapter(
    pool_connections=4, pool_maxsize=WORKERS * 2))
LOCK = threading.Lock()


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def list_files():
    url = "%s/api/models/%s/tree/main?recursive=true" % (ENDPOINT, REPO)
    r = SESSION.get(url, timeout=30)
    r.raise_for_status()
    out = []
    for f in r.json():
        if f.get("type") != "file":
            continue
        size = (f.get("lfs") or {}).get("size") or f.get("size") or 0
        out.append((f["path"], size))
    return out


def fetch_chunk(url, fd, index, start, end):
    headers = {"Range": "bytes=%d-%d" % (start, end - 1)}
    deadline = time.time() + CHUNK_DEADLINE
    offset = start
    with SESSION.get(url, headers=headers, stream=True, timeout=(15, 20)) as r:
        r.raise_for_status()
        for block in r.iter_content(chunk_size=1024 * 1024):
            if not block:
                continue
            os.pwrite(fd, block, offset)
            offset += len(block)
            if time.time() > deadline and offset < end:
                raise TimeoutError("块 %d 超时，已传 %dB" % (index, offset - start))
    if offset != end:
        raise IOError("块 %d 长度不符: %d != %d" % (index, offset - start, end - start))
    return True


def chunk_worker(url, fd, index, start, end, progress_path, done):
    for attempt in range(CHUNK_RETRY):
        try:
            fetch_chunk(url, fd, index, start, end)
            with LOCK:
                done.add(index)
                with open(progress_path, "a") as pf:
                    pf.write("%d\n" % index)
            return True
        except Exception as exc:
            if attempt == CHUNK_RETRY - 1:
                log("  块 %d 放弃: %s" % (index, exc))
                return False
            time.sleep(1)
    return False


def seed_from_incomplete(dest, size, progress_path, total_chunks):
    """把 hf download 留下的 .incomplete 前缀接续过来。

    hf 是顺序写的，所以已落盘的 n 字节就是文件的前 n 字节。完整覆盖的块
    （即 (i+1)*CHUNK <= n 的那些）可以直接算作已完成，剩下的照常下。
    """
    if os.path.exists(dest) and os.path.getsize(dest) >= size:
        return set()
    cache = os.path.join(LOCAL_DIR, ".cache", "huggingface", "download")
    if not os.path.isdir(cache) or os.path.exists(progress_path):
        return set()
    cands = [os.path.join(cache, f) for f in os.listdir(cache)
             if f.endswith(".incomplete")]
    if not cands:
        return set()
    src = max(cands, key=os.path.getsize)
    got = os.path.getsize(src)
    if got < CHUNK:
        return set()
    full = min(got // CHUNK, total_chunks)
    log("  发现 hf 残留 %.2fGB，可接续前 %d 块（共 %d 块）"
        % (got / 2 ** 30, full, total_chunks))
    fd = os.open(dest, os.O_RDWR | os.O_CREAT)
    try:
        os.ftruncate(fd, size)
        with open(src, "rb") as fin:
            remaining = full * CHUNK
            offset = 0
            while remaining > 0:
                block = fin.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    break
                os.pwrite(fd, block, offset)
                offset += len(block)
                remaining -= len(block)
    finally:
        os.close(fd)
    done = set(range(full))
    with open(progress_path, "w") as pf:
        for i in sorted(done):
            pf.write("%d\n" % i)
    os.remove(src)
    log("  已接续 %.2fGB，删除 hf 残留文件" % (full * CHUNK / 2 ** 30))
    return done


def is_complete(dest, size, progress_path, total_chunks):
    """判断文件是否真的下完。

    不能只看文件大小：dest 是 os.ftruncate 预分配成完整大小的，缺块时大小照样对得上。
    这正是 models/download_models.py 第 98 行的隐患——第 2 轮重试一进来就被大小
    骗过去，直接 return True，把缺块的文件当成完成品。所以有分块记录时必须以记录为准。
    """
    if not os.path.exists(dest) or os.path.getsize(dest) != size:
        return False
    if not os.path.exists(progress_path):
        return True                                              
    with open(progress_path) as pf:
        done = {int(x) for x in pf.read().split() if x.strip()}
    return len(done) >= total_chunks


def download_file(rel_path, size):
    dest = os.path.join(LOCAL_DIR, rel_path)
    os.makedirs(os.path.dirname(dest) or LOCAL_DIR, exist_ok=True)
    if size == 0:
        open(dest, "wb").close()
        return True

    url = "%s/%s/resolve/main/%s" % (ENDPOINT, REPO, rel_path)
    prog_dir = os.path.join(LOCAL_DIR, ".dlprogress")
    os.makedirs(prog_dir, exist_ok=True)
    progress_path = os.path.join(prog_dir, rel_path.replace("/", "__") + ".chunks")
    total_chunks = max(1, (size + CHUNK - 1) // CHUNK)

    if is_complete(dest, size, progress_path, total_chunks):
        return True

    done = seed_from_incomplete(dest, size, progress_path, total_chunks)
    if os.path.exists(progress_path) and os.path.exists(dest):
        with open(progress_path) as pf:
            done |= {int(x) for x in pf.read().split() if x.strip()}

    fd = os.open(dest, os.O_RDWR | os.O_CREAT)
    try:
        os.ftruncate(fd, size)
        todo = [i for i in range(total_chunks) if i not in done]
        if not todo:
            return True
        log("  %s  %.2fGB  共 %d 块，待下 %d 块"
            % (rel_path, size / 2 ** 30, total_chunks, len(todo)))
        start_t = time.time()
        ok = True
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futs = {}
            for i in todo:
                s = i * CHUNK
                e = min(s + CHUNK, size)
                futs[pool.submit(chunk_worker, url, fd, i, s, e, progress_path, done)] = i
            finished = 0
            for fut in as_completed(futs):
                if not fut.result():
                    ok = False
                finished += 1
                if finished % 10 == 0 or finished == len(todo):
                    rate = (finished * CHUNK) / (time.time() - start_t) / 2 ** 20
                    log("    %s: %d/%d 块  本地 %.2fGB  %.1fMB/s"
                        % (rel_path, finished, len(todo),
                           len(done) * CHUNK / 2 ** 30, rate))
        return ok
    finally:
        os.close(fd)


def main():
    t0 = time.time()
    files = list_files()
    total = sum(s for _, s in files)
    log("=== %s  %d 个文件  合计 %.2fGB -> %s（endpoint=%s，%d 线程）"
        % (REPO, len(files), total / 2 ** 30, LOCAL_DIR, ENDPOINT, WORKERS))
    all_ok = True
    for rel_path, size in sorted(files, key=lambda x: x[1]):
        for attempt in range(3):
            if download_file(rel_path, size):
                break
            log("  %s 第 %d 轮未完成，重试" % (rel_path, attempt + 1))
        else:
            all_ok = False
            log("  FAIL %s" % rel_path)
    got = sum(os.path.getsize(os.path.join(LOCAL_DIR, p)) for p, _ in files
              if os.path.exists(os.path.join(LOCAL_DIR, p)))
    log("=== %s %s  本地 %.2fGB / %.2fGB  用时 %.1f 分钟"
        % (REPO, "完成" if all_ok else "未完整",
           got / 2 ** 30, total / 2 ** 30, (time.time() - t0) / 60))
                                         
    for rel_path, size in files:
        prog = os.path.join(LOCAL_DIR, ".dlprogress",
                            rel_path.replace("/", "__") + ".chunks")
        if not os.path.exists(prog):
            continue
        total_chunks = max(1, (size + CHUNK - 1) // CHUNK)
        with open(prog) as pf:
            done = {int(x) for x in pf.read().split() if x.strip()}
        miss = [i for i in range(total_chunks) if i not in done]
        if miss:
            all_ok = False
            log("  !! %s 缺 %d 块: %s" % (rel_path, len(miss), miss[:20]))
        else:
            log("  OK %s  %d/%d 块" % (rel_path, len(done), total_chunks))
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
