# MiniMax H3 API ile video basımı — planlar.json'daki her plan için görev açar, bekler, MP4'ü indirir.
# GPU gerekmez: üretim MiniMax sunucularında. Sadece Python 3.9+ (ek paket yok).
#
# Kullanım:
#   set MINIMAX_API_KEY=...            (Windows)   |  export MINIMAX_API_KEY=...  (Linux/Colab)
#   python h3_api.py planlar.json                     -> tüm planlar, aynı anda en fazla 4 görev
#   python h3_api.py planlar.json --planlar S01,S03   -> sadece seçilenler
#   python h3_api.py planlar.json --kuru              -> API'ye göndermeden istek gövdelerini yazdırır
#
# Uç noktalar (platform.minimax.io belgeleri):
#   POST https://api.minimax.io/v2/video_generation          -> görev açar
#   GET  https://api.minimax.io/v2/query/video_generation/ID -> durum; başarıda task.content.url
# İndirme bağlantısı ~9 saat geçerli, betik hemen indirir.
import argparse, json, os, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

API = os.environ.get("MINIMAX_API_BASE", "https://api.minimax.io")
KOK = os.path.dirname(os.path.abspath(__file__))


def istek(yontem, yol, anahtar, govde=None, zaman_asimi=120):
    veri = json.dumps(govde).encode("utf-8") if govde is not None else None
    req = urllib.request.Request(API + yol, data=veri, method=yontem, headers={
        "Authorization": f"Bearer {anahtar}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=zaman_asimi) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} {yol}: {e.read().decode('utf-8', 'replace')[:500]}")


def govde_kur(plan, varsayilan):
    """Bir planı MiniMax v2 istek gövdesine çevirir. Alan adları belgeye göre; ret gelirse ilk bakılacak yer burası."""
    ayar = {**varsayilan, **plan}
    icerik = [{"type": "text", "text": ayar["istem"]}]
    for url in ayar.get("ref_videolar", []):        # hareket/kompozisyon referansı (en fazla 3)
        icerik.append({"type": "video_url", "video_url": {"url": url}, "role": "reference_video"})
    for url in ayar.get("ref_gorseller", []):       # karakter/stil referansı (en fazla 9)
        icerik.append({"type": "image_url", "image_url": {"url": url}, "role": "reference_image"})
    if ayar.get("ilk_kare"):
        icerik.append({"type": "image_url", "image_url": {"url": ayar["ilk_kare"]}, "role": "first_frame"})
    return {
        "model": ayar.get("model", "MiniMax-H3"),
        "content": icerik,
        "resolution": ayar.get("cozunurluk", "2K"),          # MiniMax-H3: "768P" | "2K"
        "duration": int(ayar.get("sure", 15)),                # 4–15 sn
        "aspect_ratio": ayar.get("en_boy", "16:9"),
        "generate_audio": bool(ayar.get("ses", True)),
    }


def gorev_id(yanit):
    return yanit.get("task_id") or (yanit.get("task") or {}).get("id") or (yanit.get("data") or {}).get("task_id")


def bas(ad, govde, anahtar, hedef, bekleme=15, en_uzun=3600):
    t0 = time.time()
    yanit = istek("POST", "/v2/video_generation", anahtar, govde)
    tid = gorev_id(yanit)
    if not tid:
        raise RuntimeError(f"görev kimliği yok: {json.dumps(yanit)[:500]}")
    print(f"[{ad}] görev açıldı {tid}", flush=True)
    while time.time() - t0 < en_uzun:
        time.sleep(bekleme)
        g = istek("GET", f"/v2/query/video_generation/{tid}", anahtar)
        gorev = g.get("task", g)
        durum = str(gorev.get("status", "")).lower()
        if durum in ("succeeded", "success"):
            url = (gorev.get("content") or {}).get("url") or gorev.get("video_url")
            if not url:
                raise RuntimeError(f"başarılı ama bağlantı yok: {json.dumps(g)[:500]}")
            yol = os.path.join(hedef, f"{ad}.mp4")
            urllib.request.urlretrieve(url, yol)
            return {"sn": round(time.time() - t0), "dosya": yol, "gorev": tid}
        if durum in ("failed", "fail", "cancelled", "canceled"):
            raise RuntimeError(f"görev başarısız: {json.dumps(g)[:500]}")
        print(f"[{ad}] {durum or 'bekliyor'} | {time.time() - t0:.0f} sn", flush=True)
    raise TimeoutError(f"{en_uzun} sn içinde bitmedi (görev {tid})")


def main():
    ap = argparse.ArgumentParser(description="MiniMax H3 API toplu video basımı")
    ap.add_argument("plan_dosyasi")
    ap.add_argument("--planlar", default="", help="virgülle: S01,S03 (boş = hepsi)")
    ap.add_argument("--paralel", type=int, default=4, help="aynı anda açık görev sayısı")
    ap.add_argument("--cikti", default=os.path.join(KOK, "ciktilar"))
    ap.add_argument("--kuru", action="store_true", help="göndermeden istek gövdelerini yazdır")
    a = ap.parse_args()

    with open(a.plan_dosyasi, encoding="utf-8") as f:
        dosya = json.load(f)
    varsayilan, planlar = dosya.get("varsayilan", {}), dosya["planlar"]
    secili = [p for p in a.planlar.split(",") if p]
    planlar = {k: v for k, v in planlar.items() if not secili or k in secili}
    govdeler = {ad: govde_kur(p, varsayilan) for ad, p in sorted(planlar.items())}

    if a.kuru:
        print(json.dumps(govdeler, ensure_ascii=False, indent=1))
        return
    anahtar = os.environ.get("MINIMAX_API_KEY")
    if not anahtar:
        sys.exit("MINIMAX_API_KEY ortam değişkeni yok.")
    os.makedirs(a.cikti, exist_ok=True)

    sonuc = {}
    with ThreadPoolExecutor(max_workers=a.paralel) as havuz:
        isler = {havuz.submit(bas, ad, g, anahtar, a.cikti): ad for ad, g in govdeler.items()}
        for i in as_completed(isler):
            ad = isler[i]
            try:
                sonuc[ad] = i.result()
                print(f"[{ad}] BİTTİ {sonuc[ad]['sn']} sn -> {sonuc[ad]['dosya']}", flush=True)
            except Exception as e:
                sonuc[ad] = {"hata": f"{type(e).__name__}: {e}"}
                print(f"[{ad}] HATA {sonuc[ad]['hata']}", flush=True)
    with open(os.path.join(a.cikti, "sureler.json"), "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=1)
    print("TOPLAM", json.dumps(sonuc, ensure_ascii=False))


if __name__ == "__main__":
    main()
