# UZAY KAYDIRAĞI — 15 SN REF VİDEO DÖNÜŞÜMÜ HIZ TESTİ (hedef: 768p yüksek kalite, 10–12 dk)
# Yöntem: H3 I2V (fl2va pruned bf16, 20 adım) + Fun ControlNet Union 2.0, control_video = Canny(ref video).
#   ref2va [video editing] yerine bu yol: ref video diziye ek token olarak girmez, kontrol sinyali olarak girer
#   -> dizi yarı uzunlukta, dikkat (karesel) ~4 kat ucuz. uk_h3v2v_test: görüntü tamamen yeniden üretilir,
#   hareket/kompozisyon ref'ten gelir (ref2va stili kopyalıyordu).
# Varyantlar (aynı tohum/istem, model yüklü ölçülür; ISINMA süresi ölçüme girmez):
#   T1 = H3_HIZ (kitchen + BlockSparse), önbellek yok  -> mevcut kalite kuralı
#   T2 = T1 + EasyCache 0.08                           -> uk_h3v2v_test E2'de 5 sn: ~165 -> 127 sn; hayalet kontrol et
#   T3 = T1, dikkat = sage attention                   -> SageAttention kurulu değilse hata verip atlanır
# Çıktı: ciktilar/uk08_15sn_hiz_testi/{T1,T2,T3}.mp4 + sureler.json
# Kullanım: ref videoyu KAYNAK yoluna koy, ISTEM'i değiştir, bu dosyayı AI_Studyo/isler/ klasörüne bırak.
from studyo import *
import studyo, os, time, json, shutil, subprocess

AD = "uk08_15sn_hiz_testi"
HEDEF = f"{KOK}/ciktilar/{AD}"
KAYNAK = f"{KOK}/girdiler/ref15.mp4"       # dönüştürülecek 15 sn ref video (herhangi çözünürlük/fps)
SURE = 15
TOHUM = 7
GUC = 1.0                                   # ControlNet gücü: 1.0 = ref hareketine sıkı bağlı, 0.7–0.85 = daha serbest
ISTEM = """integrated_multimodal_description: [Shot 1] Live-action, cinematic, photorealistic science-fiction feature film footage, shot on an ARRI Alexa 65 with an anamorphic lens, hard white sunlight with deep black shadows in space, subtle film grain and natural motion blur. The small friendly humanoid alien, about 1.4 meters tall, with a large smooth rounded head, big glossy jet-black almond-shaped eyes, pale lavender-gray skin with fine pores, two short thin antennae ending in tiny glowing teal bulbs and a fitted tangerine-orange flight suit, slides at high speed down the colossal space slide: an open half-pipe chute of glossy white ceramic hull panels with glowing cyan rail strips and gunmetal support ribs, while planet Earth turns slowly far below among the stars.
overall_soundscape: Rushing wind whoosh, rising friction hiss and doppler swooshes of support ribs whipping past.
non_diegetic_music: N/A"""

os.makedirs(HEDEF, exist_ok=True)
t0 = time.time()
studyo.onyukle("h3")
A = studyo.H3_AYAR["yuksek"]
KARE = h3_kare_sayisi(SURE)
ISINMA_KARE = h3_kare_sayisi(2)


def hazirla(kare, dosya):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", KAYNAK, "-vf",
                    "fps=24,scale=1344:768:force_original_aspect_ratio=increase,crop=1344:768",
                    "-frames:v", str(kare), "-an", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p",
                    f"{KOK}/girdiler/{dosya}"], check=True)
    return dosya


def wf_cn(istem, kaynak, ad, hiz, kare, tohum=TOHUM, guc=GUC, dusuk=0.25, yuksek=0.6):
    wf = {
        "v1": {"class_type": "LoadVideo", "inputs": {"file": girdi(kaynak)}},
        "v2": {"class_type": "GetVideoComponents", "inputs": {"video": ["v1", 0]}},
        "v3": {"class_type": "Canny", "inputs": {"image": ["v2", 0], "low_threshold": dusuk, "high_threshold": yuksek}},
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": A["fl2va"], "weight_dtype": "default"}},
        "cn": {"class_type": "ModelPatchLoader", "inputs": {"name": "minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": studyo.H3_CLIP, "type": "minimax", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": studyo.H3_VAE}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "6": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["3", 0], "vae": ["4", 0], "prompt": istem,
              "width": 1344, "height": 768, "length": kare}},
        "7": {"class_type": "RandomNoise", "inputs": {"noise_seed": tohum}},
        "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "12": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["7", 0], "guider": ["11", 0], "sampler": ["8", 0],
               "sigmas": ["9", 0], "latent_image": ["6", 1]}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}},
        "14": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["12", 0], "vae": ["5", 0]}},
        "15": {"class_type": "CreateVideo", "inputs": {"images": ["13", 0], "audio": ["14", 0], "fps": 24}},
        "16": {"class_type": "SaveVideo", "inputs": {"video": ["15", 0], "filename_prefix": f"studyo/{ad}", "format": "auto", "codec": "auto"}},
    }
    model = ["1", 0]
    if hiz.get("dikkat"):
        wf["ab"] = {"class_type": "ModelAttentionBackend", "inputs": {"model": model, "attention": hiz["dikkat"]}}; model = ["ab", 0]
    if hiz.get("seyrek_tau"):
        wf["sp"] = {"class_type": "BlockSparseAttention", "inputs": {"model": model, "selection": "sol-attn",
                    "selection.tau": hiz["seyrek_tau"], "start_percent": hiz.get("seyrek_bas", 0.2), "end_percent": 1.0,
                    "dense_blocks": "", "min_tokens": 12288, "extra_tokens": 256, "sink_conditioning": "exact_kv_and_rows", "verbose": False}}
        model = ["sp", 0]
    wf["ap"] = {"class_type": "MiniMaxH3FunControlNetApply", "inputs": {"model": model, "model_patch": ["cn", 0], "vae": ["4", 0],
                "strength": guc, "start_percent": 0.0, "end_percent": 1.0, "control_video": ["v3", 0]}}
    model = ["ap", 0]
    if hiz.get("onbellek"):
        wf["ec"] = {"class_type": "EasyCache", "inputs": {"model": model, "reuse_threshold": hiz["onbellek"],
                    "start_percent": 0.15, "end_percent": 0.95, "verbose": True}}; model = ["ec", 0]
    wf["9"] = {"class_type": "BasicScheduler", "inputs": {"model": model, "scheduler": "simple", "steps": 20, "denoise": 1.0}}
    wf["11"] = {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": ["6", 0]}}
    return wf


girdi_15 = hazirla(KARE, "uk08_kaynak.mp4")
girdi_2 = hazirla(ISINMA_KARE, "uk08_isinma.mp4")
not_("hazır", round(time.time() - t0), "sn", {"kare": KARE})

# ISINMA: DiT + ControlNet yaması + metin kodlayıcı + VAE'ler belleğe alınsın, çekirdekler derlensin (ölçüme girmez)
t = time.time()
try:
    comfy(wf_cn(ISTEM, girdi_2, f"{AD}_isinma", dict(studyo.H3_HIZ), ISINMA_KARE), ad=f"{AD}/_gecici", zaman_asimi=3600)
except Exception as e:
    not_("ISINMA hatası", f"{type(e).__name__}: {str(e)[:300]}")
not_("ISINMA", round(time.time() - t), "sn")

VARYANT = {
    "T1": dict(studyo.H3_HIZ),
    "T2": dict(studyo.H3_HIZ, onbellek=0.08),
    "T3": dict(studyo.H3_HIZ, dikkat="sage attention"),
}
sonuc = {}
for k, hiz in VARYANT.items():
    ilerleme(f"{AD}: {k} başladı {hiz} | toplam {time.time() - t0:.0f} sn")
    t = time.time()
    try:
        cik = comfy(wf_cn(ISTEM, girdi_15, f"{AD}_{k}", hiz, KARE), ad=f"{AD}/_gecici", zaman_asimi=5400)
        for c in cik:
            shutil.move(c, f"{HEDEF}/{k}.mp4")
        sonuc[k] = {"sn": round(time.time() - t), "dk": round((time.time() - t) / 60, 1), "hiz": hiz}
    except Exception as e:
        sonuc[k] = {"sn": round(time.time() - t), "hiz": hiz, "hata": f"{type(e).__name__}: {str(e)[:300]}"}
    not_("VARYANT", k, sonuc[k])
    json.dump(sonuc, open(f"{HEDEF}/sureler.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

shutil.rmtree(f"{HEDEF}/_gecici", ignore_errors=True)
not_("TOPLAM", round(time.time() - t0), "sn", json.dumps({k: v.get("dk") for k, v in sonuc.items()}))
