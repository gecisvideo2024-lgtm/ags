# MiniMax H3 API — toplu video basımı

`planlar.json` içindeki her plan için MiniMax'in sunucularında H3 görevi açar, bitmesini bekler, MP4'ü `ciktilar/` klasörüne indirir. GPU gerekmez, ek Python paketi gerekmez (Python 3.9+).

## Kurulum
1. [platform.minimax.io](https://platform.minimax.io) hesabında **Pay-as-you-go** API anahtarı al (H3 bunu istiyor).
2. Anahtarı ortam değişkenine koy, dosyaya yazma:
   - Windows: `set MINIMAX_API_KEY=anahtarın`
   - Colab: `os.environ["MINIMAX_API_KEY"] = userdata.get("MINIMAX_API_KEY")` (Colab "Secrets" bölümüne ekle)
3. `planlar.ornek.json` dosyasını `planlar.json` olarak kopyala, istemleri ve referans adreslerini doldur.

## Çalıştırma
```
python h3_api.py planlar.json --kuru              # önce gönderilecek istekleri gör (ücretsiz)
python h3_api.py planlar.json --planlar S03       # tek planla dene
python h3_api.py planlar.json                     # hepsi, aynı anda 4 görev
```

## Plan alanları
| Alan | Değerler | Not |
|---|---|---|
| `istem` | metin | zorunlu |
| `ref_videolar` | URL listesi (en fazla 3) | hareket/kompozisyon referansı, ör. `uk_plain_S03.mp4` |
| `ref_gorseller` | URL listesi (en fazla 9) | karakter/stil referansı |
| `ilk_kare` | URL | başlangıç karesi |
| `cozunurluk` | `768P` / `2K` | 2K ≈ $0.13/sn, 768P ≈ $0.08/sn |
| `sure` | 4–15 | saniye |
| `en_boy` | `16:9`, `9:16`, `1:1`, `4:3`, `3:4`, `21:9` | |
| `ses` | true/false | yerel ses üretimi |

`varsayilan` bloğundaki değerler her plana uygulanır, plan içinde yazılan değer onu ezer.

## Bilinmesi gerekenler
- **Referans dosyaları internetten erişilebilir bir adreste olmalı** (MiniMax URL ile alıyor). Drive paylaşım bağlantıları doğrudan MP4 vermediği için çalışmayabilir.
- İstek alan adları MiniMax v2 belgesine göre yazıldı ama gerçek API ile henüz denenmedi. İlk gerçek çağrı hata verirse `govde_kur()` fonksiyonu ilk bakılacak yer; hata mesajı hangi alanın yanlış olduğunu söyler.
- İndirme bağlantısı ~9 saat geçerli, betik hemen indirir. Süreler ve görev kimlikleri `ciktilar/sureler.json` dosyasına yazılır.
