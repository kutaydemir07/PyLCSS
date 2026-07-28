# Aktif Öğrenme — Mevcut Kod İncelemesi ve Entegrasyon Planı

*Kod okuması: `surrogate_training_widget.py`, `training_engine.py`, `models.py`,
`dispatcher.py`. İddiaların bir kısmı sklearn 1.7.2 ile deneysel olarak doğrulandı.*

---

## 1. Mevcut akış (olduğu gibi)

```
[GUI] btn_adaptive  (widget:530)
   └─> start_adaptive_training()  (widget:1404)
         ├─ X_train var mı? node seçili mi? config = get_config()
         ├─ PyTorch + dropout==0 ise uyarı
         ├─ GraphBuilder.build_spy_model(...) -> spy_code, spy_inputs, spy_outputs
         ├─ input_nodes'lardan input_bounds (min/max) topla
         └─> AdaptiveTrainingWorker(QThread)  (widget:1940)
               for round in 1..5:
                 1. trainer.train_model(X, y, config, X_test, y_test)
                 2. 1000 LHS aday üret (bounds'a ölçekli)
                 3. model.predict(candidates, return_std=True)
                 4. np.argsort(y_std)[-10:]  -> 10 nokta
                 5. _evaluate_points(new_X)  -> exec(spy_code)
                 6. X, y'ye ekle
               final train_model(tüm veri)
               done_sig -> adaptive_training_finished()  (widget:1487)
```

AI asistanı bağlantısı: `dispatcher.py:1375 _adaptive_training()` — sadece butona
`QMetaObject.invokeMethod` ile tıklıyor. Parametre geçirmiyor.

---

## 2. Kritik bulgu: özellik çoğu model için sessizce çalışmıyor

`AdaptiveTrainingWorker` belirsizliği şöyle alıyor (widget:1993):

```python
try:
    _, y_std = model.predict(candidates, return_std=True)
    if y_std is None:
        y_std = np.ones(n_candidates)
except:                      # <-- çıplak except
    y_std = np.ones(n_candidates)
```

Hangi modelin gerçekten `return_std` verdiğini izledim (`UncertaintyWrapper`,
training_engine:363; sarmalananın ne olduğu her stratejinin `return` satırında):

| Model | Sarmalanan nesne | `return_std=True` sonucu | Durum |
|---|---|---|---|
| Random Forest | çıplak `RandomForestRegressor` (:291, :307) | `estimators_` üzerinden ağaç varyansı | ✅ çalışıyor |
| PyTorch DNN (dropout>0) | `PyTorchWrapper` (:1160) | MC-dropout, `n_mc_samples` | ✅ çalışıyor |
| **Gaussian Process** | **`TransformedTargetRegressor`** (:249, :270) | **`AttributeError`** | ❌ **bozuk** |
| MLP Regressor *(varsayılan)* | `TransformedTargetRegressor` | `TypeError` | ❌ sarmalanmamış |
| Gradient Boosting | `TransformedTargetRegressor` | `TypeError` | ❌ sarmalanmamış |

**GP bozukluğu doğrulandı** (sklearn 1.7.2):

```
ttr.predict(X, return_std=True)
-> AttributeError: 'tuple' object has no attribute 'ndim'
```

Sebep: `TransformedTargetRegressor.predict` `**predict_params`'ı alt regresöre
geçiriyor, GP `(mean, std)` tuple'ı döndürüyor, sonra TTR dönen değerin `.ndim`'ine
bakıp ters ölçeklemeye çalışıyor. `UncertaintyWrapper`'ın GP dalı bir TTR'yi
çıplak GP sanıyor.

**Sonucu:** hata çıplak `except:` tarafından yutuluyor, `y_std = np.ones(1000)`
oluyor. Sabit dizide `np.argsort(...)[-10:]` "en belirsiz 10 nokta" değil,
**gelişigüzel 10 LHS adayı** seçiyor. Yani:

> Varsayılan MLP'de, Gradient Boosting'de ve Gaussian Process'te "Adaptive
> Training (Active Learning)" butonu pratikte **rastgele örneklemeye** eşit.
> Benchmark'ımızda rastgele stratejinin kazancı **+%2.2** — yani gürültü.

Ek ironi: GUI'deki dropout uyarısı kullanıcıya *"Gaussian Process has built-in
uncertainty"* diye GP'yi öneriyor — bozuk olan tek yol.

Bu tek başına, komiteye geçmek için yeterli gerekçe: **komite akuizisyonu modelin
`return_std` desteğine hiç ihtiyaç duymaz.** (X, y) üzerine kendi GP+RF ikilisini
kurar. Yani "model belirsizlik veremiyor" durumu çıkmaz sokak olmaktan çıkıp
*en iyi* stratejiye dönüşür.

---

## 3. Diğer boşluklar (kum havuzu ↔ üretim)

| # | Konu | Üretimde | Kum havuzunda | Etki |
|---|---|---|---|---|
| 1 | Akuizisyon | saf std | `norm(std)*(0.3+0.7*norm(disagreement))` | +%14.7 → +%32.5 |
| 2 | Keşif tabanı | yok | `EXPLORE_FLOOR=0.3` | kenar/iğne dengesi |
| 3 | Çeşitlilik | yok — 10 nokta tek tepeye çökebilir | `diverse_top_k(min_dist=0.06)` | parti çeşitliliği |
| 4 | Aday havuzu | her tur **yeni** 1000 LHS, seed yok | sabit havuz + `taken` maskesi | zaten değerlendirilmiş noktanın komşusu tekrar seçilebilir |
| 5 | Sabitler | `n_rounds=5`, `n_candidates=1000`, `n_new=10` gömülü | argüman | ayarlanamıyor |
| 6 | Başarısız sim | `results.append(0.0)` | — | **veri setini zehirliyor** |
| 7 | Değerlendirme | `_evaluate_points` seri, `generate_data`'nın paralel yolundan kopya | — | kod tekrarı + yavaş |
| 8 | `stop_flag` | sadece tur başında | — | gerçek FEA'da tur = saatler |
| 9 | Katman | akuizisyon 97k satırlık GUI dosyasında | ayrı modül | test edilemez |
| 10 | Çok çıktılı | `y_std.mean(axis=1)` | tek çıktı, hiç test edilmedi | tanımsız davranış |

**6. madde özellikle aktif öğrenmede tehlikeli:** başarısız bir simülasyon veri
setine `y=0.0` olarak giriyor. Tek atışlık eğitimde bu sadece gürültü; aktif
öğrenmede döngü bu sahte noktanın yarattığı yapay süreksizliği "zor bölge" sanıp
**kalan bütçeyi oraya akıtır**. Başarısız noktalar atılmalı, uydurulmamalı.

**Ölçek tuzağı (3. madde):** kum havuzundaki `min_dist=0.06` birim küpte
(`[0,1]^d`) anlamlı. Üretimde sınırlar fiziksel birimlerde ve ölçekleri çok
farklı (kalınlık 1–3 mm, kuvvet 1000–50000 N). Mesafe filtresi **mutlaka birim
küpe normalize edilmiş uzayda** çalışmalı, yoksa büyük ölçekli değişken filtreyi
tek başına belirler.

---

## 4. Önerilen hedef mimari

```
pylcss/surrogate_modeling/active_learning.py        <-- YENİ, saf numpy/sklearn, GUI'siz
    ALConfig            (strategy, n_rounds, batch_size, n_candidates,
                         explore_floor, min_dist, random_state)
    acquisition_scores(strategy, X_train, y_train, X_pool, ...)
    diverse_top_k(scores, pool_unit, k, min_dist)
    select_next_batch(X_train, y_train, bounds, cfg) -> indeks/nokta dizisi
        └─ içeride kendi GP+RF komitesini kurar; kullanıcının modelinden bağımsız

pylcss/surrogate_modeling/training_engine.py
    SurrogateTrainer.evaluate_points(spy_code, ..., X)   <-- _evaluate_points'in
        generate_data'daki paralel yoldan türetilmiş, paylaşılan hali

pylcss/user_interface/surrogate/surrogate_training_widget.py
    AdaptiveTrainingWorker  <-- sadece orkestrasyon: döngü, sinyal, stop_flag
    GUI: strateji combo + parametre alanları
```

Neden komite kendi modelini kursun: kullanıcının modeli PyTorch da olabilir, MLP
de. Akuizisyonun kalitesi kullanıcının model seçimine bağlı olmamalı; ayrıca
GP+RF fit'i (100–500 nokta ölçeğinde) tek bir FEA çağrısının yanında bedava.

---

## 5. Uygulama adımları

1. **`active_learning.py` modülünü yaz** — kum havuzundaki `acquisition_scores`,
   `diverse_top_k`, `EXPLORE_FLOOR` mantığını taşı; birim küp normalizasyonu ekle;
   `bounds` ile fiziksel uzay arasında dönüşümü modül içinde hallet.
2. **`UncertaintyWrapper` GP hatasını düzelt** — TTR/Pipeline'ı açıp içteki
   `GaussianProcessRegressor`'a ulaş (`model.regressor_.named_steps['regressor']`)
   ve girdi ölçeklemesini elle uygula; ya da GP'yi TTR'ye sarmayı bırak.
   *Komiteye geçsek de bu bağımsız bir hata, ayrıca düzeltilmeli — `metrics['y_std']`
   ve belirsizlik grafikleri de aynı yoldan besleniyor.*
3. **Çıplak `except:`'leri daralt** — `except (TypeError, AttributeError)` + `logger.warning`.
   Sessiz düşüş yerine görünür uyarı; GUI'de "komite belirsizliğine geçildi" mesajı.
4. **`AdaptiveTrainingWorker`'ı yeniden bağla** — 2–4. adımları `select_next_batch`
   çağrısına indir; aday havuzunu sabitle + `taken` maskesi; her tur `stop_flag` kontrolü.
5. **Değerlendirmeyi birleştir** — `_evaluate_points`'i sil, `generate_data`'nın
   paralel değerlendiricisini ortak bir metoda çıkar; **başarısız noktaları at**,
   0.0 yazma; kaç noktanın düştüğünü kullanıcıya raporla.
6. **GUI parametreleri** — strateji combo (uncertainty / committee / random),
   tur sayısı, parti boyutu, aday sayısı, keşif tabanı. `get_config()`'e ekle.
7. **Dispatcher aracını güncelle** — `_adaptive_training()` butona tıklamak yerine
   strateji/bütçe parametresi alabilsin.
8. **Doğrulama** — önce sentetik spy model ile GUI üzerinden uçtan uca; sonra
   gerçek bir `cad.fea` iş akışıyla; RMSE eğrisini tur bazında kaydedip kum
   havuzundaki +%32.5'in gerçek çözücüde de görünüp görünmediğine bak.

**Sıralama önerisi:** 2 → 3 → 1 → 4 → 5 → 6 → 7 → 8. Önce hatayı düzelt (küçük,
bağımsız, hemen değer üretir), sonra mimariyi taşı.

---

## 6. Açık sorular

- **Çok çıktılı komite:** anlaşmazlık her çıktıda ayrı hesaplanıp ortalaması mı
  alınacak, yoksa çıktılar normalize edilip birleştirilecek mi? Kum havuzu bunu
  hiç test etmedi — üretime geçmeden 2-çıktılı bir sentetik testle karara bağlanmalı.
- **Tur başına yeniden eğitim maliyeti:** kullanıcı PyTorch seçtiyse 5 tur × 2000
  epoch. Akuizisyon için kullanıcının modelini eğitmeye gerek yok (komite kendi
  modelini kuruyor); kullanıcı modeli sadece **sonda bir kez** eğitilebilir. Bu,
  uyarlanabilir eğitimi belirgin şekilde hızlandırır.
- **Bütçe muhasebesi:** kullanıcıya "toplam kaç simülasyon" bütçesi mi sorulmalı,
  yoksa tur×parti mi? Bütçe daha anlaşılır bir arayüz.
