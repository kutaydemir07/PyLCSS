# PyLCSS — Aktif Öğrenme Çalışması: Yapay Zekaya Devir Prompt'u

> Bu dosyayı yeni bir AI oturumunun başına yapıştır. Projenin durumu, yapılanlar
> ve sıradaki iş burada tam olarak anlatılıyor.

---

## PROMPT (kopyala-yapıştır)

Sen deneyimli bir Python/ML mühendisisin. **PyLCSS** adlı bir masaüstü mühendislik
simülasyon ve optimizasyon platformu üzerinde çalışıyorsun. Proje kökü: `C:\dev\PyLCSS`.

### Proje bağlamı

PyLCSS (Python Low-Code System Solutions), node tabanlı görsel arayüzle çok
disiplinli mühendislik sistemlerinin modellenip analiz edildiği, tek bir masaüstü
uygulaması içinde çalışan bir ürün geliştirme ortamıdır. Telif: Kutay Demir,
lisans PolyForm Shield 1.0.0. Python 3.10+, PySide6 GUI, Windows hedefli.

Ana modüller (`pylcss/` altında):

| Modül | İşlev |
|---|---|
| `design_studio/` | Parametrik CAD (CadQuery), FreeCAD köprüsü, FEA (CalculiX), crash (OpenRadioss), topoloji optimizasyonu (pyMOTO) |
| `solution_space/` | Zimmermann & von Hoessle (2013) çözüm uzayı yaklaşımı — tek optimum yerine geçerli tasarım kutuları |
| `optimization/` | 7 çözücü: SLSQP, COBYLA, trust-constr, Differential Evolution, Nevergrad, NSGA-II, Multi-Start |
| `sensitivity/` | Sobol, Morris, FAST, Delta (DMIM) |
| `surrogate_modeling/` | **Vekil (surrogate) model eğitimi** — MLP, Random Forest, Gradient Boosting, Gaussian Process; ayrıca PyTorch, GeomDeepONet, GINO |
| `system_modeling/` | Graf tabanlı fonksiyonel mimari editörü |
| `assistant_systems/` | 25 araçlı PydanticAI ajanı, çok sağlayıcılı LLM desteği |
| `user_interface/` | PySide6 widget'ları |

### Üzerinde çalıştığımız problem

Vekil model eğitimi PyLCSS'in en pahalı adımı: her eğitim örneği bir FEA veya
crash simülasyonu demek (dakikalar–saatler). Mevcut boru hattı
(`pylcss/surrogate_modeling/training_engine.py`) **tek atışlık (one-shot)**:
LHS ile bir kere örnekle → her noktada pahalı simülasyonu koştur → modeli fit et.

Bu, bütçeyi tasarım uzayına **düzgün** dağıtır. Ama gerçek FEA/crash tepkileri
çoğunlukla pürüzsüzdür ve içlerinde **keskin geçişler** barındırır (burkulma mod
değişimi, temas başlangıcı). Düzgün örnekleme bu uçurumları kaçırır; bütçenin
büyük kısmı zaten kolay olan pürüzsüz bölgeye gider.

**Hedef:** aynı simülasyon bütçesiyle daha doğru vekil model — yani *aktif
öğrenme* / uyarlanabilir örnekleme.

### PyLCSS'te hâlihazırda ne var (mevcut, zayıf sürüm)

`pylcss/user_interface/surrogate/surrogate_training_widget.py` içinde
`AdaptiveTrainingWorker` sınıfı ("Adaptive Training (Active Learning)" butonu,
~satır 530 ve ~1940). Mantığı:

1. 5 tur döngü
2. Her turda: modeli eğit → 1000 LHS aday üret → `predict(return_std=True)`
   ile belirsizliği tahmin et → **en yüksek std'li 10 noktayı** seç → değerlendir → veri setine ekle
3. Sonda tam veri setiyle final eğitim

Zayıf yanları: sadece saf belirsizlik (std) kriteri; çeşitlilik filtresi yok
(parti tek bir noktaya çökebilir); keşif tabanı yok; std desteklemeyen modellerde
sessizce rastgeleye düşüyor; hiçbir zaman ölçülüp doğrulanmamış.

### `experiments/active_learning/` içinde ne geliştirdik (yeni, ölçülmüş sürüm)

Üretim kodunu bozmadan, ground-truth'u bilinen sentetik bir fonksiyonla
izole bir kum havuzunda çalıştık. Üç dosya:

**`baseline.py` — Faz 1: statik referans**
- `expensive_function(X)`: pürüzsüz çanak `sum((x-0.5)^2)` + **keskin uçurum**
  `1.5*tanh((mean(x)-0.75)/0.03)` — crash benzeri rejim değişimini taklit eder.
  Entegrasyonda bunun yerine gerçek `cad.fea` / `cad.crash` çağrısı konacak,
  geri kalan her şey aynı kalır.
- `lhs_sample`, `build_surrogate` (mlp/gp/rf — PyLCSS ile aynı ölçekleme
  stratejisi: hem girdi hem hedef `StandardScaler`), `evaluate` (RMSE, R²),
  `run_baseline`.

**`active_learning.py` — Faz 2: kendi kendine eğitim döngüsü, 4 strateji**
- `ALGP`: dürüst belirsizlik veren GP (`ConstantKernel * RBF + WhiteKernel`,
  `normalize_y`, 3 restart) — AL motoru.
- Akuizisyon stratejileri (`acquisition_scores`):
  - `uncertainty` (v1): skor = GP tahmin std'si
  - `gradient` (v2a): skor = `norm(std) * (0.3 + 0.7*norm(|grad(mu)|))` —
    gradyan merkezi farklarla vekil modelin ortalaması üzerinden hesaplanır
    (pahalı simülasyon gerektirmez, çağrı başına 2·d ekstra `predict`)
  - `committee` (v2b): skor = `norm(std) * (0.3 + 0.7*norm(|GP - RF|))` —
    model anlaşmazlığı
  - `random`: ablasyon kontrolü
- **`EXPLORE_FLOOR = 0.3`**: skorun %30'u saf keşif olarak kalır. Gerekçe: düz
  std "veriye uzaklığı" ölçer, "zorluğu" değil — bütçeyi domain kenarlarına
  harcar; ama tam sömürüye geçmek de samanlıktaki iğneyi kaçırır.
- **`diverse_top_k`**: skora göre açgözlü seçim + `min_dist=0.06` filtresi, parti
  tek bir sıcak noktaya çökmesin diye.
- `cliff_hits`: seçilen noktaların kaçının uçurum bandına düştüğünü sayan tanı aracı.

**`benchmark.py` — Faz 2.5: çok tohumlu karşılaştırma**
- 12 tohum × 4 strateji + statik LHS, eşit bütçe (20 başlangıç + 8×10 = **100 simülasyon**).
- Ortalama ± std, LHS'e karşı kazanma oranı, % iyileşme, "LHS'in final RMSE'sini
  kaç simülasyonda yakaladı" (`sims-to-match`) raporlar; `benchmark_results.csv` yazar.

### Ölçülen sonuçlar (12 tohum, 100 simülasyon bütçesi, final RMSE — düşük iyi)

| Strateji | RMSE | LHS'e karşı galibiyet | İyileşme | Eşleşme |
|---|---|---|---|---|
| LHS (statik) | 0.1591 ± 0.0405 | — | — | — |
| uncertainty | 0.1357 ± 0.0175 | 8/12 | +14.7% | 80 sim (8/12) |
| gradient | 0.1505 ± 0.0396 | 7/12 | +5.4% | 75 sim (8/12) |
| **committee** | **0.1074 ± 0.0136** | **12/12** | **+32.5%** | 78 sim (12/12) |
| random (kontrol) | 0.1556 ± 0.0273 | 5/12 | +2.2% | 72 sim (5/12) |

**Sonuç:** komite (GP–RF anlaşmazlığı) net kazanan — 12 tohumun 12'sinde LHS'i
yendi, hatayı %32.5 düşürdü ve **varyansı üçte bire indirdi** (0.0405 → 0.0136),
ki mühendislik açısından ortalama kazançtan bile değerli. `random`'ın +%2'de
kalması, kazancın döngünün kendisinden değil akuizisyon fonksiyonundan geldiğini
kanıtlar. Gradyan varyantı beklentinin altında kaldı.

### Sıradaki iş (senin görevin)

Kum havuzunda kanıtlanan **committee** stratejisini üretime taşı:

1. `AdaptiveTrainingWorker`'ı komite akuizisyonuna geçir: mevcut "en yüksek 10
   std" seçimini `norm(std) * (0.3 + 0.7*norm(|GP - RF|))` skoru + `diverse_top_k`
   mesafe filtresi ile değiştir.
2. Akuizisyon mantığını GUI'den ayır — `pylcss/surrogate_modeling/` altında test
   edilebilir bir modüle (ör. `active_learning.py`) taşı; widget sadece çağırsın.
3. Strateji seçimini kullanıcıya aç (uncertainty / committee / random) ve
   `EXPLORE_FLOOR`, parti boyutu, tur sayısı, `min_dist` parametrelerini
   yapılandırılabilir yap.
4. Std üretemeyen modelleri düzgün ele al: sessizce rastgeleye düşmek yerine
   komite anlaşmazlığını yedek belirsizlik kaynağı olarak kullan.
5. `pylcss/assistant_systems/api/dispatcher.py:1377`'deki AI ajanı aracını yeni
   parametrelerle güncelle.
6. Sentetik yerine gerçek bir `cad.fea` iş akışıyla uçtan uca doğrula.

Kod tarzı: mevcut PyLCSS kalıplarını izle — GUI iş parçacıkları için QThread +
Signal, ölçekleme için `TransformedTargetRegressor`, `training_engine.py`'deki
strateji-sınıfı deseni.

---

## Notlar

- `benchmark_results.csv` proje kökünde (12 satır, tohum başına ham RMSE'ler).
- `experiments/` klasörü henüz git'e eklenmemiş (untracked).
- Kum havuzu PyLCSS'i import etmez; `experiments/active_learning/` içinde
  `python benchmark.py` ile bağımsız çalışır (~1-2 dk).

---

## 22 Temmuz 2026 — Üretim entegrasyonu tamamlandı

Bu devir prompt'undaki 1–6 numaralı üretim işleri `feature/active-learning`
dalında uygulandı. Dal önce GitHub'daki güncel `main` (`f0a92e2`, v1.4.0)
üzerine fast-forward edildi.

Yapılanlar:

- `pylcss/surrogate_modeling/active_learning.py` eklendi: doğrulanmış GP–RF
  committee skoru, sabit/tohumlu LHS aday havuzu, `[0,1]^d` uzayında
  `diverse_top_k`, çok çıktıda çıktı-başına normalizasyon ve yapılandırma sınıfı.
- `uncertainty` stratejisinde seçili model anlamlı std üretemezse rastgeleye
  sessiz düşmek yerine GP–RF anlaşmazlığı açıkça yedek kaynak olarak kullanılıyor.
- GP + `TransformedTargetRegressor` std hatası düzeltildi; ortalama ve std hem
  girdi hem hedef ölçekleyicilerinden açılarak mühendislik birimlerine dönüyor.
- `AdaptiveTrainingWorker` yalnızca orkestrasyon yapıyor. Committee/random
  turlarında kullanıcının pahalı final modeli tekrar tekrar eğitilmiyor; başarısız
  simülasyonlar `0.0` olarak uydurulmak yerine atlanıp raporlanıyor.
- GUI'ye strategy, tur, parti, aday havuzu, explore floor ve normalize min-distance
  kontrolleri eklendi; proje ayarlarıyla kaydedilip yükleniyor.
- AI asistanına parametreli `adaptive_training` aracı eklendi.
- NodeGraphQt'nin kaydedilmiş node ID'lerini yüklemede değiştirmesi yüzünden
  `cad.fea(_settings=...)` çalışmıyordu. Runtime'a saved-ID → runtime-node eşlemesi
  eklendi; keşfedilen ayar anahtarları artık yeniden uygulanabiliyor.

Doğrulama:

- `tests/` altında 10 test geçti: committee formülü, fiziksel ölçekten bağımsız
  mesafe, çok çıktılı birim bağımsızlığı, sabit havuz, açık fallback, GP std
  dönüşümü, başarısız sim atlama ve CAD saved-ID ayarı.
- Qt offscreen smoke testinde GUI varsayılanları ve programatik override'lar geçti.
- Gerçek CalculiX doğrulaması (`experiments/active_learning/real_fea_validation.py`):
  `data/cad_environment/01_fea/03_fea_plate_solution_space.cad` basıncı üzerinde 4 başlangıç + 2 tur × 2 örnek = 8 eğitim
  FEA koşusu; 3 bağımsız FEA test noktasında **RMSE 0.07631, R² 0.99733**.

Sıradaki bilimsel iş: bu plaka problemi yaklaşık lineer olduğu için yalnızca
entegrasyonu kanıtlar. Committee'nin gerçek dünyada LHS'e karşı kazancını ölçmek
için burkulma/temas gibi rejim değişimi içeren bir CAD çalışmasında eşit bütçeli,
çok tohumlu benchmark çalıştırılmalı.

---

## 23 Temmuz 2026 — Nonlinear FEA mimari benchmark'ı tamamlandı

Önceki “gerçek nonlinear benchmark” açığı kapatıldı:

- `data/cad_environment/01_fea/04_nonlinear_fea_benchmark_plate.cad` eklendi. Delikli çelik plaka
  CalculiX `NLGEOM` + bilinear plastik malzemeyle çözülüyor.
- Üç tasarım girdisi kullanıldı: basınç `15–160 MPa`, kalınlık `6–14 mm`,
  merkez delik yarıçapı `12–28 mm`.
- İki çıktı birlikte öğrenildi: maksimum von Mises gerilmesi ve tepe deplasmanı.
- 80 tasarım havuzu + eğitimde hiç görülmeyen 24 test noktası olmak üzere
  **104/104 gerçek FEA yakınsadı**; 14 nokta akma/plastik rejimine girdi.
- Gaussian Process, Random Forest, Gradient Boosting, sklearn MLP ve PyTorch DNN;
  16/32/64 FEA bütçesinde 5 tohumla, toplam 75 mimari koşusunda karşılaştırıldı.
- Gaussian Process üç bütçede de kazandı:
  - 16 FEA: aggregate NRMSE `0.2110`, aggregate R² `0.9429`
  - 32 FEA: aggregate NRMSE `0.1745`, aggregate R² `0.9609`
  - 64 FEA: aggregate NRMSE `0.1690`, aggregate R² `0.9629`
- 32 FEA eşit bütçesinde static maximin ile GP–RF committee 5 tohum × 5 final
  mimaride karşılaştırıldı. Committee yalnızca Gradient Boosting'de küçük
  (`+%1.7`) kazanç verdi; 5 mimarinin 4'ünde kaybetti ve mimari ortalamasında
  NRMSE'yi `%5.5` kötüleştirdi.
- Teşhis: committee ortalama 6.4 plastik nokta seçerek geçişe odaklandı; static
  maximin 3.8 plastik noktayla yetindi ama test uzayını daha iyi kapladı
  (ortalama normalize en-yakın mesafe `0.170` vs `0.193`).

Ürün guideline'ı:

- Düşük boyutlu ve global olarak düzgün scalar FEA tepkisinde güvenli başlangıç:
  **static maximin/LHS + Gaussian Process**.
- Committee evrensel varsayılan olarak sunulmamalı. Temas, burkulma, kırılma gibi
  gerçek lokal rejim değişimi biliniyorsa veya küçük bir pilot benchmark tekrar
  eden kazanç gösteriyorsa etkinleştirilmeli.
- GINO ve Geom-DeepONet nodal alan modelleridir; scalar gerilme/deplasman yarışıyla
  doğrudan karşılaştırılmadı ve ayrı field-surrogate benchmark gerektirir.

Tekrarlanabilir çalışma:

- Runner: `experiments/active_learning/nonlinear_fea_benchmark.py`
- Veri ve ham sonuçlar: `experiments/active_learning/results/nonlinear_fea/`
- Okunabilir rapor: `experiments/active_learning/results/nonlinear_fea/REPORT.md`
- Toplam test durumu: **14 geçti**.

Bu scalar FEA araştırma fazı kapanmıştır. Sıradaki ürün işi Design Studio
modelini Modeling Environment içinde giriş/çıkış portlu bir fonksiyon bloğu
olarak sunan UI/backend köprüsüdür.

---

## 23 Temmuz 2026 — Design Studio → Modeling Environment Faz 1 tamamlandı

Design Studio analizlerini Modeling Environment içinde elle kod yazmadan
kullanılabilir hâle getiren profesyonel köprü tamamlandı:

- Design Studio araç çubuğuna **Create Function** eklendi. Aktif çalışma önce
  atomik olarak kaydediliyor, ardından arayüz seçim ekranı açılıyor.
- Modeling Environment araç çubuğuna **Design Studio** eylemi eklendi; daha önce
  kaydedilmiş bir `.cad` dosyası buradan da içe aktarılabiliyor.
- Yeni seçim ekranı çalışma içindeki geometri parametrelerini; malzeme, mesh,
  sınır şartı, yük ve solver kontrollerini; ayrıca standart FEA/crash/TopOpt
  sonuçlarını otomatik keşfediyor.
- Geometri parametreleri varsayılan tasarım değişkenleri olarak seçiliyor.
  Malzeme/mesh/yük/solver kontrolleri yanlışlıkla optimizasyon değişkenine
  dönüşmemeleri için kullanıcı seçene kadar kapalı kalıyor.
- Yeni `Design Studio Simulation` node'u kaynak dosya, analiz türü ve arayüz
  sözleşmesini kendi içinde saklıyor. Seçilen girişler için Design Variable,
  sonuçlar için Quantity of Interest node'ları oluşturulup otomatik bağlanıyor.
- Node, derleme sırasında mevcut `cad.fea`, `cad.crash` veya `cad.topopt`
  runtime'ını çağırıyor; dolayısıyla görsel bağlantı gerçek mühendislik
  analizine bağlı. Sistem dosyası kaydetme/geri yükleme de destekleniyor.

Ana dosyalar:

- `pylcss/system_modeling/design_studio_bridge.py`
- `pylcss/user_interface/system_modeling/design_studio_bridge_dialog.py`
- `pylcss/user_interface/system_modeling/system_node_types.py`
- `pylcss/user_interface/system_modeling/system_modeling_widget.py`
- `pylcss/user_interface/cad/cad_widget.py`
- `pylcss/user_interface/main_application_window.py`
- `tests/test_design_studio_bridge.py`

Doğrulama:

- Köprüye özel **5/5 test** geçti: study keşfi, parametre/ayar kod üretimi,
  dialog varsayılanları, bağlı/derlenebilir grafik ve sistem dosyası round-trip.
- Tüm proje testleri: **19/19 geçti**.
- Otomatik oluşturulan Modeling Environment sistemi gerçek
  `04_nonlinear_fea_benchmark_plate.cad` çalışmasını CalculiX ile koşturdu:
  maksimum gerilme `72.367465 MPa`, tepe deplasmanı `0.00515961 mm`,
  kütle `0.000678312 tonne` (yaklaşık `0.678 kg`).

Bu fazda crash benchmark'ı veya diğer uygulama ekranları değiştirilmedi.

---

## 27 Temmuz 2026 — Hard-nonlinear FEA qualification tamamlandı

Önceki delikli plaka çalışmasının global olarak fazla düzgün olması nedeniyle,
committee yönteminin tasarlandığı gerçek limit-point rejiminde ikinci ve daha
sert bir benchmark yapıldı:

- Gerçek CalculiX 2.23 çözümleriyle, B31 beam elemanlı, geometrik kusurlu ve
  deplasman kontrollü sığ kemer kuruldu. `NLGEOM` çözümü tepe kuvveti sonrasında
  negatif teğet rijitlik ve belirgin snap-through davranışı üretiyor; yapay
  gerilme/cevap tavanı kullanılmıyor.
- Dört girdi: kemer yükseliği `6–18 mm`, kalınlık `1–3 mm`, deplasman/yükselti
  oranı `0.10–1.40`, kusur oranı `-0.03–0.03`.
- Dört çıktı: son aktüatör kuvveti, limit öncesi tepe kuvveti, tepe kuvveti
  deplasmanı ve işaretli şekil değiştirme enerjisi.
- Analitik iki-çubuk korelasyon NRMSE'si `%0.408`; 40→80 eleman mesh farkı en
  fazla `%0.853`; `0.02→0.01` increment farkı en fazla `%0.867`; pilot kuvvet
  düşüşü `%88.90`. Dört ön kabul kriterinin tamamı geçti.
- Bağımsız LHS tasarımıyla `160` seçim havuzu + eğitimden tamamen saklı `64`
  holdout üretildi: **224/224 gerçek FEA başarılı**, başarısız çözüm yok.
- Maximin ve committee aynı ilk 12 maximin noktayı, aynı son Gaussian Process'i,
  aynı FEA bütçesini ve aynı holdout'u kullandı. Akuizisyon yalnızca daha önce
  seçilmiş noktaların etiketlerini görebildi.
- `20` bağımsız seçim tohumu ve `16/24/32/48/64` FEA bütçesiyle toplam `200`
  eşit-bütçeli final model değerlendirmesi yapıldı. İstatistik birimi seed'dir.

Eşleşmiş aggregate NRMSE sonuçları:

| Bütçe | Maximin | Committee | Ortalama eşleşmiş kazanç | %95 bootstrap CI | Galibiyet | Katı karar |
|---:|---:|---:|---:|---:|---:|:---:|
| 16 | 0.2235 | 0.2297 | -%3.96 | [-%12.94, +%4.30] | 8/20 | FAIL |
| 24 | 0.1320 | 0.1312 | -%1.20 | [-%9.54, +%7.85] | 9/20 | FAIL |
| 32 | 0.1006 | 0.0892 | +%6.64 | [-%5.01, +%17.58] | 11/20 | FAIL |
| 48 | 0.0689 | 0.0573 | +%14.99 | [+%9.12, +%21.05] | 19/20 | FAIL* |
| 64 | 0.0489 | 0.0435 | +%10.53 | [+%6.40, +%14.64] | 18/20 | **PASS*** |

Katı doğrulama için dört kapının birlikte geçmesi önceden şart koşuldu: ortalama
eşleşmiş kazanç en az `%5`, bootstrap CI alt sınırı sıfırdan büyük, seed
galibiyeti en az `%60` ve transition-band NRMSE kötüleşmiyor. `48` FEA'da ilk üç
kapı güçlü biçimde geçti fakat yalnızca beş transition holdout noktasındaki NRMSE
`0.07429 → 0.07523` olduğu için kapı FAIL kaldı. `64` FEA'da dört kapının tamamı
geçti (`transition 0.05176 → 0.04452`). Ancak önceden ürün kararı için seçilen
birincil bütçe `32` FEA'dır; diğer bütçelere aynı kapıları uygulamak ikincil
bütçe-bazlı analizdir ve başarısız birincil kararı değiştirmez. `32` FEA'da CI
sıfırı kesiyor ve yalnızca 11/20 seed kazanıyor; bu bütçedeki avantaj umut verici
olsa da kanıtlanmış sayılmıyor.

Nihai ürün guideline'ı:

- `32` FEA'ya kadar güvenli varsayılan: **static maximin + Gaussian Process**.
- GP–RF committee evrensel varsayılan değildir; bilinen hard-nonlinear
  limit-point probleminde ve yeterli bütçede opt-in kullanılmalıdır.
- İkincil analizde committee hard-nonlinear rejimde `64` FEA bütçesinde aynı dört
  qualification kapısını geçti. Test edilen eğri, savunulabilir bir eş-hata
  kesişimi içermediği için kesin bir “FEA sayısını yüzde X azalttı” iddiası
  yapılmıyor.
- Sonuç contact, fracture, topology change veya yüksek boyutlu alan vekilleri
  için evrenselleştirilemez; bunlar ayrı qualification kapsamlarıdır.

Tekrarlanabilir kanıt paketi:

- Runner: `experiments/active_learning/hard_nonlinear_fea_benchmark.py`
- Otomatik testler: `tests/test_hard_nonlinear_fea_benchmark.py`
- Veri/ham solver dosyaları: `experiments/active_learning/results/hard_nonlinear_snapthrough/`
- Mühendislik raporu: `experiments/active_learning/results/hard_nonlinear_snapthrough/REPORT.md`
- Solver, commit, paket sürümleri, veri ve kaynak SHA-256 kayıtları:
  `experiments/active_learning/results/hard_nonlinear_snapthrough/provenance.json`

### 64 FEA gerçekten 100 FEA'nın yerini tutuyor mu?

Bu iddia ayrıca önceden sabitlenen non-inferiority kriterleriyle doğrudan test
edildi. `64 committee + GP`, aynı 20 seed ve aynı saklı 64 gerçek-FEA holdout
üzerinde `64/72/80/88/96/100 maximin + GP` referanslarıyla karşılaştırıldı.
Kabul için aggregate bozulmanın tek yönlü %95 üst sınırı en fazla `%10`, her
çıktının ortalama bozulması en fazla `%15`, transition bozulması en fazla `%10`,
mutlak aggregate NRMSE en fazla `0.05` ve R² en az `0.99` olarak koşuldu.

- Committee-64 mutlak olarak güçlü kaldı: NRMSE `0.04353`, R² `0.99792`.
- Maximin-100 NRMSE `0.03780`; committee-64 ortalama `%15.19` daha hatalı ve
  tek yönlü %95 üst sınır `%18.32`. Ayrıca transition hatası `%39.68` daha yüksek.
  Bu nedenle **64 → 100 eşdeğerliği ve %36 tasarruf iddiası reddedildi**.
- Tüm kapıları geçen en yüksek referans `maximin-80` oldu: NRMSE `0.04500`;
  committee-64 ortalamada `%3.03` daha iyi, üst sınır yalnızca `+%0.30` ve
  transition farkı `+%2.64`.
- Savunulabilir mevcut verimlilik iddiası: **80 yerine 64 gerçek FEA**, yani
  doğrulanmış **%20 FEA azalması**. Bu sonuç yalnızca mevcut hard-nonlinear scalar
  benchmark kapsamındadır.

Kanıt paketi:

- Rapor: `experiments/active_learning/results/hard_nonlinear_snapthrough/replacement_test/REPORT.md`
- Ham 20-seed sonuçları: `replacement_test/replacement_runs.csv`
- İstatistik/kapılar: `replacement_test/replacement_statistics.csv`
- İzlenebilirlik: `replacement_test/provenance.json`
