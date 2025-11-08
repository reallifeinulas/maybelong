# Hybrid Metric-Constrained Reinforcement Trading Bot

Bu depo, Binance US spot piyasasından kaydedilmiş gerçek 5 dakikalık OHLCV verileri ile çalışan, kısıt farkındalıklı bir reinforcement learning ticaret botu iskeletini içerir. Sistem gerçek emir göndermek yerine kağıt üzerinde işlem simülasyonu gerçekleştirir ve kararlarını teknik olarak nötr özelliklerden üretir.

> **Not:** Bu proje yalnızca eğitim ve araştırma amaçlıdır. Gerçek piyasalarda kullanım için tasarlanmamıştır ve yatırım tavsiyesi niteliği taşımaz.

## İçindekiler

1. [Mimari Genel Bakış](#mimari-genel-bakış)
2. [Kurulum](#kurulum)
3. [Çalıştırma](#çalıştırma)
4. [Yapılandırma](#yapılandırma)
5. [Testler](#testler)
6. [Proje Dizin Yapısı](#proje-dizin-yapısı)
7. [Yol Haritası](#yol-haritası)

## Mimari Genel Bakış

- **Veri Katmanı** (`src/data/`)
  - `live_feed.py`: Binance Futures'tan indirilen gerçek OHLCV barlarını CSV üzerinden yayınlayan ya da ihtiyaç halinde sentetik akış oluşturan yardımcıları içerir.
  - `feature_engineering.py`: OHLCV verisinden nötr faktörleri çıkarır.
- **Politika Katmanı** (`src/policy/`)
  - `bandit.py`: LinUCB/SGD tabanlı eylem seçimi yapar.
  - `constraints.py`: Performans metriklerini takip eder, ödül/ceza ve kısıt ihlali skorlarını hesaplar.
- **Sinyal Birleştirme** (`src/signals/decision.py`): Model çıktılarını kural tabanlı önyargılarla harmanlar.
- **Yürütme** (`src/execution/`)
  - `simulator.py`: İşlem sonuçlarını hesaplayan paper-trade motoru.
 - `risk.py`: Kill-switch kontrollerini ve dinamik pozisyon boyutlandırmasını uygular.
- **Değerlendirme** (`src/evaluation/`)
  - `metrics.py`: Temel performans metriklerini hesaplar.
  - `reporting.py`: Rich kullanarak terminale tablo halinde rapor yazar.

Pipeline, `runtime.symbols` altında listelenen tüm pariteler için aynı karar mantığını paralel olarak uygular ve her sembolün performans metriklerini ayrı ayrı izler.

### Veri Akışı
1. `BinanceLiveFeed` gerçek zamanlı barları üretir.
2. `compute_features` fonksiyonu teknik göstergeleri çıkartır.
3. `ConstraintAwareBandit` ve `ConstraintEvaluator` ödül/ceza sinyalleri oluşturur.
4. `DecisionBlender` nihai pozisyon önerisini belirler.
5. `PaperTrader` işlemleri simüle eder ve equity eğrisini günceller.
6. `RiskManager` en güncel Sharpe, maksimum gerileme ve ROI'yi yorumlayarak pozisyon boyutunu ve kill-switch durumunu üretir.
7. `LiveReporter` metrikleri anlık olarak ekrana yansıtır.

## Kurulum

Projeyi yerel ortamınıza almak için aşağıdaki adımları izleyin:

```bash
git clone https://github.com/<kullanici>/maybelong.git
cd maybelong
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Geliştirme ortamını kapatmak için `deactivate` komutunu çalıştırabilirsiniz.

## Çalıştırma

Varsayılan ayarlarla pipelini başlatmak:

```bash
python -m src.main
```

- Varsayılan kurulumda `data/binance_top10usdt_5m_recent.csv` dosyasındaki, son 24 saate ait hacimce büyük 10 USDT paritesinin gerçek Binance US spot verileri kullanılır; veri dosyası bittiğinde akış durur.
- Akışın daha kısa sürmesini isterseniz `run_pipeline` fonksiyonuna `max_steps` parametresi verilebilir (ör. testlerde olduğu gibi 50 adım).

## Yapılandırma

Tüm ayarlar `config/settings.yaml` dosyasında tutulur. Başlıca bloklar:

- `runtime`: Ana sembol(ler), zaman dilimi, komisyon/slippage varsayımları, minimum bar tutma süresi ve veri kaynağı seçimi.
- `metrics`: Hedef metrikler, rolling pencere boyutları ve ceza katsayıları.
- `sizing`: Sharpe ve maksimum gerilemeye duyarlı pozisyon boyutu formülü katsayıları.
- `safety`: Kill-switch için eşik değerleri ve soğuma süresi.
- `bandit`: Keşif oranı sınırları ve ceza durumundaki ayarlamalar.

Yapılandırmayı değiştirirken dosya formatını (YAML) koruduğunuzdan emin olun. Değişiklikler uygulama yeniden başlatıldığında otomatik olarak yüklenir.

## Testler

Tüm testleri çalıştırmak için:

```bash
pytest
```

- `tests/test_metrics.py`: Performans metriklerinin doğruluğunu sınar.
- `tests/test_policy.py`: Bandit keşif davranışını ve kısıt değerleyicisinin ROI hesabını kontrol eder.
- `tests/test_pipeline.py`: Uçtan uca pipeline'ın duman testini gerçekleştirir.
- `tests/test_data_feed.py`: CSV tabanlı gerçek veri akışının doğru okunduğunu kontrol eder.

## Proje Dizin Yapısı

```
maybelong/
├─ config/           # YAML tabanlı ayarlar
├─ src/
│  ├─ data/          # Veri akışı ve özellik mühendisliği
│  ├─ evaluation/    # Metrikler ve raporlama
│  ├─ execution/     # Simülatör ve risk yönetimi
│  ├─ policy/        # Bandit ve kısıt mantığı
│  ├─ signals/       # Karar harmanlama
│  └─ utils/         # Yardımcı tip ve zaman fonksiyonları
└─ tests/            # Pytest senaryoları
```

## Yol Haritası

- Volatilite rejim algısının otomatikleştirilmesi.
- Meta-bandit hiperparametre uyarlaması.
- Gerçek emir katmanının opsiyonel olarak eklenmesi.
