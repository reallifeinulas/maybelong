Aşağıdaki gereksinimlerle, boş bir repoya kopyalanabilir durumda **tam bir proje iskeleti** ve başlangıç kodlarını üret. Üretim bittiğinde dosya ağacını da göster. Kod dili Türkçe yorumlarla açık olmalı.

# Proje: Hybrid Metric-Constrained Reinforcement Trading Bot

## Genel Amaç
- Binance Futures **testnet** canlı verisinden (websocket: kline/aggTrade) bağlam (nötr teknik özellikler) çıkar.
- Karar kümesi: {LONG, SHORT, FLAT}. Karar veri-yönlü verilecek; FVG vb. DOĞRUDAN strateji sinyali **olmayacak**.
- ÇOK ÖLÇÜTLÜ kalite hedefleri: WinRate, Profit Factor, Sharpe, ROI, MDD **eşik/pencere** bazlı izlenir.
- Politika, risk-ayarlı **adım ödülü** + **kısıt cezaları** ile çevrimiçi uyarlanır (LinUCB veya SGDClassifier.partial_fit).
- Gerçek emir YOK; **simülasyon/paper-trade**. Komisyon ve slippage zorunlu. Minimum tutma süresi var.

## Teknoloji ve Kalite
- Python 3.11, asyncio.
- Bağımlılıklar: numpy, pandas, pyarrow, pyyaml, websockets, httpx, scikit-learn, ta, loguru, rich, pytest, pytest-asyncio, ruff, black.
- Modüler mimari; testler (pytest); format/lint (black/ruff); Dockerfile ve Makefile.
- Konfig dosyası: `config/settings.yaml`. Tüm sabitler buradan gelsin.

## Orta Seviye Başlangıç Profili (Balanced Baseline)
- **Pencereler**: WR=500 trade, PF=500 trade, Sharpe=1000 bar, ROI=30 gün, MDD=2000 bar.
- **Hedefler (τ)**: WR≥0.48, PF≥1.10, Sharpe≥0.5, ROI(30g)≥%2, MDD≤%15.
- **Ödül**: r_t = PnL_t − 0.3×std_50 (komisyon+slippage sonrası PnL; std_50 rolling).
- **Kısıt cezaları (Lagrange başlat)**: α_WR=0.5, α_PF=0.5, α_Sharpe=0.7, α_ROI=0.4, **α_MDD=1.0**; güncelleme: ihlal→×1.10, düzelme→×0.98; [0.1, 5.0] sınırı.
- **Keşif**: normal 0.20; küçük ihlal 0.10; büyük ihlal 0.03; asla 0’a düşürme.
- **Maliyet & tutma**: fee=2 bps, slip=1 bps, min_hold=5 bar.
- **Kill-switch**:
  - MDD>15% ⇒ pozisyon boyutu ×0.5
  - MDD>20% ⇒ FLAT
  - Sharpe(1000)<0 ⇒ model_weight −0.2 (rule_weight +0.2)
  - PF<0.9 **ve** WR<0.4 (rolling 300) ⇒ size ×0.7
  - ROI(30g)<0 ⇒ size ×0.7
  - Tek işlem zararı > %2 equity ⇒ 30 dk cooldown
  - Veri kopması >30 sn veya latency >5 sn ⇒ FLAT
- **Dinamik boyutlandırma**:
  size = base × clip(0.5 + 0.8(Sharpe−0.5) − 1.2(MDD−0.15)_+, 0, 3), base=1.0
- **Rejim profilleri (opsiyonel)**:
  - High-Vol: Sharpe hedefi 0.4, MDD sınırı 20%
  - Low-Vol: PF hedefi 1.2, ROI %3

## Dosya Yapısı (oluştur)
.
├─ src/
│ ├─ main.py
│ ├─ config/
│ │ └─ settings.py
│ ├─ data/
│ │ ├─ live_feed.py
│ │ ├─ storage.py
│ │ └─ feature_engineering.py
│ ├─ policy/
│ │ ├─ bandit.py
│ │ └─ constraints.py
│ ├─ signals/
│ │ └─ decision.py
│ ├─ execution/
│ │ ├─ simulator.py
│ │ └─ risk.py
│ ├─ evaluation/
│ │ ├─ metrics.py
│ │ └─ reporting.py
│ └─ utils/
│ ├─ logging.py
│ ├─ time.py
│ └─ types.py
├─ config/
│ └─ settings.yaml
├─ tests/
│ ├─ test_metrics.py
│ ├─ test_simulator.py
│ └─ test_policy.py
├─ data/ # çıktı klasörü (gitignore)
├─ artifacts/ # model/param persist
├─ .env.example
├─ .gitignore
├─ requirements.txt
├─ Dockerfile
├─ Makefile
└─ README.md
 

## Modül Sorumlulukları (özet; her dosyada docstring ve örnek kullanım belirt)
- `src/main.py`: asyncio pipeline: feed → features → policy → simulator → metrics → update. Ctrl+C’te zarif kapanış.
- `config/settings.py`: pydantic benzeri yapı ile `settings.yaml` yükleyip type-safe erişim.
- `data/live_feed.py`: Binance testnet kline/aggTrade websocket; reconnect/backoff; asyncio.Queue ile yayın.
- `data/feature_engineering.py`: NÖTR özellikler (EMA farkı, slope, ATR, realized_vol, volume_zscore, body/wick ratio, autocorr, regime proxy). **FVG, OB gibi doğrudan sinyal yok.**
- `data/storage.py`: Parquet/CSV append; atomic write; küçük sqlite sayaçlar opsiyon.
- `policy/bandit.py`: LinUCB (veya SGDClassifier + ε-greedy) ile {L,S,F} seçimi; exploration katsayısı kısıt sağlığına göre ayarlansın.
- `policy/constraints.py`: r_t hesapla; rolling metrikleri güncelle; kısıt ihlali ölç; Lagrange α’ları güncelle; J_t = r_t − ceza.
- `signals/decision.py`: kural ağırlığı vs model ağırlığı birleşimi; FLAT olasılığı ihlal durumunda artsın.
- `execution/simulator.py`: market fill + fee/slippage + min_hold + ATR tabanlı stop/TP; tek işlem zararı eşiği ve cooldown.
- `execution/risk.py`: dinamik boyutlandırma (size fonksiyonu); kill-switch kararları (MDD, Sharpe vb).
- `evaluation/metrics.py`: rolling WinRate, ProfitFactor, Sharpe, ROI(30g), MDD; hepsi pencere ve hedefleri `settings.yaml`’dan okusun. Anlık özet ve “uyarı bayrakları”.
- `evaluation/reporting.py`: Rich/Loguru ile canlı panel; ihlal/uyarı, pozisyon, equity eğrisi özetleri (metin tabanlı). (Grafik şart değil.)
- `utils/*`: saat/zaman, tipler, yapılandırılmış logger.

## config/settings.yaml Örneği (doldur ve kullan)
- runtime: symbols, timeframe, fee_bps, slippage_bps, min_hold_bars
- metrics: windows {winrate, profit_factor, sharpe, mdd, roi_days}, targets {…}, reward {lambda, vola_window}, penalties {wr, pf, sharpe, mdd, roi, alpha_cap, alpha_floor}
- sizing: base, kappa_max, beta0, beta_sharpe, beta_mdd
- safety: drawdown_soft, drawdown_hard, sharpe_floor, pf_wr_floor, roi_floor, cooldown_after_stop
- bandit: algo, base_exploration, mild_penalty, severe_penalty, recovery_rate
(Tam değerleri README’deki orta profil ile başlat.)

## README.md İçeriği (oluştur)
- Proje özeti, mimari diyagram metni, akış şeması (metin ile).
- Başlangıç hedefleri tablosu (WR/PF/Sharpe/ROI/MDD).
- Kurulum: venv, `pip install -r requirements.txt`, `python -m src.main`.
- Uyarı: Eğitim/AR-GE, yatırım tavsiyesi değildir.
- Yol haritası: rejim algılama otomasyonu, meta-bandit hiperparametre uyarlama, gerçek emir katmanı (opsiyon).

## Makefile
- `make init` (venv + pip install)
- `make run`
- `make test`
- `make fmt` (black/ruff)
- `make lint`

## Testler (iskele)
- `test_metrics.py`: sentetik PnL ile rolling WR/PF/Sharpe/MDD/ROI doğrulaması.
- `test_simulator.py`: fee/slippage/min_hold ve ATR stop/TP davranışı.
- `test_policy.py`: ihlal durumunda exploration azalması, kill-switch tetiklenmesi.

## Kurallar
- Kodlar çalışabilir iskelet olsun; karmaşık kısımlar için makul stub ve TODO bırakılabilir, ancak tüm dosyalar ve fonksiyon imzaları oluşturulmalı.
- Dışa bağımlı kimlik bilgisi gerektirmeyin (sadece public stream).
- Tüm sabitler `settings.yaml`’dan gelsin; sihirli sayı olmasın.
- Yorumlar net ve eğitim amaçlı olsun.
- Üretim bittiğinde **dosya ağacını** yazdır.

Şimdi projeyi oluştur.