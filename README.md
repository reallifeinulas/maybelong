# Hybrid Metric-Constrained Reinforcement Trading Bot

Bu proje, Binance Futures testnet üzerinden akan verilerle çevrimiçi uyarlanan, çok kriterli kısıtlarla yönetilen bir reinforcement öğrenme ticaret botu iskeletidir. Sistem, teknik olarak nötr özellikler kullanarak karar verir ve gerçek emir göndermek yerine simülasyon yapar.

## Mimari Özeti
- **Veri Katmanı**: `src/data/live_feed.py` websocket ile barları toplar, `feature_engineering.py` nötr özellikleri çıkarır, `storage.py` kalıcılaştırır.
- **Politika Katmanı**: `policy/bandit.py` ve `policy/constraints.py` ile LinUCB/SGD tabanlı seçim ve kısıt yönetimi uygulanır.
- **Sinyal Birleştirme**: `signals/decision.py` modeli ve kural ağırlıklarını harmanlar.
- **Yürütme**: `execution/simulator.py` paper-trade simülasyonu yapar, `execution/risk.py` kill-switch ve boyutlandırmayı yönetir.
- **Değerlendirme**: `evaluation/metrics.py` ve `evaluation/reporting.py` metrikleri hesaplayıp Rich ile raporlar.

### Akış Şeması (Metin)
1. Websocket akışı OHLCV barlarını üretir.
2. Barlar özellik mühendisliğine girer, nötr teknik göstergeler çıkar.
3. Politika banditi ve kısıt modülü ödül/ceza hesaplar.
4. Sinyal harmanlayıcı nihai karar olasılığını üretir.
5. Paper-trader simülasyonu PnL ve equity üretir.
6. Metrikler güncellenir, raporlanır ve risk kontrolleri tetiklenir.

## Başlangıç Hedefleri

| Metrik        | Pencere      | Hedef  |
|---------------|--------------|--------|
| WinRate       | 500 işlem    | ≥ 0.48 |
| Profit Factor | 500 işlem    | ≥ 1.10 |
| Sharpe        | 1000 bar     | ≥ 0.50 |
| ROI           | 30 gün       | ≥ %2   |
| MDD           | 2000 bar     | ≤ %15  |

## Kurulum
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.main
```

## Uyarı
Bu depo yalnızca eğitim ve AR-GE amaçlıdır; yatırım tavsiyesi değildir.

## Yol Haritası
- Volatilite rejim algısının otomatikleştirilmesi.
- Meta-bandit hiperparametre uyarlaması.
- Gerçek emir katmanının opsiyonel olarak eklenmesi.
