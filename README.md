# TazeMi? — Domates Tazelik Sınıflandırıcı

Domates fotoğrafından **Taze / Yenebilir / Çürük** tahmini yapan hibrit makine öğrenmesi uygulaması.

## Model Mimarisi

```
MobileNetV2 (dondurulmuş, 1280-dim)
  → StandardScaler → PCA(64)
  → MiniSom(3×1)   [topoloji görselleştirmesi]
  → SVM-RBF        [sınıflandırıcı]
```

Test doğruluğu: **%97.9** · F1 (macro): **0.98** · Veri seti: 4.287 görsel

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Kullanım

```bash
# Arayüzü başlat
streamlit run app.py

# Modeli eğit (ilk çalıştırmada CNN özellikleri cache'lenir)
python train_hybrid.py

# CNN özelliklerini yeniden çıkar ve yeniden eğit
python train_hybrid.py --force
```

## Veri Seti

`level_1/`, `level_2/`, `level_3/` klasörlerine görseller yerleştirilmeli:

| Klasör  | Sınıf      | Görsel Sayısı |
|---------|------------|---------------|
| level_1 | Taze       | 1.416         |
| level_2 | Yenebilir  | 1.279         |
| level_3 | Çürük      | 1.592         |

## Claude AI

Uygulama, Claude Haiku Vision ile Türkçe analiz ve tarif önerisi sunar.  
`ANTHROPIC_API_KEY` ortam değişkenini ayarlayın veya arayüzden girin.

## Proje Yapısı

```
src/
  config.py             # Merkezi konfigürasyon
  data_loader.py        # Veri yükleme ve split
  feature_extractor.py  # MobileNetV2 özellik çıkarıcı
  hybrid_model.py       # HybridCNNSOM sınıfı
  evaluate.py           # Metrik hesaplama ve grafikler
app.py                  # Streamlit arayüzü
train_hybrid.py         # Eğitim pipeline'ı
models/                 # Kayıtlı model (.pkl)
outputs/                # Çıktı grafikleri
```
