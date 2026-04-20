# Optik Form PDF Değerlendirme (Profesyonel OMR)

Bu çözüm, optik form PDF'lerini okuyup:
- işaretli alanları tespit eder,
- marker tabanlı hizalama ile kaymayı düzeltir,
- sonuçları CSV olarak kaydeder,
- görselleştirilmiş (annotated) PDF üretir.

## Güçlü teknik yaklaşım

- **Yüksek çözünürlükte render** (PyMuPDF, 300 DPI+)
- **Gürültü dayanıklı ön işleme** (bilateral + CLAHE + adaptif threshold)
- **Marker tabanlı sayfa hizalama** (4+ marker ile homography + RANSAC)
- **Balonlarda sıkı tespit** (lokal merkez araması)
- **İstatistik tabanlı karar** (z-score + minimum doluluk)
- **Belirsizlik yönetimi**: `ok`, `blank`, `multiple`

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Şablon tasarım önerisi (kritik)

En iyi sonuç için formun 4 köşesine dolu siyah kare marker yerleştirin.
- Boyut: 6-10 mm
- Kenarlardan sabit mesafe
- Yazı/çizgilerden ayrı alan

Bu marker'lar kayma, döndürme ve ölçek farklılıklarını düzeltmek için kullanılır.

## Kullanım

```bash
python omr_professional.py \
  --input /path/pdfs_or_file \
  --template template.example.yaml \
  --output output \
  --dpi 300 \
  --mark-z-threshold 1.0 \
  --min-fill-ratio 0.22
```

## Çıktılar

- `*_results.csv`: alan bazlı sonuç + güven + marker hizalama bilgisi
- `*_annotated.pdf`: işaretlerin çizildiği görsel çıktı
- `batch_summary.csv`: toplu işlem özeti

## Test ve kalibrasyon akışı

1. 20-50 örnek form ile pilot set oluşturun.
2. Yanlış pozitif/negatif örnekleri inceleyin.
3. `mark-z-threshold` ve `min-fill-ratio` değerlerini optimize edin.
4. Form baskısı değişirse şablon koordinatlarını güncelleyin.
