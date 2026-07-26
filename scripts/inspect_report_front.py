from pathlib import Path
from docx import Document

path = Path(r"D:\TA Skripsi\aplikasi\HeightMeasurement\output\documents\LAPORAN_SKRIPSI_ROBUSTNESS_SITASI_SELARAS_FINAL.docx")
doc = Document(path)
for i, paragraph in enumerate(doc.paragraphs[:100]):
    if paragraph.text.strip():
        print(i, paragraph.text)
