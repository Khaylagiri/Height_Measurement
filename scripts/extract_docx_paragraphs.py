from pathlib import Path
from zipfile import ZipFile
from lxml import etree

ROOT = Path(r"D:\TA Skripsi\aplikasi\HeightMeasurement\output\documents")
FILES = [
    "LAPORAN_SKRIPSI_ROBUSTNESS_SITASI_SELARAS.docx",
    "JURNAL_ROBUSTNESS_SITASI_SELARAS.docx",
]
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

for filename in FILES:
    with ZipFile(ROOT / filename) as archive:
        root = etree.fromstring(archive.read("word/document.xml"))
    paragraphs = []
    for paragraph in root.xpath(".//w:body//w:p", namespaces=NS):
        text = "".join(paragraph.xpath(".//w:t/text()", namespaces=NS)).strip()
        if text:
            paragraphs.append(text)
    print(f"\n===== {filename} =====")
    for text in paragraphs[-100:]:
        print(text)
