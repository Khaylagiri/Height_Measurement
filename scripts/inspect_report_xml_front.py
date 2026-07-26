from pathlib import Path
from zipfile import ZipFile
from lxml import etree

path = Path(r"D:\TA Skripsi\aplikasi\HeightMeasurement\output\documents\LAPORAN_SKRIPSI_ROBUSTNESS_SITASI_SELARAS_FINAL.docx")
ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
with ZipFile(path) as archive:
    root = etree.fromstring(archive.read("word/document.xml"))
paragraphs = root.xpath(".//w:body//w:p", namespaces=ns)
for i, paragraph in enumerate(paragraphs[:180]):
    text = "".join(paragraph.xpath(".//w:t/text()", namespaces=ns)).strip()
    if text:
        print(i, text)
