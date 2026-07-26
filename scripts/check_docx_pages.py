from pathlib import Path
from zipfile import ZipFile
from lxml import etree

ROOT = Path(r"D:\TA Skripsi\aplikasi\HeightMeasurement\output\documents")
for name in [
    "LAPORAN_SKRIPSI_ROBUSTNESS_SUDUT_30.docx",
    "LAPORAN_SKRIPSI_ROBUSTNESS_SITASI_SELARAS_FINAL.docx",
    "JURNAL_ROBUSTNESS_SUDUT_30.docx",
    "JURNAL_ROBUSTNESS_SITASI_SELARAS_FINAL.docx",
]:
    with ZipFile(ROOT / name) as archive:
        app = etree.fromstring(archive.read("docProps/app.xml"))
        pages = app.xpath("string(//*[local-name()='Pages'])")
        words = app.xpath("string(//*[local-name()='Words'])")
    print(name, "pages=", pages, "words=", words)
