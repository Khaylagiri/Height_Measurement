from pathlib import Path
from docx import Document

ROOT = Path(r"D:\TA Skripsi\aplikasi\HeightMeasurement\output\documents")
CASES = [
    ("LAPORAN_SKRIPSI_ROBUSTNESS_SUDUT_30.docx", [85, 86, 87, 88]),
    ("LAPORAN_SKRIPSI_ROBUSTNESS_SITASI_SELARAS_FINAL.docx", [85, 86, 87, 88]),
    ("JURNAL_ROBUSTNESS_SUDUT_30.docx", [24, 25, 26, 27, 30]),
    ("JURNAL_ROBUSTNESS_SITASI_SELARAS_FINAL.docx", [24, 25, 26, 27, 30]),
]

for filename, indexes in CASES:
    doc = Document(ROOT / filename)
    print(f"\n{filename}")
    page = 1
    for i, paragraph in enumerate(doc.paragraphs):
        if i in indexes:
            print(
                f"p{i}: page-marker={page}; style={paragraph.style.name}; "
                f"align={paragraph.alignment}; before={paragraph.paragraph_format.space_before}; "
                f"after={paragraph.paragraph_format.space_after}; lines={paragraph.paragraph_format.line_spacing}"
            )
        page += len(paragraph._p.xpath(".//w:lastRenderedPageBreak"))
