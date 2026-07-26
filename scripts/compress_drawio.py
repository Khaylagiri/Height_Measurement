import base64
import sys
import urllib.parse
import zlib
from pathlib import Path
from xml.etree import ElementTree as ET


root = Path(r"D:\TA Skripsi\aplikasi\HeightMeasurement")
source = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "output" / "diagrams" / "blok_diagram_robustness_v2.drawio"
target = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "output" / "diagrams" / "blok_diagram_robustness_final.drawio"

tree = ET.parse(source)
mxfile = tree.getroot()

output_root = ET.Element(
    "mxfile",
    {
        "host": "app.diagrams.net",
        "agent": "Mozilla/5.0",
        "version": "24.7.17",
        "type": "device",
    },
)
for source_diagram in mxfile.findall("diagram"):
    model = source_diagram.find("mxGraphModel")
    model_xml = ET.tostring(model, encoding="unicode")

    # diagrams.net compressed format: encodeURIComponent -> raw DEFLATE -> Base64.
    encoded = urllib.parse.quote(model_xml, safe="~()*!.'-")
    compressor = zlib.compressobj(level=9, wbits=-15)
    compressed = compressor.compress(encoded.encode("utf-8")) + compressor.flush()
    payload = base64.b64encode(compressed).decode("ascii")

    output_diagram = ET.SubElement(
        output_root,
        "diagram",
        {
            "id": source_diagram.get("id", "page-1"),
            "name": source_diagram.get("name", "Page-1"),
        },
    )
    output_diagram.text = payload

    # Round-trip validation.
    decoded = zlib.decompress(base64.b64decode(payload), wbits=-15).decode("utf-8")
    restored = urllib.parse.unquote(decoded)
    ET.fromstring(restored)

ET.ElementTree(output_root).write(target, encoding="UTF-8", xml_declaration=True)
print(target)
