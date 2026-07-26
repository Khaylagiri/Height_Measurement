from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(r"D:\TA Skripsi\aplikasi\HeightMeasurement")
DIAGRAMS = ROOT / "output" / "diagrams"
LEFT = DIAGRAMS / "flowchart_1_perspective_correction_v1.drawio"
RIGHT = DIAGRAMS / "flowchart_2_pose_dan_pengukuran_v1.drawio"
OUTPUT = DIAGRAMS / "flowchart_sistem_dua_bagian_v1.drawio"
RIGHT_OFFSET_X = 850.0


def model_from(path):
    root = ET.parse(path).getroot()
    return root.find("diagram").find("mxGraphModel")


left_model = deepcopy(model_from(LEFT))
right_model = model_from(RIGHT)
left_model.set("pageWidth", "1700")
left_model.set("pageHeight", "1600")
left_root = left_model.find("root")
right_root = right_model.find("root")

for cell in right_root.findall("mxCell"):
    if cell.get("id") in {"0", "1"}:
        continue

    new_cell = deepcopy(cell)
    for attribute in ("id", "parent", "source", "target"):
        value = new_cell.get(attribute)
        if value is not None:
            if attribute == "parent" and value in {"0", "1"}:
                new_cell.set(attribute, value)
            else:
                new_cell.set(attribute, f"r_{value}")

    geometry = new_cell.find("mxGeometry")
    if geometry is not None and geometry.get("x") is not None:
        geometry.set("x", str(float(geometry.get("x")) + RIGHT_OFFSET_X))

    for point in new_cell.findall(".//mxPoint"):
        if point.get("x") is not None:
            point.set("x", str(float(point.get("x")) + RIGHT_OFFSET_X))

    left_root.append(new_cell)

mxfile = ET.Element(
    "mxfile",
    {
        "host": "app.diagrams.net",
        "agent": "Mozilla/5.0",
        "version": "24.7.17",
        "type": "device",
        "compressed": "false",
    },
)
diagram = ET.SubElement(mxfile, "diagram", {"id": "combined-system-flow", "name": "Flowchart Sistem"})
diagram.append(left_model)
ET.ElementTree(mxfile).write(OUTPUT, encoding="UTF-8", xml_declaration=True)

# Structural validation: unique IDs and all referenced cells exist.
parsed = ET.parse(OUTPUT).getroot().find("diagram").find("mxGraphModel").find("root")
cells = parsed.findall("mxCell")
ids = [cell.get("id") for cell in cells]
assert len(ids) == len(set(ids))
known = set(ids)
for cell in cells:
    for attribute in ("parent", "source", "target"):
        value = cell.get(attribute)
        if value is not None:
            assert value in known, (cell.get("id"), attribute, value)

print(OUTPUT)
