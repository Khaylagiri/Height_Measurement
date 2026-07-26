from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


output_dir = Path(r"D:\TA Skripsi\aplikasi\HeightMeasurement\outputs\gambar_rumus")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "rumus_estimasi_titik_kepala.png"

canvas = Image.new("RGB", (2800, 820), "white")
draw = ImageDraw.Draw(canvas)

main_font = ImageFont.truetype(r"C:\Windows\Fonts\timesi.ttf", 108)
small_font = ImageFont.truetype(r"C:\Windows\Fonts\timesi.ttf", 68)


def token_width(text: str, font: ImageFont.FreeTypeFont) -> float:
    return draw.textlength(text, font=font)


def draw_math_line(tokens, y: int) -> None:
    total_width = sum(token_width(text, small_font if level != "main" else main_font) for text, level in tokens)
    x = (canvas.width - total_width) / 2
    for text, level in tokens:
        font = main_font if level == "main" else small_font
        if level == "sub":
            token_y = y + 62
        elif level == "super":
            token_y = y - 24
        else:
            token_y = y
        draw.text((x, token_y), text, fill="black", font=font)
        x += token_width(text, font)


draw_math_line(
    [
        ("d(A,B) = sqrt[(x", "main"),
        ("B", "sub"),
        (" - x", "main"),
        ("A", "sub"),
        (")", "main"),
        ("2", "super"),
        (" + (y", "main"),
        ("B", "sub"),
        (" - y", "main"),
        ("A", "sub"),
        (")", "main"),
        ("2", "super"),
        ("]", "main"),
    ],
    75,
)

draw_math_line(
    [("k = 0,35 d(mata,bahu) + 0,10 d(telinga,mata)", "main")],
    325,
)

draw_math_line(
    [
        ("P", "main"),
        ("kepala", "sub"),
        (" = P", "main"),
        ("wajah", "sub"),
        (" + k u", "main"),
    ],
    570,
)

bbox = canvas.getbbox()
canvas = canvas.crop(bbox)
canvas.save(output_path, format="PNG", dpi=(300, 300), optimize=True)

print(output_path)
