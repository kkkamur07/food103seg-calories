from PIL import Image, ImageDraw, ImageFont
import os

os.makedirs("static", exist_ok=True)

favicon = Image.new("RGB", (32, 32), color="white")
draw = ImageDraw.Draw(favicon)

try:
    font = ImageFont.truetype("arial.ttf", 24)
except IOError:
    font = ImageFont.load_default()

text = "FS"
text_color = (200, 0, 0)

bbox = draw.textbbox((0, 0), text, font=font)
text_width = bbox[2] - bbox[0]
text_height = bbox[3] - bbox[1]
x = (32 - text_width) // 2
y = (32 - text_height) // 2

draw.text((x, y), text, font=font, fill=text_color)

favicon.save("static/favicon.ico", format="ICO")
