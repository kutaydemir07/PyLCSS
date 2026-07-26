# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Create Windows installer artwork from the canonical PyLCSS application logo."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


NAVY_TOP = (7, 25, 45, 255)
NAVY_BOTTOM = (3, 12, 24, 255)
BLUE = (60, 178, 255, 255)
GREEN = (127, 239, 64, 255)
WHITE = (245, 249, 255, 255)


def _font(name: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("C:/Windows/Fonts") / name
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _trimmed_logo(source: Path) -> Image.Image:
    logo = Image.open(source).convert("RGBA")
    alpha_box = logo.getchannel("A").getbbox()
    return logo.crop(alpha_box) if alpha_box else logo


def _contain(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    return copy


def _create_wizard_image(logo: Image.Image, destination: Path) -> None:
    width, height = 480, 918
    canvas = Image.new("RGBA", (width, height), NAVY_TOP)
    pixels = canvas.load()
    for y in range(height):
        fraction = y / max(height - 1, 1)
        color = tuple(
            round(NAVY_TOP[channel] * (1 - fraction) + NAVY_BOTTOM[channel] * fraction)
            for channel in range(3)
        ) + (255,)
        for x in range(width):
            pixels[x, y] = color

    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.ellipse((-210, 500, 430, 1140), fill=(16, 91, 158, 38))
    draw.ellipse((120, -180, 680, 380), fill=(104, 239, 66, 25))
    draw.rounded_rectangle(
        (44, 62, width - 44, height - 62),
        radius=28,
        outline=(89, 180, 255, 65),
        width=2,
    )

    fitted_logo = _contain(logo, (350, 350))
    logo_x = (width - fitted_logo.width) // 2
    canvas.alpha_composite(fitted_logo, (logo_x, 155))

    title_font = _font("segoeuib.ttf", 62)
    subtitle_font = _font("seguisb.ttf", 18)
    detail_font = _font("segoeui.ttf", 18)
    title = "PyLCSS"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    title_x = (width - (title_box[2] - title_box[0])) // 2
    draw.text((title_x, 560), title, font=title_font, fill=WHITE)

    subtitle = "ENGINEERING DESIGN PLATFORM"
    subtitle_box = draw.textbbox((0, 0), subtitle, font=subtitle_font)
    subtitle_x = (width - (subtitle_box[2] - subtitle_box[0])) // 2
    draw.text((subtitle_x, 647), subtitle, font=subtitle_font, fill=BLUE)
    draw.rounded_rectangle((116, 696, 364, 701), radius=2, fill=GREEN)

    detail = "Model  •  Simulate  •  Optimize"
    detail_box = draw.textbbox((0, 0), detail, font=detail_font)
    detail_x = (width - (detail_box[2] - detail_box[0])) // 2
    draw.text((detail_x, 738), detail, font=detail_font, fill=(190, 210, 230, 255))
    canvas.convert("RGB").save(destination, format="PNG", optimize=True)


def _create_small_image(logo: Image.Image, destination: Path) -> None:
    canvas = Image.new("RGBA", (256, 256), (255, 255, 255, 0))
    fitted_logo = _contain(logo, (224, 224))
    canvas.alpha_composite(
        fitted_logo,
        ((canvas.width - fitted_logo.width) // 2, (canvas.height - fitted_logo.height) // 2),
    )
    canvas.save(destination, format="PNG", optimize=True)


def create_assets(source: Path, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    logo = _trimmed_logo(source)
    logo.save(
        output_directory / "PyLCSS.ico",
        format="ICO",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    _create_small_image(logo, output_directory / "PyLCSS-wizard-small.png")
    _create_wizard_image(logo, output_directory / "PyLCSS-wizard.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    create_assets(arguments.source.resolve(), arguments.output_directory.resolve())


if __name__ == "__main__":
    main()
