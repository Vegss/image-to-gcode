from __future__ import annotations
import subprocess
from sys import argv
import cv2
from PIL import Image
import numpy as np

# base = 200mm X 200mm
SIDE_LENGTH = 200
DISTANCE_TO_LIFT_PEN = 10


def image_to_outlines(filename: str):
    # Ladataan kuva
    image = cv2.imread(f"./images/{filename}", cv2.IMREAD_GRAYSCALE)

    alpha = 1.2  # Kontrasti (1.0 = ei muutosta)
    beta = 40  # Kirkkaus (+50 kirkastaa)

    bright_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Pehmennys (kohinan vähentäminen)
    blurred = cv2.GaussianBlur(bright_image, (5, 5), 0)

    # Reunojen havaitseminen
    edges = cv2.Canny(blurred, threshold1=20, threshold2=100)

    img = cv2.cvtColor(edges, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(img)

    # Tallennetaan tulos
    cv2.imwrite("./processing/edges.jpg", img)


def outlines_to_gcode(
    src: str = "./processing/edges.jpg", output: str = "./processing/output.nc"
):
    subprocess.run(
        ["python3", "image_to_gcode.py", "-i", src, "-o", output, "-e", "white"]
    )


class Gcode_line:
    def __init__(self, penDown: bool, x: float, y: float):
        self.penDown = penDown
        self.x = x
        self.y = y

    def to_string(self) -> str:
        return f"G{int(self.penDown)} X{self.x} Y{self.y}\n"

    @staticmethod
    def from_string(line: str) -> Gcode_line:
        command, x, y = line.split(" ")
        command = True if int(command[1:]) == 1 else False
        x, y = int(x[1:]), int(y[1:])
        return Gcode_line(bool(command), x, y)


def lines_to_objects(lines: list[str]) -> list[Gcode_line]:
    return [Gcode_line.from_string(line) for line in lines]


def finish_gcode(dest: str, scaling_factor: float | None = None):
    result = [
        "G21 ; Aseta mittayksiköksi millimetrit\n",
        "G17 ; Valitse XY-taso\n",
        "G90 ; absoluuttinen koordinaatisto\n",
        "F10000 ; Aseta nopeudeksi 10000 mm/min\n",
        "M3 S2000 ; kynä ylös alussa \n\n",
    ]

    with open("./processing/output.nc", "r") as file, open(dest, "w") as res:
        lines = file.readlines()
        objects = lines_to_objects(lines)

        x_max, y_max = get_max_values(objects)
        x_min, _ = get_min_values(objects)
        max_val = max(x_max, y_max)

        if not scaling_factor:
            scaling_factor = SIDE_LENGTH / max_val

        updated_lines = []
        isPenDown = False
        for index, line in enumerate(objects):
            # Kynä
            # Ylös M3 S2000
            # Alas M5

            if line.penDown and not isPenDown:
                updated_lines.append("M5\n")
                isPenDown = True

            distance = 0
            if index < len(objects) - 1:
                nextLine = objects[index + 1]
                distance = np.linalg.norm(
                    np.array([line.x, line.y]) - np.array([nextLine.x, nextLine.y])
                )
                line.y = (line.y + y_max) * scaling_factor
                line.x = (line.x - x_min) * scaling_factor
                updated_lines.append(line.to_string())
                if line.penDown and not nextLine.penDown:
                    if distance > DISTANCE_TO_LIFT_PEN:
                        updated_lines.append("M3 S2000\n")
                        isPenDown = False
            else:
                line.y = (line.y + y_max) * scaling_factor
                line.x = (line.x - x_min) * scaling_factor
                updated_lines.append(line.to_string())

        result += updated_lines
        res.writelines(result)


def get_max_values(lines: list[Gcode_line]) -> tuple[float, float]:
    return max([abs(line.x) for line in lines]), max([abs(line.y) for line in lines])


def get_min_values(lines: list[Gcode_line]) -> tuple[float, float]:
    return min([abs(line.x) for line in lines]), min([abs(line.y) for line in lines])


def get_min_y(lines: list[str]):
    y_values = []
    for line in lines:
        _, _, y = line.split(" ")
        y_values.append(int(y[1:]))
    return min(y_values)


def main():
    if len(argv) < 1:
        print("Usage: python main.py <filename>")
        return
    filename = argv[1]
    filename_omit_extension = filename.split(".")[0]
    image_to_outlines(filename)
    outlines_to_gcode()
    finish_gcode(f"./output/{filename_omit_extension}.nc")


if __name__ == "__main__":
    main()
