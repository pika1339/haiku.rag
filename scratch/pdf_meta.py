"""Print the first page text of each PDF given on the command line.

Used to read title, journal and year off the papers' cover pages.
"""

import sys
from pathlib import Path

import pypdfium2 as pdfium


def main(paths: list[str]) -> int:
    for arg in paths:
        path = Path(arg)
        pdf = pdfium.PdfDocument(path)
        print(f"{'=' * 70}\nFILE: {path.name}\nPAGES: {len(pdf)}\n{'-' * 70}")
        text = pdf[0].get_textpage().get_text_range()
        print(text.strip()[:1600])
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
