from pathlib import Path

from pypdf import PdfReader


def extract_pdf_text(pdf_path: Path, output_path: Path) -> None:
    reader = PdfReader(str(pdf_path))

    with output_path.open("w", encoding="utf-8") as out:
        out.write(f"Source PDF: {pdf_path.name}\n")
        out.write(f"Total pages: {len(reader.pages)}\n\n")

        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            out.write(f"===== Page {i} =====\n")
            out.write(text.strip())
            out.write("\n\n")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    pdf_path = base_dir / "Binder1.pdf"
    output_path = base_dir / "Binder1_extracted.txt"

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    extract_pdf_text(pdf_path, output_path)
    print(f"Extracted text saved to: {output_path}")


if __name__ == "__main__":
    main()
