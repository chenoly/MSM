import os
import PyPDF2
import shutil
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader


def add_filename_to_image(image_path: str, output_path: str, font_size=72):
    """
    Adds the filename to the top-right corner of a binary grayscale image.

    :param image_path: Path to input image file.
    :param output_path: Path to save the modified image.
    :param font_size: Font size for the text (based on DPI and desired width).
    """
    img = Image.open(image_path).convert("L")  # 确保是灰度图
    width, height = img.size

    draw = ImageDraw.Draw(img)

    filename = os.path.splitext(os.path.basename(image_path))[0]
    filename = f"d_netattack_{filename}"
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    text_size = draw.textsize(filename, font=font)

    margin = 10
    text_position = (width - text_size[0] - margin, margin)

    draw.text(text_position, filename, fill=0, font=font)  # 白色文字

    img.save(output_path)


def convert_png_to_pdf_lossless(input_image, output_pdf):
    """
    Converts a single PNG image to PDF with high quality using ReportLab.

    :param input_image: Path to the input PNG image.
    :param output_pdf: Path to the output PDF file.
    """
    try:
        # Create a canvas object (PDF document)
        c = canvas.Canvas(output_pdf, pagesize=A4)

        # Load the black and white image using ImageReader
        img = ImageReader(input_image)

        # Get the dimensions of the image
        img_width, img_height = img.getSize()

        # Calculate scaling to fit the image onto A4 page
        aspect_ratio = img_width / img_height
        page_width, page_height = A4
        if aspect_ratio >= 1:
            scaled_width = page_width
            scaled_height = scaled_width / aspect_ratio
        else:
            scaled_height = page_height
            scaled_width = scaled_height * aspect_ratio

        # Center the image on the page
        x_pos = (page_width - scaled_width) / 2
        y_pos = (page_height - scaled_height) / 2

        # Draw the image onto the PDF
        c.drawImage(img, x_pos, y_pos, width=scaled_width, height=scaled_height,
                    preserveAspectRatio=True, mask='auto')

        # Save the PDF document
        c.save()
    finally:
        # Explicitly delete the canvas objects
        del c


def merge_pdfs(pdf_list, output_path):
    """
    Merges multiple PDF files into one.

    :param pdf_list: List of paths to PDF files.
    :param output_path: Path to the merged PDF file.
    """
    merger = PyPDF2.PdfMerger()

    for pdf in pdf_list:
        merger.append(pdf)

    merger.write(output_path)
    merger.close()


def process_images_in_folder(input_folder: str, temp_pdf_folder: str, final_output_pdf: str):
    """
    Process all PNG images in the input folder:
    1. Add filename watermark
    2. Convert each to PDF
    3. Merge all PDFs into one

    :param input_folder: Folder containing original PNG images
    :param temp_pdf_folder: Temporary folder for individual PDFs
    :param final_output_pdf: Final merged PDF path
    """
    if os.path.exists(temp_pdf_folder):
        shutil.rmtree(temp_pdf_folder)
    os.makedirs(temp_pdf_folder, exist_ok=True)

    tmp_img_folder = os.path.join(temp_pdf_folder, "tmp_images")
    os.makedirs(tmp_img_folder, exist_ok=True)

    pdf_files = []

    for i, filename in enumerate(os.listdir(input_folder)):
        if filename.lower().endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(tmp_img_folder, filename)
            output_pdf_path = os.path.join(temp_pdf_folder, f"{i}.pdf")

            add_filename_to_image(input_path, output_image_path)

            convert_png_to_pdf_lossless(output_image_path, output_pdf_path)
            pdf_files.append(output_pdf_path)

    merge_pdfs(pdf_files, final_output_pdf)

    shutil.rmtree(tmp_img_folder)


if __name__ == "__main__":
    input_folder = "digital_outer"
    temp_pdf_folder = "temp_pdfs"
    final_output_pdf = "D_netattack.pdf"

    process_images_in_folder(input_folder, temp_pdf_folder, final_output_pdf)