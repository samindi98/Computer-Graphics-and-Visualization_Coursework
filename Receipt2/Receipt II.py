import cv2
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
def extract_receipt_text(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to preprocess the image
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Perform OCR on the processed image
    text = pytesseract.image_to_string(thresh)

    return image, text.strip()

# Call the function with the path to the image
image_path = "Recept-II.png"  # Adjust path based on the location of the image
image, extracted_text = extract_receipt_text(image_path)

# Create a plot with two panels: one for the image, and one for the text
plt.figure(figsize=(15, 6))

# Plot the original image
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Plot the extracted text
plt.subplot(1, 2, 2)
plt.title('Extracted Text')
plt.text(0.1, 0.5, extracted_text, fontsize=12, wrap=True, verticalalignment='center')
plt.axis('off')

# Show the plot
plt.tight_layout()
plt.show()

