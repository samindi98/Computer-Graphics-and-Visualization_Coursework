import cv2
import pytesseract
from matplotlib import pyplot as plt
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\sahas\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Load the image
image_path = "Recept-III.png"  # Replace with your image path
image = cv2.imread(image_path)

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding (binarization)
_, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Perform OCR to extract text
extracted_text = pytesseract.image_to_string(binary_image)

# Create a figure to display image and text side by side
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Display the original image on the left side
axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0].set_title("Original Receipt Image")
axs[0].axis('off')  # Hide axes for better display

# Display the extracted text on the right side
axs[1].text(0.1, 0.5, extracted_text, fontsize=12, verticalalignment='center', wrap=True)
axs[1].set_title("Extracted Text")
axs[1].axis('off')  # Hide axes for better display

# Adjust layout and show the result
plt.tight_layout()
plt.show()


# print the extracted text to the console
print("Extracted Text from the Image:")
print(extracted_text)

