import cv2
import pytesseract
from matplotlib import pyplot as plt
import numpy as np  # Import numpy for array manipulation


pytesseract.pytesseract.tesseract_cmd = r'C:\Users\sahas\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'


# Load the image
image_path = "Recept-IV.png"  # Replace with the actual path to your image
image = cv2.imread(image_path)

# Step 1: Resize the image to enhance text clarity
image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# Step 2: Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply GaussianBlur to reduce noise
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Step 4: Apply thresholding (binarization)
_, binary_image = cv2.threshold(blurred_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Step 5: Optional - Apply a sharpening filter
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened_image = cv2.filter2D(binary_image, -1, kernel)

# Step 6: Perform OCR to extract text
extracted_text = pytesseract.image_to_string(sharpened_image)

# Step 7: Display the original and processed images side by side
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Display the original image
axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0].set_title("Original Receipt Image")
axs[0].axis('off')  # Hide axes for better display

# Display the extracted text on the right side
axs[1].text(0.1, 0.5, extracted_text, fontsize=12, verticalalignment='center', wrap=True)
axs[1].set_title("Extracted Text")
axs[1].axis('off')  # Hide axes for better display

# Show the result
plt.tight_layout()
plt.show()

# print the extracted text to the console
print("Extracted Text from the Image:")
print(extracted_text)
