from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r'"C:\Program Files\Tesseract-OCR\tesseract.exe"'

# Load the image
image_path = "Recept-I.png"  
image = Image.open(image_path)

# Preprocess the image
image = image.convert('L')  # Convert to grayscale
image = image.filter(ImageFilter.SHARPEN)  # Sharpen the image

# Enhance contrast
enhancer = ImageEnhance.Contrast(image)
image = enhancer.enhance(2)

# Apply OCR
custom_config = r'--oem 3 --psm 6'
extracted_text = pytesseract.image_to_string(image, config=custom_config)

# Display the image and the extracted text
plt.figure(figsize=(10, 5))

# Show original image
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.title("Original Image")

# Show extracted text
plt.subplot(1, 2, 2)
plt.text(0.5, 0.5, extracted_text, fontsize=12, ha='center', va='center', wrap=True)
plt.axis('off')
plt.title("Extracted Text")

# Display the plot
plt.show()

# Print the extracted text to console
print("Extracted Text:\n")
print(extracted_text)

