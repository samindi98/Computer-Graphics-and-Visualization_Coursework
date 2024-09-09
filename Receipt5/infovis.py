import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

def preprocess_image(image):
    # Grayscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize=(10, 6))
    plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
    plt.title('Grayscale Image')
    plt.axis('off')
    plt.show()
    
    # Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    plt.figure(figsize=(10, 6))
    plt.imshow(blur, cmap='gray', vmin=0, vmax=255)
    plt.title('Gaussian Blur')
    plt.axis('off')
    plt.show()
    
    # Adaptive Thresholding
    binarized = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    plt.figure(figsize=(10, 6))
    plt.imshow(binarized, cmap='gray', vmin=0, vmax=255)
    plt.title('Adaptive Thresholding')
    plt.axis('off')
    plt.show()
    
    # Morphological Operation
    kernel = np.ones((1, 1), np.uint8)
    morph_img = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel)
    plt.figure(figsize=(10, 6))
    plt.imshow(morph_img, cmap='gray', vmin=0, vmax=255)
    plt.title('Morphological Operation')
    plt.axis('on')
    plt.show()
    
    return morph_img

def extract_text_from_section(section):
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(section, config=custom_config)
    return text

def parse_text(text):
    lines = text.split('\n')
    parsed_data = {}
    current_receipt = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "RESTAURANT" in line or "SUPERMARKET" in line or "COFFEE-SHOP" in line or "ELECTRO SHOP" in line or "BAKERY" in line or "PET SHOP" in line:
            current_receipt = line
            parsed_data[current_receipt] = {}
        elif current_receipt:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    price = float(parts[-1])
                    item_name = ' '.join(parts[:-1])
                    parsed_data[current_receipt][item_name] = price
                except ValueError:
                    continue
    return parsed_data

def plot_sales_summary(data):
    # Combine items and totals from all receipts
    combined_items = []
    combined_totals = []

    for items in data.values():
        combined_items.extend(items.keys())
        combined_totals.extend(items.values())

    # Plot the combined sales summary
    plt.figure(figsize=(14, 8))
    plt.barh(combined_items, combined_totals, color='skyblue')
    plt.xlabel('Total Sales ($)')
    plt.title('Combined Sales Summary for All Receipts')
    plt.show()

def process_image_sections(image_sections):
    with ThreadPoolExecutor(max_workers=len(image_sections)) as executor:
        text_results = list(executor.map(extract_text_from_section, image_sections))
    combined_text = "\n".join(text_results)
    return combined_text

def summarize_text(parsed_data):
    summary = []
    for receipt_name, items in parsed_data.items():
        total_sales = sum(items.values())
        summary.append(f"Receipt: {receipt_name}")
        summary.append(f"Total Sales: ${total_sales:.2f}")
        summary.append("Items Sold:")
        for item_name, price in items.items():
            summary.append(f"  - {item_name}: ${price:.2f}")
        summary.append("")  # Blank line for separation
    return "\n".join(summary)

def main():
    image_path = "Recepts.png"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    processed_image = preprocess_image(image)
    
    # Assuming 8 sections, adjust as needed
    height, width = processed_image.shape[:2]
    section_height = height // 8
    image_sections = [processed_image[i*section_height:(i+1)*section_height, :] for i in range(8)]
    
    # Process each section concurrently
    extracted_text = process_image_sections(image_sections)
    
    # Display the extracted text (optional)
    print("Extracted Text:\n", extracted_text)
    
    # Parse the extracted text into structured data
    parsed_data = parse_text(extracted_text)
    
    # Print summary of the extracted text
    summary = summarize_text(parsed_data)
    print("Summary:\n", summary)
    
    # Plot the combined sales summary for all receipts
    plot_sales_summary(parsed_data)

if __name__ == "__main__":
    main()
