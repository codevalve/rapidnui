import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Load the image and convert to grayscale
image = cv2.imread('rongo-santiago.jpg', cv2.IMREAD_GRAYSCALE)

# Threshold the image to get binary representation (black & white)
_, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

# Find contours in the thresholded image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# List to store feature vectors of symbols
features = []

# Loop through each contour and extract features
for contour in contours:
    # Compute the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Crop the symbol region
    symbol = thresh[y:y + h, x:x + w]
    
    # Resize symbol to a standard size (e.g., 32x32) to normalize features
    symbol = cv2.resize(symbol, (32, 32))
    
    # Flatten the image to create a feature vector
    feature_vector = symbol.flatten()
    features.append(feature_vector)

# Convert feature list to a numpy array
features = np.array(features)

# Use DBSCAN to cluster similar symbols
# Adjust the eps (distance threshold) and min_samples as needed
db = DBSCAN(eps=0.5, min_samples=3).fit(features)

# Count the number of unique clusters (symbols)
# Cluster labels of -1 represent noise, so exclude them from the count
n_unique_symbols = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

# Output the number of unique symbols found
print(f"Number of unique symbols detected: {n_unique_symbols}")

# Optional: visualize the bounding boxes of detected symbols
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with bounding boxes
plt.imshow(image, cmap='gray')
plt.title('Detected Symbols')
plt.show()
