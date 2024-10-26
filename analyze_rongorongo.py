import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import os

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
    x, y, w, h = cv2.boundingRect(contour)
    symbol = thresh[y:y + h, x:x + w]
    symbol = cv2.resize(symbol, (32, 32))
    feature_vector = symbol.flatten()
    features.append(feature_vector)

# Convert feature list to a numpy array
features = np.array(features)

# Use DBSCAN to cluster similar symbols
db = DBSCAN(eps=0.5, min_samples=3).fit(features)
n_unique_symbols = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
print(f"Number of unique symbols detected: {n_unique_symbols}")

# Create a directory to store extracted symbols
output_dir = 'extracted_symbols'
os.makedirs(output_dir, exist_ok=True)

# Loop through contours and save each detected symbol as an image
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    symbol = thresh[y:y + h, x:x + w]
    symbol_resized = cv2.resize(symbol, (32, 32))
    cv2.imwrite(os.path.join(output_dir, f'symbol_{i}.png'), symbol_resized)

# Use t-SNE to reduce feature dimensions to 2D for visualization
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features)

# Plot the t-SNE results, coloring by the DBSCAN cluster label
plt.figure(figsize=(8, 6))
for label in set(db.labels_):
    if label == -1:
        color = 'gray'
    else:
        color = plt.cm.jet(float(label) / max(db.labels_))
    plt.scatter(features_2d[db.labels_ == label, 0],
                features_2d[db.labels_ == label, 1],
                color=color,
                label=f'Symbol {label}')

plt.title('t-SNE Visualization of Detected Symbols')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(loc='best', bbox_to_anchor=(1.05, 1))
plt.show()
