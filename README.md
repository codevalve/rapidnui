
# Rapidnui: Rongorongo Symbol Analysis

## Project Background
The *Rongorongo* script from Easter Island has long intrigued researchers due to its complex symbols and undeciphered status. Despite extensive study, little is understood about its function or meaning. This project, **Rapidnui**, was created to apply modern machine learning and image processing techniques to analyze the script's symbols. Using methods such as contour detection and clustering, we aim to identify unique glyphs, study their relationships, and explore possible symbolic or proto-linguistic elements in *Rongorongo*.

## Project Goals
- Identify unique glyphs within the *Rongorongo* text.
- Analyze symbol frequency and clustering for patterns or repeated elements.
- Explore hypotheses about the structure of the script, including potential proto-writing or mnemonic functions.

## Features
- **Image Preprocessing**: Grayscale conversion, thresholding, and contour detection to isolate symbols.
- **Clustering Analysis**: DBSCAN clustering to detect unique glyphs and visualize relationships.
- **t-SNE Visualization**: Dimensionality reduction to illustrate the distribution of detected symbols.

## Getting Started
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/codevalve/rapidnui.git
   ```
2. **Install Dependencies**:
   Make sure you have Python and the required libraries installed:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Analysis**:
   Process a Rongorongo image and analyze its symbols:
   ```bash
   python analyze_rongorongo.py
   ```

## Results
Initial analyses reveal clusters of 13 unique symbols, indicating a limited glyph set possibly used for proto-writing or mnemonic purposes. The t-SNE and contour visualizations reveal both consistent glyphs and variations, supporting theories of symbolic encoding in *Rongorongo*.

## Contributions
Contributions to improve detection accuracy, feature extraction, and hypothesis testing are welcome. Feel free to submit pull requests or issues!

## License
This project is licensed under the MIT License.

## Acknowledgments
Special thanks to Thomas Barthel’s work on *Rongorongo* and the open-source Python community.

