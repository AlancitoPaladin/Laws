# Laws

Texture Analysis Tool (Go)

Overview
This command-line tool performs basic texture analysis on an input image. It:
- Converts the image to grayscale.
- Applies Laws’ texture filters (L5L5 and E5E5) and reports local energy ranges.
- Builds a Gray Level Co-occurrence Matrix (GLCM) with 32 gray levels (dx=1, dy=0) and computes Haralick features: Contrast, Energy, Entropy, Homogeneity, Variance.
- Computes Local Binary Patterns (LBP) and prints the first 10 bins of the LBP histogram.

Requirements
- Go 1.18+ (earlier versions may work, but this was tested with Go 1.20+)

Project Structure
- main.go: Implementation of the texture analysis pipeline.
- images/billow1.jpg: Sample image you can use for testing.
- go.mod: Go module definition.

Build and Run
From the project root:

Run without building:
go run main.go <image_path>

Build a binary:
go build -o texture-analyzer
./texture-analyzer <image_path>

Examples
Using the provided sample image:
go run main.go images/billow1.jpg

Expected output (example snippet):
Processing image: billow1.jpg (format: .jpg)
Successfully loaded image format: jpeg
Image size: <width>x<height>

Applying Laws' texture filters...
L5L5 energy range: <min> - <max>
E5E5 energy range: <min> - <max>

Calculating GLCM features...
Contrast: <value>
Energy: <value>
Entropy: <value>
Homogeneity: <value>
Variance: <value>

Calculating Local Binary Pattern...
LBP histogram (first 10 bins): [b0 b1 b2 b3 b4 b5 b6 b7 b8 b9]

Supported Image Formats
- JPEG, PNG, GIF (as enabled via blank imports in main.go)

Troubleshooting
- If you see "Usage: go run main.go <image_path>", ensure you passed a valid image path.
- For large images, processing may take longer; consider downscaling if necessary.
- If format decode fails, make sure the file extension and content are a supported image type.

License
- Add your preferred license here (e.g., MIT). If omitted, this code is provided as-is without warranty.

Credits
- Texture concepts: Laws’ filters, GLCM/Haralick features, and LBP are standard techniques in image processing.

