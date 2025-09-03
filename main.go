package main

import (
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"
)

//  Laws' texture masks

// 3 x 3

var (
	L3 = []float64{1, 2, 1}  // Level
	E3 = []float64{-1, 0, 1} // Edge
	S3 = []float64{1, -2, 1} // Spot

)

// 5 x 5
var (
	L5 = []float64{1, 4, 6, 4, 1}   // Level (nivel)
	E5 = []float64{-1, -2, 0, 2, 1} // Edge (borde)
	S5 = []float64{-1, 0, 2, 0, -1} // Spot (punto)
	W5 = []float64{-1, 2, 0, -2, 1} // Wave (rizado)
	R5 = []float64{1, -4, 6, -4, 1} // Ripple (onda)
)

// 7 x 7

var (
	L7 = []float64{1, 1, 1, 1, 1, 1, 1}
	E7 = []float64{-1, -1, -1, 0, 1, 1, 1}
	S7 = []float64{-1, -1, 2, 3, 2, -1, -1}
	W7 = []float64{-1, 2, -1, -1, -1, 2, -1}
	R7 = []float64{1, -2, 1, 0, -1, 2, -1}
)

// TextureFeatures represents computed texture features
type TextureFeatures struct {
	Contrast    float64
	Energy      float64
	Entropy     float64
	Homogeneity float64
	Mean        float64
	Variance    float64
}

// Convert image to grayscale matrix
func imageToGray(img image.Image) [][]uint8 {
	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y
	gray := make([][]uint8, height)

	for y := 0; y < height; y++ {
		gray[y] = make([]uint8, width)
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			// Convert to grayscale using luminance formula
			grayValue := uint8((0.299*float64(r) + 0.587*float64(g) + 0.114*float64(b)) / 256)
			gray[y][x] = grayValue
		}
	}
	return gray
}

// --- NEW: grayscale as float64 in [0,1] ---
func imageToGrayFloat(img image.Image) [][]float64 {
	b := img.Bounds()
	w, h := b.Max.X, b.Max.Y
	out := make([][]float64, h)
	for y := 0; y < h; y++ {
		out[y] = make([]float64, w)
		for x := 0; x < w; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			// r,g,b in [0, 65535]
			val := (0.299*float64(r) + 0.587*float64(g) + 0.114*float64(b)) / 65535.0
			out[y][x] = val
		}
	}
	return out
}

// Apply 1D convolution (horizontal or vertical)
func convolve1D(data [][]uint8, kernel []float64, horizontal bool) [][]float64 {
	height, width := len(data), len(data[0])
	result := make([][]float64, height)
	kernelSize := len(kernel)
	offset := kernelSize / 2

	for y := 0; y < height; y++ {
		result[y] = make([]float64, width)
		for x := 0; x < width; x++ {
			sum := 0.0

			if horizontal {
				// Horizontal convolution
				for k := 0; k < kernelSize; k++ {
					px := x + k - offset
					if px >= 0 && px < width {
						sum += float64(data[y][px]) * kernel[k]
					}
				}
			} else {
				// Vertical convolution
				for k := 0; k < kernelSize; k++ {
					py := y + k - offset
					if py >= 0 && py < height {
						sum += float64(data[py][x]) * kernel[k]
					}
				}
			}
			result[y][x] = sum
		}
	}
	return result
}

// --- NEW: 1D convolution float64, no casting ---
func convolve1DFloat(data [][]float64, kernel []float64, horizontal bool) [][]float64 {
	h, w := len(data), len(data[0])
	res := make([][]float64, h)
	k := len(kernel)
	off := k / 2
	for y := 0; y < h; y++ {
		res[y] = make([]float64, w)
		for x := 0; x < w; x++ {
			acc := 0.0
			if horizontal {
				for i := 0; i < k; i++ {
					px := x + i - off
					if px >= 0 && px < w {
						acc += data[y][px] * kernel[i]
					}
				}
			} else {
				for i := 0; i < k; i++ {
					py := y + i - off
					if py >= 0 && py < h {
						acc += data[py][x] * kernel[i]
					}
				}
			}
			res[y][x] = acc
		}
	}
	return res
}

// Apply Laws' texture filters
func applyLawsFilters(gray [][]uint8, mask1, mask2 []float64) [][]float64 {
	// Apply the first mask horizontally
	temp := convolve1D(gray, mask1, true)

	// Convert temp to uint8 for the second convolution
	tempUint8 := make([][]uint8, len(temp))
	for y := range temp {
		tempUint8[y] = make([]uint8, len(temp[y]))
		for x := range temp[y] {
			val := math.Abs(temp[y][x])
			if val > 255 {
				val = 255
			}
			tempUint8[y][x] = uint8(val)
		}
	}

	// Aplica la segunda máscara verticalmente
	return convolve1D(tempUint8, mask2, false)
}

// --- NEW: apply Laws filters fully in float (no quantization) ---
func applyLawsFiltersFloat(gray [][]float64, maskH, maskV []float64) [][]float64 {
	tmp := convolve1DFloat(gray, maskH, true)
	out := convolve1DFloat(tmp, maskV, false)
	return out
}

// Calcular la energía de textura
func calculateTextureEnergy(filtered [][]float64, windowSize int) [][]float64 {
	height, width := len(filtered), len(filtered[0])
	energy := make([][]float64, height)
	offset := windowSize / 2

	for y := 0; y < height; y++ {
		energy[y] = make([]float64, width)
		for x := 0; x < width; x++ {
			sum := 0.0
			count := 0

			// Calculate energy in a local window
			for wy := -offset; wy <= offset; wy++ {
				for wx := -offset; wx <= offset; wx++ {
					ny, nx := y+wy, x+wx
					if ny >= 0 && ny < height && nx >= 0 && nx < width {
						sum += filtered[ny][nx] * filtered[ny][nx]
						count++
					}
				}
			}
			if count > 0 {
				energy[y][x] = math.Sqrt(sum / float64(count))
			}
		}
	}
	return energy
}

// --- NEW: local mean removal (illumination normalization) ---
func removeLocalMean(gray [][]float64, window int) [][]float64 {
	h, w := len(gray), len(gray[0])
	off := window / 2
	out := make([][]float64, h)
	for y := 0; y < h; y++ {
		out[y] = make([]float64, w)
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			sum := 0.0
			n := 0
			for vy := -off; vy <= off; vy++ {
				for vx := -off; vx <= off; vx++ {
					yy, xx := y+vy, x+vx
					if yy >= 0 && yy < h && xx >= 0 && xx < w {
						sum += gray[yy][xx]
						n++
					}
				}
			}
			mean := sum / float64(n)
			out[y][x] = gray[y][x] - mean
		}
	}
	return out
}

// --- NEW: build Laws filter bank definitions ---
type lawsFilter struct {
	Name string
	H    []float64
	V    []float64
}

func lawsFilterBank() []lawsFilter {
	vecs := []struct {
		n string
		v []float64
	}{
		{"L5", L5}, {"E5", E5}, {"S5", S5}, {"W5", W5}, {"R5", R5},
	}
	var bank []lawsFilter
	for _, a := range vecs {
		for _, b := range vecs {
			bank = append(bank, lawsFilter{
				Name: a.n + b.n, // e.g., "L5E5" ≡ horizontal L5, vertical E5
				H:    a.v,
				V:    b.v,
			})
		}
	}
	return bank
}

// --- NEW: compute mean energy per filter (feature vector) ---
func lawsMeanEnergies(gray [][]float64, window int) []struct {
	Name  string
	MeanE float64
} {
	pre := removeLocalMean(gray, 15) // local mean removal (typical in Laws)
	bank := lawsFilterBank()
	features := make([]struct {
		Name  string
		MeanE float64
	}, 0, len(bank))

	for _, f := range bank {
		resp := applyLawsFiltersFloat(pre, f.H, f.V)
		emap := calculateTextureEnergy(resp, window)
		// Promedio de energia por filtro
		sum := 0.0
		n := 0
		for y := range emap {
			for x := range emap[y] {
				sum += emap[y][x]
				n++
			}
		}
		features = append(features, struct {
			Name  string
			MeanE float64
		}{Name: f.Name, MeanE: sum / float64(n)})
	}
	return features
}

// Calculate Gray Level Co-occurrence Matrix (GLCM)
func calculateGLCM(gray [][]uint8, dx, dy, levels int) [][]float64 {
	glcm := make([][]float64, levels)
	for i := range glcm {
		glcm[i] = make([]float64, levels)
	}

	height, width := len(gray), len(gray[0])

	// Build co-occurrence matrix
	for y := 0; y < height-dy; y++ {
		for x := 0; x < width-dx; x++ {
			i := int(gray[y][x]) * levels / 256
			j := int(gray[y+dy][x+dx]) * levels / 256
			if i < levels && j < levels {
				glcm[i][j]++
			}
		}
	}

	// Normalize
	total := 0.0
	for i := 0; i < levels; i++ {
		for j := 0; j < levels; j++ {
			total += glcm[i][j]
		}
	}

	if total > 0 {
		for i := 0; i < levels; i++ {
			for j := 0; j < levels; j++ {
				glcm[i][j] /= total
			}
		}
	}

	return glcm
}

// Calculate Haralick texture features from GLCM
func calculateHaralickFeatures(glcm [][]float64) TextureFeatures {
	levels := len(glcm)
	var features TextureFeatures

	// Calculate mean
	for i := 0; i < levels; i++ {
		for j := 0; j < levels; j++ {
			features.Mean += glcm[i][j] * float64(i)
		}
	}

	// Calculate other features
	for i := 0; i < levels; i++ {
		for j := 0; j < levels; j++ {
			p := glcm[i][j]
			if p > 0 {
				// Contrast
				features.Contrast += p * math.Pow(float64(i-j), 2)

				// Energy (Angular Second Moment)
				features.Energy += p * p

				// Entropy
				features.Entropy -= p * math.Log2(p)

				// Homogeneity (Inverse Difference Moment)
				features.Homogeneity += p / (1.0 + math.Abs(float64(i-j)))

				// Variance
				features.Variance += math.Pow(float64(i)-features.Mean, 2) * p
			}
		}
	}

	return features
}

// Implementación de patrones locales binarios
func calculateLBP(gray [][]uint8) [][]uint8 {
	height, width := len(gray), len(gray[0])
	lbp := make([][]uint8, height)

	// 8-neighborhood offsets
	dx := []int{-1, -1, -1, 0, 0, 1, 1, 1}
	dy := []int{-1, 0, 1, -1, 1, -1, 0, 1}

	for y := 1; y < height-1; y++ {
		lbp[y] = make([]uint8, width)
		for x := 1; x < width-1; x++ {
			center := gray[y][x]
			pattern := uint8(0)

			for i := 0; i < 8; i++ {
				neighbor := gray[y+dy[i]][x+dx[i]]
				if neighbor >= center {
					pattern |= 1 << uint(i)
				}
			}
			lbp[y][x] = pattern
		}
	}
	return lbp
}

// Función principal de análisis de texturas
func analyzeTexture(imagePath string) error {
	// Valida la existencia de la imagen
	if _, err := os.Stat(imagePath); os.IsNotExist(err) {
		return fmt.Errorf("image file does not exist: %s", imagePath)
	}

	// Cargar imagen
	file, err := os.Open(imagePath)
	if err != nil {
		return fmt.Errorf("failed to open image: %v", err)
	}
	defer file.Close()

	// Check file extension
	ext := strings.ToLower(filepath.Ext(imagePath))
	fmt.Printf("Processing image: %s (format: %s)\n", filepath.Base(imagePath), ext)

	img, format, err := image.Decode(file)
	if err != nil {
		return fmt.Errorf("failed to decode image (detected format: %s): %v", format, err)
	}

	fmt.Printf("Successfully loaded image format: %s\n", format)

	// Convert to grayscale
	gray := imageToGray(img)
	fmt.Printf("Image size: %dx%d\n", len(gray[0]), len(gray))

	// Apply Laws' texture filters
	fmt.Println("\nAplicando Laws' texture filtros...")

	// L5L5 - Level detection
	l5l5 := applyLawsFilters(gray, L5, L5)
	l5l5Energy := calculateTextureEnergy(l5l5, 5)
	fmt.Printf("L5L5 rango de energia: %.2f - %.2f\n",
		findMin(l5l5Energy), findMax(l5l5Energy))

	// E5E5 - Edge detection
	e5e5 := applyLawsFilters(gray, E5, E5)
	e5e5Energy := calculateTextureEnergy(e5e5, 5)
	fmt.Printf("E5E5 rango de energia: %.2f - %.2f\n",
		findMin(e5e5Energy), findMax(e5e5Energy))

	// --- NEW: Full Laws' filter bank features (mean energy) ---
	fmt.Println("\nLaws' filter banco (promedio de energia por filtro):")
	grayF := imageToGrayFloat(img)
	lawsFeatures := lawsMeanEnergies(grayF, 15)
	for _, f := range lawsFeatures {
		fmt.Printf("%-4s: %.6f\n", f.Name, f.MeanE)
	}

	// Calculate GLCM features
	fmt.Println("\nCalculando caracteristicas GLCM...")
	glcm := calculateGLCM(gray, 1, 0, 32) // horizontal, distance=1, 32 gray levels
	features := calculateHaralickFeatures(glcm)

	fmt.Printf("Contraste: %.4f\n", features.Contrast)
	fmt.Printf("Energia: %.4f\n", features.Energy)
	fmt.Printf("Entropia: %.4f\n", features.Entropy)
	fmt.Printf("Homogeneidad: %.4f\n", features.Homogeneity)
	fmt.Printf("Varianza: %.4f\n", features.Variance)

	// Calculate Local Binary Pattern
	fmt.Println("\nCalculando patrones binarios locales...")
	lbp := calculateLBP(gray)
	lbpHistogram := calculateLBPHistogram(lbp)
	fmt.Printf("Histograma LBP (primeras 10 bins): %v\n", lbpHistogram[:10])

	return nil
}

// Helper functions
func findMin(matrix [][]float64) float64 {
	min := matrix[0][0]
	for _, row := range matrix {
		for _, val := range row {
			if val < min {
				min = val
			}
		}
	}
	return min
}

func findMax(matrix [][]float64) float64 {
	max := matrix[0][0]
	for _, row := range matrix {
		for _, val := range row {
			if val > max {
				max = val
			}
		}
	}
	return max
}

func calculateLBPHistogram(lbp [][]uint8) []int {
	histogram := make([]int, 256)
	for _, row := range lbp {
		for _, val := range row {
			histogram[val]++
		}
	}
	return histogram
}

func main() {
	if len(os.Args) < 2 {
		log.Fatal("Usage: go run main.go <ruta>")
	}

	imagePath := os.Args[1]
	if err := analyzeTexture(imagePath); err != nil {
		log.Fatal(err)
	}
}
