import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Set matplotlib to use default fonts for better compatibility
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


class AutoImageEnhancement:
    """
    Intelligent Auto Image Enhancement System

    This class provides automatic image enhancement using multiple spatial domain methods:
    1. Histogram Equalization - improves contrast by redistributing gray levels
    2. Grayscale Stretching - expands the dynamic range of the image
    3. Gamma Correction - non-linear brightness adjustment

    The system can automatically select the best method or apply all methods for comparison.
    """

    def __init__(self, output_dir="spatial_domain_enhancement_results"):
        """
        Initialize the enhancement system

        Args:
            output_dir (str): Directory to save results (default: "spatial_domain_enhancement_results")
        """
        # Initialize without loading any specific image
        self.original_image = None

        # Create output directory for saving results
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def analyze_image_features(self, image):
        """
        Analyze statistical features of an image to guide enhancement decisions

        Args:
            image (numpy.ndarray): Input grayscale image

        Returns:
            dict: Dictionary containing image statistics
        """
        features = {
            "mean": np.mean(image),  # Average brightness (0-255)
            "std": np.std(image),  # Contrast measure (standard deviation)
            "min": np.min(image),  # Darkest pixel value
            "max": np.max(image),  # Brightest pixel value
            "dynamic_range": np.max(image) - np.min(image),  # Total gray level span
            "median": np.median(image),  # Middle brightness value
        }
        return features

    def histogram_equalization(self, image):
        """
        Apply histogram equalization to improve image contrast

        Principle: Redistributes gray levels to make histogram more uniform
        Best for: Images with poor contrast or narrow histogram

        Args:
            image (numpy.ndarray): Input grayscale image

        Returns:
            numpy.ndarray: Enhanced image with equalized histogram
        """
        # OpenCV's built-in histogram equalization
        enhanced_img = cv2.equalizeHist(image)
        return enhanced_img

    def grayscale_stretching(self, image, percentile=2):
        """
        Apply linear grayscale stretching to expand dynamic range

        Principle: Linearly maps input range to full 0-255 range
        Best for: Images with narrow gray level range

        Args:
            image (numpy.ndarray): Input grayscale image
            percentile (int): Percentage of outliers to ignore (default: 2%)

        Returns:
            numpy.ndarray: Stretched image with expanded dynamic range
        """
        # Use percentiles to ignore outliers and determine stretch range
        min_val = np.percentile(image, percentile)
        max_val = np.percentile(image, 100 - percentile)

        # If range is too small, use full range
        if max_val - min_val < 50:
            min_val = np.min(image)
            max_val = np.max(image)

        # Apply linear stretching formula: new = (old - min) * 255 / (max - min)
        if max_val > min_val:
            stretched = (image - min_val) * 255.0 / (max_val - min_val)
            result = np.clip(stretched, 0, 255).astype(np.uint8)
        else:
            result = image.copy()

        return result

    def gamma_correction(self, image, gamma=1.0):
        """
        Apply gamma correction for non-linear brightness adjustment

        Principle: Uses power function transformation: output = input^gamma
        - gamma < 1: brightens image (enhances dark regions)
        - gamma > 1: darkens image (enhances bright regions)
        - gamma = 1: no change

        Args:
            image (numpy.ndarray): Input grayscale image
            gamma (float): Gamma value for correction

        Returns:
            numpy.ndarray: Gamma corrected image
        """
        # Normalize to 0-1 range for gamma correction
        normalized = image / 255.0

        # Apply gamma correction: new_pixel = pixel^gamma
        corrected = np.power(normalized, gamma)

        # Convert back to 0-255 range
        result = (corrected * 255).astype(np.uint8)

        return result

    def auto_gamma_correction(self, image, target_mean=128):
        """
        Automatically calculate optimal gamma value based on image brightness

        Args:
            image (numpy.ndarray): Input grayscale image
            target_mean (int): Desired mean brightness (default: 128)

        Returns:
            tuple: (corrected_image, gamma_value_used)
        """
        current_mean = np.mean(image)

        # Calculate gamma to achieve target mean brightness
        if current_mean > 0:
            # Gamma formula: gamma = log(target/255) / log(current/255)
            gamma = np.log(target_mean / 255.0) / np.log(current_mean / 255.0)
            # Constrain gamma to reasonable range
            gamma = np.clip(gamma, 0.3, 3.0)
        else:
            gamma = 1.0  # No correction if image is completely black

        # Apply the calculated gamma correction
        result = self.gamma_correction(image, gamma)
        return result, gamma

    def calculate_image_quality_score(self, image):
        """
        Calculate a composite image quality score for enhancement evaluation

        Combines multiple metrics:
        - Contrast (standard deviation): Higher is generally better
        - Information entropy: Measures information content
        - Dynamic range utilization: How well the full 0-255 range is used

        Args:
            image (numpy.ndarray): Input grayscale image

        Returns:
            float: Quality score (higher is better)
        """
        # Measure 1: Contrast (standard deviation of pixel values)
        contrast = np.std(image)

        # Measure 2: Information entropy (measure of information content)
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)  # Normalize histogram
        hist = hist[hist > 0]  # Remove zero entries to avoid log(0)
        entropy = -np.sum(hist * np.log2(hist))

        # Measure 3: Dynamic range utilization
        dynamic_range = np.max(image) - np.min(image)
        range_utilization = dynamic_range / 255.0

        # Weighted combination of quality measures
        quality_score = (
            0.4 * contrast  # 40% weight on contrast
            + 0.4 * entropy * 8  # 40% weight on entropy (scaled)
            + 0.2 * range_utilization * 100  # 20% weight on range utilization
        )

        return quality_score

    def apply_all_enhancement_methods(self, image):
        """
        Apply all three enhancement methods to the input image

        Args:
            image (numpy.ndarray): Input grayscale image

        Returns:
            dict: Dictionary containing all enhanced versions and their quality scores
        """
        results = {}

        print("Applying all enhancement methods...")

        # Method 1: Histogram Equalization
        print("  1. Histogram Equalization...")
        hist_eq = self.histogram_equalization(image)
        hist_eq_quality = self.calculate_image_quality_score(hist_eq)
        results["histogram_equalization"] = {
            "image": hist_eq,
            "quality": hist_eq_quality,
            "name": "Histogram Equalization",
        }

        # Method 2: Grayscale Stretching
        print("  2. Grayscale Stretching...")
        stretched = self.grayscale_stretching(image)
        stretched_quality = self.calculate_image_quality_score(stretched)
        results["grayscale_stretching"] = {
            "image": stretched,
            "quality": stretched_quality,
            "name": "Grayscale Stretching",
        }

        # Method 3: Auto Gamma Correction
        print("  3. Auto Gamma Correction...")
        gamma_corrected, gamma_val = self.auto_gamma_correction(image)
        gamma_quality = self.calculate_image_quality_score(gamma_corrected)
        results["gamma_correction"] = {
            "image": gamma_corrected,
            "quality": gamma_quality,
            "name": f"Gamma Correction (γ={gamma_val:.2f})",
            "gamma_value": gamma_val,
        }

        # Calculate original image quality for comparison
        original_quality = self.calculate_image_quality_score(image)
        results["original"] = {
            "image": image,
            "quality": original_quality,
            "name": "Original",
        }

        return results

    def intelligent_auto_enhancement(self, image):
        """
        Intelligently select the best enhancement method based on image characteristics

        Decision tree:
        1. If dynamic range < 100 → Grayscale stretching
        2. If contrast (std) < 30 → Histogram equalization
        3. If too dark (mean < 85) → Gamma correction (brighten)
        4. If too bright (mean > 170) → Gamma correction (darken)
        5. Otherwise → Test all methods and select best

        Args:
            image (numpy.ndarray): Input grayscale image

        Returns:
            tuple: (enhanced_image, method_name, improvement_flag)
        """
        print("\n" + "=" * 60)
        print("Intelligent Auto Image Enhancement Analysis")
        print("=" * 60)

        # Analyze original image characteristics
        features = self.analyze_image_features(image)
        original_quality = self.calculate_image_quality_score(image)

        print(f"Original Image Features:")
        print(f"  Mean: {features['mean']:.1f}")
        print(f"  Std Dev: {features['std']:.1f}")
        print(f"  Dynamic Range: {features['dynamic_range']:.0f}")
        print(f"  Quality Score: {original_quality:.1f}")

        # Decision tree for automatic method selection
        enhanced_image = None
        method_used = ""

        # Strategy 1: Dynamic range too small → Grayscale stretching
        if features["dynamic_range"] < 100:
            print(
                f"\nStrategy: Dynamic range too small ({features['dynamic_range']:.0f} < 100)"
            )
            enhanced_image = self.grayscale_stretching(image)
            method_used = "Adaptive Grayscale Stretching"
            print(f"Applied Method: {method_used}")

        # Strategy 2: Contrast too low → Histogram equalization
        elif features["std"] < 30:
            print(f"\nStrategy: Low contrast (Std Dev {features['std']:.1f} < 30)")
            enhanced_image = self.histogram_equalization(image)
            method_used = "Histogram Equalization"
            print(f"Applied Method: {method_used}")

        # Strategy 3: Image too dark → Gamma correction (brighten)
        elif features["mean"] < 85:
            print(f"\nStrategy: Image too dark (Mean {features['mean']:.1f} < 85)")
            enhanced_image, gamma = self.auto_gamma_correction(image, target_mean=120)
            method_used = f"Auto Gamma Correction (γ={gamma:.2f}, Brighten)"
            print(f"Applied Method: {method_used}")

        # Strategy 4: Image too bright → Gamma correction (darken)
        elif features["mean"] > 170:
            print(f"\nStrategy: Image too bright (Mean {features['mean']:.1f} > 170)")
            enhanced_image, gamma = self.auto_gamma_correction(image, target_mean=135)
            method_used = f"Auto Gamma Correction (γ={gamma:.2f}, Darken)"
            print(f"Applied Method: {method_used}")

        # Strategy 5: Quality-based selection when image appears normal
        else:
            print(
                f"\nStrategy: Image appears normal, performing multi-method quality assessment"
            )
            enhanced_image, method_used = self.quality_based_selection(image)

        # Evaluate enhancement effectiveness
        enhanced_features = self.analyze_image_features(enhanced_image)
        enhanced_quality = self.calculate_image_quality_score(enhanced_image)

        print(f"\nEnhanced Image Features:")
        print(
            f"  Mean: {enhanced_features['mean']:.1f} (Change: {enhanced_features['mean']-features['mean']:+.1f})"
        )
        print(
            f"  Std Dev: {enhanced_features['std']:.1f} (Change: {enhanced_features['std']-features['std']:+.1f})"
        )
        print(
            f"  Dynamic Range: {enhanced_features['dynamic_range']:.0f} (Change: {enhanced_features['dynamic_range']-features['dynamic_range']:+.0f})"
        )
        print(
            f"  Quality Score: {enhanced_quality:.1f} (Change: {enhanced_quality-original_quality:+.1f})"
        )

        improvement = enhanced_quality > original_quality
        print(
            f"\nEnhancement Result: {'✓ Improved' if improvement else '✗ No Clear Improvement'}"
        )

        return enhanced_image, method_used, improvement

    def quality_based_selection(self, image):
        """
        Test all enhancement methods and select the one with highest quality score

        Args:
            image (numpy.ndarray): Input grayscale image

        Returns:
            tuple: (best_enhanced_image, method_name)
        """
        print("  Testing multiple methods, selecting highest quality score:")

        methods = {}

        # Test Method 1: Histogram equalization
        hist_eq = self.histogram_equalization(image)
        hist_eq_quality = self.calculate_image_quality_score(hist_eq)
        methods["Histogram Equalization"] = (hist_eq, hist_eq_quality)
        print(f"    Histogram Equalization quality score: {hist_eq_quality:.1f}")

        # Test Method 2: Adaptive grayscale stretching
        stretch = self.grayscale_stretching(image)
        stretch_quality = self.calculate_image_quality_score(stretch)
        methods["Adaptive Grayscale Stretching"] = (stretch, stretch_quality)
        print(f"    Adaptive Grayscale Stretching quality score: {stretch_quality:.1f}")

        # Test Method 3: Auto gamma correction
        gamma_corrected, gamma_val = self.auto_gamma_correction(image)
        gamma_quality = self.calculate_image_quality_score(gamma_corrected)
        methods[f"Auto Gamma Correction (γ={gamma_val:.2f})"] = (
            gamma_corrected,
            gamma_quality,
        )
        print(f"    Auto Gamma Correction quality score: {gamma_quality:.1f}")

        # Select method with highest quality score
        best_method = max(methods.items(), key=lambda x: x[1][1])
        method_name = best_method[0]
        best_image = best_method[1][0]

        print(f"  Selected best method: {method_name}")

        return best_image, method_name

    def save_all_enhanced_images_to_dir(self, results, output_dir):
        """
        Save all enhanced images to a specific directory

        Args:
            results (dict): Dictionary containing all enhancement results
            output_dir (str): Specific directory to save images
        """
        print(f"Saving all enhanced images to {output_dir}...")

        # Save original image
        cv2.imwrite(f"{output_dir}/01_original.jpg", results["original"]["image"])
        print(f"  Saved: 01_original.jpg")

        # Save histogram equalization result
        cv2.imwrite(
            f"{output_dir}/02_histogram_equalization.jpg",
            results["histogram_equalization"]["image"],
        )
        print(f"  Saved: 02_histogram_equalization.jpg")

        # Save grayscale stretching result
        cv2.imwrite(
            f"{output_dir}/03_grayscale_stretching.jpg",
            results["grayscale_stretching"]["image"],
        )
        print(f"  Saved: 03_grayscale_stretching.jpg")

        # Save gamma correction result
        cv2.imwrite(
            f"{output_dir}/04_gamma_correction.jpg",
            results["gamma_correction"]["image"],
        )
        print(f"  Saved: 04_gamma_correction.jpg")

    def create_comprehensive_comparison_to_dir(self, results, output_dir):
        """
        Create a comprehensive comparison plot and save to specific directory

        Args:
            results (dict): Dictionary containing all enhancement results
            output_dir (str): Directory to save the comparison plot
        """
        # Create a 3x4 subplot layout for comprehensive comparison
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))

        # Row 1: Display all images
        methods = [
            "original",
            "histogram_equalization",
            "grayscale_stretching",
            "gamma_correction",
        ]

        for i, method in enumerate(methods):
            axes[0, i].imshow(results[method]["image"], cmap="gray")
            title = (
                f"{results[method]['name']}\nQuality: {results[method]['quality']:.1f}"
            )
            axes[0, i].set_title(title, fontsize=10)
            axes[0, i].axis("off")

        # Row 2: Individual histograms
        colors = ["blue", "red", "green", "orange"]
        for i, method in enumerate(methods):
            axes[1, i].hist(
                results[method]["image"].ravel(), bins=256, alpha=0.7, color=colors[i]
            )
            axes[1, i].set_title(f"{results[method]['name']} Histogram", fontsize=10)
            axes[1, i].set_xlabel("Gray Level")
            axes[1, i].set_ylabel("Pixel Count")

        # Row 3: Comparison plots

        # Quality score comparison (bar chart)
        axes[2, 0].bar(
            range(len(methods)),
            [results[method]["quality"] for method in methods],
            color=colors,
        )
        axes[2, 0].set_title("Quality Score Comparison")
        axes[2, 0].set_ylabel("Quality Score")
        axes[2, 0].set_xticks(range(len(methods)))
        axes[2, 0].set_xticklabels(
            [results[method]["name"].split("(")[0].strip() for method in methods],
            rotation=45,
            ha="right",
        )

        # Overlaid histogram comparison
        for i, method in enumerate(methods):
            axes[2, 1].hist(
                results[method]["image"].ravel(),
                bins=256,
                alpha=0.4,
                label=results[method]["name"].split("(")[0].strip(),
                color=colors[i],
            )
        axes[2, 1].set_title("All Histograms Overlaid")
        axes[2, 1].set_xlabel("Gray Level")
        axes[2, 1].set_ylabel("Pixel Count")
        axes[2, 1].legend()

        # Statistical comparison
        stats = ["mean", "std", "dynamic_range"]
        stat_names = ["Mean", "Std Dev", "Dynamic Range"]

        for j, stat in enumerate(stats):
            if j < 2:  # Only plot mean and std in available subplots
                method_names = [
                    results[method]["name"].split("(")[0].strip() for method in methods
                ]
                features_list = [
                    self.analyze_image_features(results[method]["image"])
                    for method in methods
                ]
                stat_values = [features[stat] for features in features_list]

                axes[2, 2 + j].bar(range(len(methods)), stat_values, color=colors)
                axes[2, 2 + j].set_title(f"{stat_names[j]} Comparison")
                axes[2, 2 + j].set_ylabel(stat_names[j])
                axes[2, 2 + j].set_xticks(range(len(methods)))
                axes[2, 2 + j].set_xticklabels(method_names, rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/comprehensive_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()  # Close the figure to free memory
        print(f"  Saved: comprehensive_comparison.png")

    def save_all_enhanced_images(self, results):
        """
        Save all enhanced images to the main output directory (for single image processing)

        Args:
            results (dict): Dictionary containing all enhancement results
        """
        self.save_all_enhanced_images_to_dir(results, self.output_dir)

    def create_comprehensive_comparison(self, results):
        """
        Create comprehensive comparison plot in main output directory (for single image processing)

        Args:
            results (dict): Dictionary containing all enhancement results
        """
        self.create_comprehensive_comparison_to_dir(results, self.output_dir)

    def print_detailed_analysis(self, results):
        """
        Print detailed analysis of all enhancement methods

        Args:
            results (dict): Dictionary containing all enhancement results
        """
        print("\n" + "=" * 80)
        print("DETAILED ENHANCEMENT ANALYSIS")
        print("=" * 80)

        # Print quality scores ranking
        sorted_methods = sorted(
            results.items(), key=lambda x: x[1]["quality"], reverse=True
        )

        print("\nQuality Score Ranking (Higher is Better):")
        print("-" * 50)
        for i, (method_key, data) in enumerate(sorted_methods, 1):
            improvement = (
                "↑" if data["quality"] > results["original"]["quality"] else "↓"
            )
            quality_change = data["quality"] - results["original"]["quality"]
            print(
                f"{i}. {data['name']:<30} Quality: {data['quality']:.1f} "
                f"({improvement} {quality_change:+.1f})"
            )

        # Print detailed statistics
        print(f"\nDetailed Statistics:")
        print("-" * 70)
        print(f"{'Method':<30} {'Mean':<8} {'Std':<8} {'Range':<8} {'Quality':<8}")
        print("-" * 70)

        for method_key in [
            "original",
            "histogram_equalization",
            "grayscale_stretching",
            "gamma_correction",
        ]:
            data = results[method_key]
            features = self.analyze_image_features(data["image"])
            print(
                f"{data['name']:<30} {features['mean']:<8.1f} {features['std']:<8.1f} "
                f"{features['dynamic_range']:<8.0f} {data['quality']:<8.1f}"
            )

        # Recommendations
        best_method = sorted_methods[0]
        print(f"\nRecommendation:")
        print(f"Best method: {best_method[1]['name']}")

        if best_method[0] == "original":
            print(
                "The original image is already well-balanced and doesn't need enhancement."
            )
        else:
            improvement = best_method[1]["quality"] - results["original"]["quality"]
            print(f"This method improves quality score by {improvement:.1f} points.")

    def process_single_image(self, image_path):
        """
        Process a single image with all enhancement methods

        Args:
            image_path (str): Path to the input image file

        Returns:
            dict: Complete results dictionary with all enhanced images

        Raises:
            ValueError: If the image cannot be read
        """
        # Load image in grayscale mode for processing
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        self.original_image = image

        print(f"Processing single image: {image_path}")

        # Apply all enhancement methods
        results = self.apply_all_enhancement_methods(image)

        # Save all enhanced images
        self.save_all_enhanced_images(results)

        # Create comprehensive comparison visualization
        self.create_comprehensive_comparison(results)

        # Print detailed analysis
        self.print_detailed_analysis(results)

        # Also run intelligent auto-selection for comparison
        print("\n" + "=" * 60)
        print("INTELLIGENT AUTO-SELECTION COMPARISON")
        print("=" * 60)
        auto_enhanced, auto_method, auto_improvement = (
            self.intelligent_auto_enhancement(image)
        )

        # Save auto-selected result
        cv2.imwrite(f"{self.output_dir}/auto_selected_best.jpg", auto_enhanced)

        return results

    def batch_process_images(self, image_folder):
        """
        Process multiple images in a folder using the comprehensive enhancement approach

        Args:
            image_folder (str): Path to folder containing images

        Returns:
            list: Results summary for all processed images
        """
        print(f"\nStarting batch processing folder: {image_folder}")

        # Supported image file formats
        supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
        batch_results = []

        for filename in os.listdir(image_folder):
            if filename.lower().endswith(supported_formats):
                print(f"\n{'='*40}")
                print(f"Processing: {filename}")
                print("=" * 40)

                try:
                    # Read image
                    image_path = os.path.join(image_folder, filename)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                    if image is None:
                        print(f"  Skipping: Cannot read {filename}")
                        continue

                    # Temporarily update original image for processing
                    self.original_image = image

                    # Create subfolder for this image's results
                    base_name = os.path.splitext(filename)[0]
                    image_output_dir = f"{self.output_dir}/{base_name}"
                    if not os.path.exists(image_output_dir):
                        os.makedirs(image_output_dir)

                    # Process with all methods (pass the specific output directory)
                    results = self.apply_all_enhancement_methods(image)
                    self.save_all_enhanced_images_to_dir(results, image_output_dir)

                    # Create comprehensive comparison for this image
                    self.create_comprehensive_comparison_to_dir(
                        results, image_output_dir
                    )

                    # Print detailed analysis for this image
                    self.print_detailed_analysis(results)

                    # Also run intelligent auto-selection
                    print(f"\nIntelligent Auto-Selection for {filename}:")
                    print("-" * 50)
                    auto_enhanced, auto_method, auto_improvement = (
                        self.intelligent_auto_enhancement(image)
                    )
                    cv2.imwrite(
                        f"{image_output_dir}/auto_selected_best.jpg", auto_enhanced
                    )

                    # Find best method
                    best_method = max(
                        results.items(),
                        key=lambda x: x[1]["quality"] if x[0] != "original" else 0,
                    )

                    batch_results.append(
                        {
                            "filename": filename,
                            "best_method": best_method[1]["name"],
                            "quality_improvement": best_method[1]["quality"]
                            - results["original"]["quality"],
                            "original_quality": results["original"]["quality"],
                            "best_quality": best_method[1]["quality"],
                            "all_methods_quality": {
                                "original": results["original"]["quality"],
                                "histogram_eq": results["histogram_equalization"][
                                    "quality"
                                ],
                                "grayscale_stretch": results["grayscale_stretching"][
                                    "quality"
                                ],
                                "gamma_correction": results["gamma_correction"][
                                    "quality"
                                ],
                            },
                            "auto_selected_method": auto_method,
                            "auto_improvement": auto_improvement,
                        }
                    )

                    print(f"  Best method: {best_method[1]['name']}")
                    print(
                        f"  Quality improvement: {best_method[1]['quality'] - results['original']['quality']:+.1f}"
                    )
                    print(f"  Auto-selected method: {auto_method}")
                    print(
                        f"  Auto-selection improvement: {'Yes' if auto_improvement else 'No'}"
                    )

                    # Save detailed results to text file
                    self.save_detailed_results_to_file(
                        filename,
                        results,
                        auto_method,
                        auto_improvement,
                        image_output_dir,
                    )

                except Exception as e:
                    print(f"  Error: Processing {filename} failed: {e}")

        # Print batch summary
        self.print_batch_summary(batch_results)
        return batch_results

    def save_detailed_results_to_file(
        self, filename, results, auto_method, auto_improvement, output_dir
    ):
        """
        Save detailed analysis results to a text file for each image

        Args:
            filename (str): Original filename
            results (dict): Enhancement results dictionary
            auto_method (str): Auto-selected method name
            auto_improvement (bool): Whether auto-selection improved quality
            output_dir (str): Directory to save the analysis file
        """
        analysis_file = f"{output_dir}/analysis_report.txt"

        with open(analysis_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write(f"IMAGE ENHANCEMENT ANALYSIS REPORT\n")
            f.write(f"Image: {filename}\n")
            f.write("=" * 80 + "\n\n")

            # Original image statistics
            original_features = self.analyze_image_features(
                results["original"]["image"]
            )
            f.write("ORIGINAL IMAGE STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Mean Brightness: {original_features['mean']:.1f}\n")
            f.write(f"Standard Deviation: {original_features['std']:.1f}\n")
            f.write(f"Dynamic Range: {original_features['dynamic_range']:.0f}\n")
            f.write(f"Min Value: {original_features['min']:.0f}\n")
            f.write(f"Max Value: {original_features['max']:.0f}\n")
            f.write(f"Quality Score: {results['original']['quality']:.1f}\n\n")

            # All enhancement methods comparison
            f.write("ENHANCEMENT METHODS COMPARISON:\n")
            f.write("-" * 40 + "\n")

            methods = [
                "histogram_equalization",
                "grayscale_stretching",
                "gamma_correction",
            ]
            method_names = [
                "Histogram Equalization",
                "Grayscale Stretching",
                "Gamma Correction",
            ]

            for i, method in enumerate(methods):
                enhanced_features = self.analyze_image_features(
                    results[method]["image"]
                )
                quality_change = (
                    results[method]["quality"] - results["original"]["quality"]
                )

                f.write(f"\n{method_names[i]}:\n")
                f.write(
                    f"  Quality Score: {results[method]['quality']:.1f} (Change: {quality_change:+.1f})\n"
                )
                f.write(
                    f"  Mean: {enhanced_features['mean']:.1f} (Change: {enhanced_features['mean'] - original_features['mean']:+.1f})\n"
                )
                f.write(
                    f"  Std Dev: {enhanced_features['std']:.1f} (Change: {enhanced_features['std'] - original_features['std']:+.1f})\n"
                )
                f.write(
                    f"  Dynamic Range: {enhanced_features['dynamic_range']:.0f} (Change: {enhanced_features['dynamic_range'] - original_features['dynamic_range']:+.0f})\n"
                )

            # Best method ranking
            f.write(f"\nMETHOD RANKING (by Quality Score):\n")
            f.write("-" * 40 + "\n")

            all_methods = [
                ("Original", results["original"]["quality"]),
                (
                    "Histogram Equalization",
                    results["histogram_equalization"]["quality"],
                ),
                ("Grayscale Stretching", results["grayscale_stretching"]["quality"]),
                ("Gamma Correction", results["gamma_correction"]["quality"]),
            ]

            sorted_methods = sorted(all_methods, key=lambda x: x[1], reverse=True)

            for i, (method_name, quality) in enumerate(sorted_methods, 1):
                improvement_symbol = (
                    "↑"
                    if quality > results["original"]["quality"]
                    else "→" if quality == results["original"]["quality"] else "↓"
                )
                f.write(
                    f"{i}. {method_name:<25} Quality: {quality:.1f} {improvement_symbol}\n"
                )

            # Auto-selection analysis
            f.write(f"\nINTELLIGENT AUTO-SELECTION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Selected Method: {auto_method}\n")
            f.write(f"Quality Improvement: {'Yes' if auto_improvement else 'No'}\n")

            # Recommendations
            f.write(f"\nRECOMMENDATIONS:\n")
            f.write("-" * 40 + "\n")

            best_manual = sorted_methods[0]
            if best_manual[0] == "Original":
                f.write(
                    "The original image is already well-balanced and doesn't need enhancement.\n"
                )
            else:
                improvement = best_manual[1] - results["original"]["quality"]
                f.write(f"Best enhancement method: {best_manual[0]}\n")
                f.write(f"Quality improvement: +{improvement:.1f} points\n")

            f.write(f"\nAuto-selected method: {auto_method}\n")
            if auto_improvement:
                f.write(
                    "The auto-selection algorithm successfully improved the image quality.\n"
                )
            else:
                f.write(
                    "The auto-selection algorithm determined the original image was satisfactory.\n"
                )

            # Technical notes
            f.write(f"\nTECHNICAL NOTES:\n")
            f.write("-" * 40 + "\n")
            f.write("Quality Score Calculation:\n")
            f.write("- 40% Contrast (Standard Deviation)\n")
            f.write("- 40% Information Entropy\n")
            f.write("- 20% Dynamic Range Utilization\n\n")

            f.write("Enhancement Method Principles:\n")
            f.write(
                "- Histogram Equalization: Redistributes gray levels for uniform histogram\n"
            )
            f.write("- Grayscale Stretching: Linearly expands dynamic range to 0-255\n")
            f.write(
                "- Gamma Correction: Non-linear brightness adjustment using power function\n"
            )

        print(f"  Detailed analysis saved to: analysis_report.txt")
    
    def print_batch_summary(self, batch_results):
        """
        Print summary of batch processing results
        
        Args:
            batch_results (list): List of processing results for each image
        """
        print(f"\n" + "=" * 80)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 80)

        if not batch_results:
            print("No images processed successfully.")
            return

        improved_count = sum(1 for r in batch_results if r["quality_improvement"] > 0)
        avg_improvement = np.mean([r["quality_improvement"] for r in batch_results])

        print(f"Total images processed: {len(batch_results)}")
        print(f"Images with improvement: {improved_count}")
        print(f"Improvement rate: {improved_count/len(batch_results)*100:.1f}%")
        print(f"Average quality improvement: {avg_improvement:+.1f}")

        # Method usage statistics
        method_counts = {}
        for result in batch_results:
            method = result["best_method"].split("(")[0].strip()
            method_counts[method] = method_counts.get(method, 0) + 1

        print(f"\nMost effective methods:")
        for method, count in sorted(
            method_counts.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {method}: {count} images ({count/len(batch_results)*100:.1f}%)")


def main():
    """
    Main function demonstrating the comprehensive image enhancement system
    """
    print("Comprehensive Auto Image Enhancement System")
    print("=" * 60)
    print("This system applies all three spatial domain enhancement methods:")
    print("1. Histogram Equalization")
    print("2. Grayscale Stretching")
    print("3. Gamma Correction")
    print("And provides detailed comparison and analysis.")
    print("=" * 60)

    try:
        # Create enhancement system (no specific image needed)
        enhancer = AutoImageEnhancement()

        # Batch process all images in MV01 folder
        print("Starting batch processing of MV01 folder...")
        batch_results = enhancer.batch_process_images("MV01")

        print(f"\nBatch processing completed successfully!")
        print(f"Results saved in: {enhancer.output_dir}")
        print(f"\nFor each image, the following files were generated:")
        print(f"  - 01_original.jpg: Original image")
        print(f"  - 02_histogram_equalization.jpg: Histogram equalization result")
        print(f"  - 03_grayscale_stretching.jpg: Grayscale stretching result")
        print(f"  - 04_gamma_correction.jpg: Gamma correction result")

        # Uncomment these lines if you want to process a single image instead
        # image_path = "MV01/filter_test01.jpg"
        # results = enhancer.process_single_image(image_path)

    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Verify the MV01 folder exists in the current directory")
        print("2. Ensure the folder contains readable image files")
        print("3. Check that required libraries are installed:")
        print("   pip install opencv-python numpy matplotlib")
        print("4. Make sure you have write permissions in the current directory")


if __name__ == "__main__":
    main()