import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Set matplotlib to use default fonts for better compatibility
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


class FrequencyDomainImageEnhancement:
    """
    Frequency Domain Image Enhancement System

    This class provides automatic image enhancement using frequency domain methods:
    1. Ideal Low-pass Filter - sharp cutoff frequency filter
    2. Butterworth Low-pass Filter - smooth transition filter
    3. Gaussian Low-pass Filter - smoothest transition filter

    The system can automatically select the best method or apply all methods for comparison.
    """

    def __init__(self, output_dir="frequency_domain_enhancement_results"):
        """
        Initialize the enhancement system

        Args:
            output_dir (str): Directory to save results (default: "frequency_domain_enhancement_results")
        """
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
            "mean": np.mean(image),
            "std": np.std(image),
            "min": np.min(image),
            "max": np.max(image),
            "dynamic_range": np.max(image) - np.min(image),
            "median": np.median(image),
        }
        return features

    def ideal_lowpass_filter(self, image, cutoff_frequency=30):
        """
        Apply ideal low-pass filter in frequency domain

        Principle: Sharp cutoff at specified frequency - removes all frequencies above cutoff
        Characteristics: Causes ringing artifacts due to sharp transition
        Best for: Noise removal when ringing is acceptable

        Args:
            image (numpy.ndarray): Input grayscale image
            cutoff_frequency (int): Cutoff frequency for the filter (default: 30)

        Returns:
            numpy.ndarray: Filtered image
        """
        # Get image dimensions
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2  # Center coordinates

        # Apply FFT to convert to frequency domain
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)  # Shift zero frequency to center

        # Create ideal low-pass filter mask
        mask = np.zeros((rows, cols), np.uint8)

        # Create distance matrix from center
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)

        # Set mask to 1 for frequencies within cutoff, 0 outside
        mask[distance <= cutoff_frequency] = 1

        # Apply filter mask in frequency domain
        f_shift_filtered = f_shift * mask

        # Convert back to spatial domain
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)  # Take magnitude (remove complex part)

        # Normalize to 0-255 range
        result = np.clip(img_back, 0, 255).astype(np.uint8)
        return result, mask

    def butterworth_lowpass_filter(self, image, cutoff_frequency=30, order=2):
        """
        Apply Butterworth low-pass filter in frequency domain

        Principle: Smooth transition filter with adjustable rolloff rate
        Formula: H(u,v) = 1 / (1 + (D(u,v)/D0)^(2n))
        Characteristics: Reduces ringing compared to ideal filter
        Best for: Good balance between noise removal and artifact reduction

        Args:
            image (numpy.ndarray): Input grayscale image
            cutoff_frequency (int): Cutoff frequency for the filter (default: 30)
            order (int): Filter order - higher order = sharper transition (default: 2)

        Returns:
            tuple: (filtered_image, filter_mask)
        """
        # Get image dimensions
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2  # Center coordinates

        # Apply FFT to convert to frequency domain
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)

        # Create distance matrix from center
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)

        # Create Butterworth filter mask
        # Avoid division by zero
        distance[distance == 0] = 1e-10
        mask = 1 / (1 + (distance / cutoff_frequency) ** (2 * order))

        # Apply filter mask in frequency domain
        f_shift_filtered = f_shift * mask

        # Convert back to spatial domain
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        # Normalize to 0-255 range
        result = np.clip(img_back, 0, 255).astype(np.uint8)
        return result, mask

    def gaussian_lowpass_filter(self, image, cutoff_frequency=30):
        """
        Apply Gaussian low-pass filter in frequency domain

        Principle: Uses Gaussian function for smoothest possible transition
        Formula: H(u,v) = exp(-(D(u,v)^2)/(2*σ^2))
        Characteristics: No ringing artifacts, smoothest filtering
        Best for: Applications where smooth filtering is critical

        Args:
            image (numpy.ndarray): Input grayscale image
            cutoff_frequency (int): Cutoff frequency (related to σ) (default: 30)

        Returns:
            tuple: (filtered_image, filter_mask)
        """
        # Get image dimensions
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2  # Center coordinates

        # Apply FFT to convert to frequency domain
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)

        # Create distance matrix from center
        y, x = np.ogrid[:rows, :cols]
        distance_squared = (x - ccol) ** 2 + (y - crow) ** 2

        # Create Gaussian filter mask
        # Convert cutoff frequency to sigma parameter
        sigma = cutoff_frequency / 2.0
        mask = np.exp(-(distance_squared) / (2 * sigma**2))

        # Apply filter mask in frequency domain
        f_shift_filtered = f_shift * mask

        # Convert back to spatial domain
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        # Normalize to 0-255 range
        result = np.clip(img_back, 0, 255).astype(np.uint8)
        return result, mask

    def analyze_frequency_content(self, image):
        """
        Analyze frequency content of an image to guide filter selection

        Args:
            image (numpy.ndarray): Input grayscale image

        Returns:
            dict: Dictionary containing frequency analysis results
        """
        # Apply FFT
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)

        # Calculate frequency distribution statistics
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)

        # Analyze energy distribution
        spectrum_magnitude = np.abs(f_shift)

        # Calculate energy in different frequency bands
        low_freq_mask = distance <= rows // 8  # Low frequencies
        mid_freq_mask = (distance > rows // 8) & (
            distance <= rows // 4
        )  # Mid frequencies
        high_freq_mask = distance > rows // 4  # High frequencies

        total_energy = np.sum(spectrum_magnitude**2)
        low_freq_energy = (
            np.sum((spectrum_magnitude * low_freq_mask) ** 2) / total_energy
        )
        mid_freq_energy = (
            np.sum((spectrum_magnitude * mid_freq_mask) ** 2) / total_energy
        )
        high_freq_energy = (
            np.sum((spectrum_magnitude * high_freq_mask) ** 2) / total_energy
        )

        # Estimate optimal cutoff frequency
        optimal_cutoff = min(50, max(10, rows // 8))  # Conservative estimate

        analysis = {
            "low_freq_energy_ratio": low_freq_energy,
            "mid_freq_energy_ratio": mid_freq_energy,
            "high_freq_energy_ratio": high_freq_energy,
            "estimated_optimal_cutoff": optimal_cutoff,
            "magnitude_spectrum": magnitude_spectrum,
            "noise_level": high_freq_energy,
            "f_shift": f_shift,
        }

        return analysis

    def calculate_image_quality_score(self, image):
        """
        Calculate a composite image quality score for enhancement evaluation

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

    def apply_all_frequency_methods(self, image):
        """
        Apply all frequency domain enhancement methods to the input image

        Args:
            image (numpy.ndarray): Input grayscale image

        Returns:
            dict: Dictionary containing all enhanced versions and their quality scores
        """
        results = {}

        print("Applying frequency domain enhancement methods...")

        # Original image
        original_quality = self.calculate_image_quality_score(image)
        results["original"] = {
            "image": image,
            "quality": original_quality,
            "name": "Original",
        }

        # Analyze frequency content to select optimal cutoff
        freq_analysis = self.analyze_frequency_content(image)
        optimal_cutoff = freq_analysis["estimated_optimal_cutoff"]
        print(f"  Estimated optimal cutoff frequency: {optimal_cutoff}")

        # Method 1: Ideal Low-pass Filter
        print("  1. Ideal Low-pass Filter...")
        ideal_filtered, ideal_mask = self.ideal_lowpass_filter(image, optimal_cutoff)
        ideal_quality = self.calculate_image_quality_score(ideal_filtered)
        results["ideal_lowpass"] = {
            "image": ideal_filtered,
            "quality": ideal_quality,
            "name": f"Ideal Low-pass (f_c={optimal_cutoff})",
            "cutoff_frequency": optimal_cutoff,
            "mask": ideal_mask,
        }

        # Method 2: Butterworth Low-pass Filter
        print("  2. Butterworth Low-pass Filter...")
        butterworth_filtered, butterworth_mask = self.butterworth_lowpass_filter(
            image, optimal_cutoff, order=2
        )
        butterworth_quality = self.calculate_image_quality_score(butterworth_filtered)
        results["butterworth_lowpass"] = {
            "image": butterworth_filtered,
            "quality": butterworth_quality,
            "name": f"Butterworth Low-pass (f_c={optimal_cutoff}, n=2)",
            "cutoff_frequency": optimal_cutoff,
            "order": 2,
            "mask": butterworth_mask,
        }

        # Method 3: Gaussian Low-pass Filter
        print("  3. Gaussian Low-pass Filter...")
        gaussian_filtered, gaussian_mask = self.gaussian_lowpass_filter(
            image, optimal_cutoff
        )
        gaussian_quality = self.calculate_image_quality_score(gaussian_filtered)
        results["gaussian_lowpass"] = {
            "image": gaussian_filtered,
            "quality": gaussian_quality,
            "name": f"Gaussian Low-pass (σ={optimal_cutoff/2:.1f})",
            "cutoff_frequency": optimal_cutoff,
            "mask": gaussian_mask,
        }

        # Add frequency analysis to results
        results["frequency_analysis"] = freq_analysis

        return results

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

        # Save ideal low-pass filter result
        cv2.imwrite(
            f"{output_dir}/02_ideal_lowpass.jpg", results["ideal_lowpass"]["image"]
        )
        print(f"  Saved: 02_ideal_lowpass.jpg")

        # Save Butterworth low-pass filter result
        cv2.imwrite(
            f"{output_dir}/03_butterworth_lowpass.jpg",
            results["butterworth_lowpass"]["image"],
        )
        print(f"  Saved: 03_butterworth_lowpass.jpg")

        # Save Gaussian low-pass filter result
        cv2.imwrite(
            f"{output_dir}/04_gaussian_lowpass.jpg",
            results["gaussian_lowpass"]["image"],
        )
        print(f"  Saved: 04_gaussian_lowpass.jpg")

    def create_frequency_comparison_plot(self, results, output_dir):
        """
        Create a comprehensive frequency domain comparison plot

        Args:
            results (dict): Dictionary containing all enhancement results
            output_dir (str): Directory to save the comparison plot
        """
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))

        methods = [
            "original",
            "ideal_lowpass",
            "butterworth_lowpass",
            "gaussian_lowpass",
        ]
        colors = ["blue", "red", "green", "orange"]

        # Row 1: Display all filtered images
        for i, method in enumerate(methods):
            axes[0, i].imshow(results[method]["image"], cmap="gray")
            title = (
                f"{results[method]['name']}\nQuality: {results[method]['quality']:.1f}"
            )
            axes[0, i].set_title(title, fontsize=10)
            axes[0, i].axis("off")

        # Row 2: Display filter masks (frequency domain)
        for i, method in enumerate(methods):
            if method == "original":
                # Show magnitude spectrum for original
                if "frequency_analysis" in results:
                    magnitude_spectrum = results["frequency_analysis"][
                        "magnitude_spectrum"
                    ]
                    axes[1, i].imshow(magnitude_spectrum, cmap="gray")
                    axes[1, i].set_title("Original Spectrum")
                else:
                    axes[1, i].axis("off")
            else:
                # Show filter masks
                mask = results[method]["mask"]
                axes[1, i].imshow(mask, cmap="gray")
                axes[1, i].set_title(
                    f"{results[method]['name'].split('(')[0].strip()} Mask"
                )
            axes[1, i].axis("off")

        # Row 3: Histograms
        for i, method in enumerate(methods):
            axes[2, i].hist(
                results[method]["image"].ravel(), bins=256, alpha=0.7, color=colors[i]
            )
            axes[2, i].set_title(
                f"{results[method]['name'].split('(')[0].strip()} Histogram"
            )
            axes[2, i].set_xlabel("Gray Level")
            axes[2, i].set_ylabel("Pixel Count")

        # Row 4: Analysis plots
        # Quality comparison
        axes[3, 0].bar(
            range(len(methods)),
            [results[method]["quality"] for method in methods],
            color=colors,
        )
        axes[3, 0].set_title("Quality Score Comparison")
        axes[3, 0].set_ylabel("Quality Score")
        axes[3, 0].set_xticks(range(len(methods)))
        method_names = [
            results[method]["name"].split("(")[0].strip() for method in methods
        ]
        axes[3, 0].set_xticklabels(method_names, rotation=45, ha="right")

        # Overlaid histograms
        for i, method in enumerate(methods):
            axes[3, 1].hist(
                results[method]["image"].ravel(),
                bins=256,
                alpha=0.4,
                label=results[method]["name"].split("(")[0].strip(),
                color=colors[i],
            )
        axes[3, 1].set_title("All Histograms Overlaid")
        axes[3, 1].set_xlabel("Gray Level")
        axes[3, 1].set_ylabel("Pixel Count")
        axes[3, 1].legend()

        # Statistical comparison
        features_list = [
            self.analyze_image_features(results[method]["image"]) for method in methods
        ]

        # Mean comparison
        mean_values = [features["mean"] for features in features_list]
        axes[3, 2].bar(range(len(methods)), mean_values, color=colors)
        axes[3, 2].set_title("Mean Brightness Comparison")
        axes[3, 2].set_ylabel("Mean Value")
        axes[3, 2].set_xticks(range(len(methods)))
        axes[3, 2].set_xticklabels(
            [name[:8] for name in method_names], rotation=45, ha="right"
        )

        # Frequency energy distribution
        if "frequency_analysis" in results:
            freq_analysis = results["frequency_analysis"]
            energy_labels = ["Low Freq", "Mid Freq", "High Freq"]
            energy_values = [
                freq_analysis["low_freq_energy_ratio"],
                freq_analysis["mid_freq_energy_ratio"],
                freq_analysis["high_freq_energy_ratio"],
            ]
            axes[3, 3].pie(energy_values, labels=energy_labels, autopct="%1.1f%%")
            axes[3, 3].set_title("Original Image\nFrequency Energy Distribution")

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/frequency_domain_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print(f"  Saved: frequency_domain_comparison.png")

    def print_detailed_analysis(self, results):
        """
        Print detailed analysis of all frequency domain enhancement methods

        Args:
            results (dict): Dictionary containing all enhancement results
        """
        print("\n" + "=" * 80)
        print("FREQUENCY DOMAIN ENHANCEMENT ANALYSIS")
        print("=" * 80)

        # Print quality scores ranking
        methods_only = {k: v for k, v in results.items() if k != "frequency_analysis"}
        sorted_methods = sorted(
            methods_only.items(), key=lambda x: x[1]["quality"], reverse=True
        )

        print("\nQuality Score Ranking (Higher is Better):")
        print("-" * 50)
        for i, (method_key, data) in enumerate(sorted_methods, 1):
            if method_key == "original":
                improvement_symbol = "→"
                quality_change = 0
            else:
                improvement = data["quality"] > results["original"]["quality"]
                improvement_symbol = "↑" if improvement else "↓"
                quality_change = data["quality"] - results["original"]["quality"]

            print(
                f"{i}. {data['name']:<30} Quality: {data['quality']:.1f} "
                f"({improvement_symbol} {quality_change:+.1f})"
            )

        # Print frequency analysis
        if "frequency_analysis" in results:
            freq_analysis = results["frequency_analysis"]
            print(f"\nFrequency Content Analysis:")
            print("-" * 40)
            print(
                f"Low Frequency Energy:  {freq_analysis['low_freq_energy_ratio']:.1%}"
            )
            print(
                f"Mid Frequency Energy:  {freq_analysis['mid_freq_energy_ratio']:.1%}"
            )
            print(
                f"High Frequency Energy: {freq_analysis['high_freq_energy_ratio']:.1%}"
            )
            print(
                f"Estimated Optimal Cutoff: {freq_analysis['estimated_optimal_cutoff']} pixels"
            )
            print(f"Noise Level Estimate: {freq_analysis['noise_level']:.1%}")

        # Print ringing analysis
        print(f"\nRinging Effect Analysis:")
        print("-" * 40)
        print("• Ideal Low-pass Filter: HIGH ringing (sharp cutoff)")
        print("• Butterworth Low-pass Filter: MODERATE ringing (smooth transition)")
        print("• Gaussian Low-pass Filter: NO ringing (smoothest transition)")

        # Recommendations
        best_method = sorted_methods[0]
        print(f"\nRecommendations:")
        print("-" * 40)
        if best_method[0] == "original":
            print("The original image quality is already optimal.")
            print("Frequency domain filtering may not be necessary.")
        else:
            improvement = best_method[1]["quality"] - results["original"]["quality"]
            print(f"Best frequency domain method: {best_method[1]['name']}")
            print(f"Quality improvement: +{improvement:.1f} points")

            if "ideal" in best_method[0]:
                print("Note: Ideal filter may introduce ringing artifacts.")
            elif "butterworth" in best_method[0]:
                print("Note: Good balance between filtering and artifact reduction.")
            elif "gaussian" in best_method[0]:
                print("Note: Smoothest filtering with no ringing artifacts.")

    def process_single_image(self, image_path):
        """
        Process a single image with all frequency domain enhancement methods

        Args:
            image_path (str): Path to the input image file

        Returns:
            dict: Complete results dictionary with all enhanced images
        """
        # Load image in grayscale mode
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        self.original_image = image
        print(f"Processing image: {image_path}")

        # Apply all frequency domain methods
        results = self.apply_all_frequency_methods(image)

        # Save enhanced images
        self.save_all_enhanced_images_to_dir(results, self.output_dir)

        # Create comparison visualization
        self.create_frequency_comparison_plot(results, self.output_dir)

        # Print detailed analysis
        self.print_detailed_analysis(results)

        return results

    def batch_process_images(self, image_folder):
        """
        Process multiple images using frequency domain enhancement

        Args:
            image_folder (str): Path to folder containing images

        Returns:
            list: Results summary for all processed images
        """
        print(f"Starting frequency domain batch processing: {image_folder}")

        supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
        batch_results = []

        for filename in os.listdir(image_folder):
            if filename.lower().endswith(supported_formats):
                print(f"\n{'='*40}")
                print(f"Processing: {filename}")
                print("=" * 40)

                try:
                    image_path = os.path.join(image_folder, filename)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                    if image is None:
                        print(f"  Skipping: Cannot read {filename}")
                        continue

                    # Create subfolder for this image's results
                    base_name = os.path.splitext(filename)[0]
                    image_output_dir = f"{self.output_dir}/{base_name}"
                    if not os.path.exists(image_output_dir):
                        os.makedirs(image_output_dir)

                    # Process with frequency domain methods
                    results = self.apply_all_frequency_methods(image)
                    self.save_all_enhanced_images_to_dir(results, image_output_dir)
                    self.create_frequency_comparison_plot(results, image_output_dir)
                    self.print_detailed_analysis(results)

                    # Find best method (excluding original)
                    methods_only = {
                        k: v
                        for k, v in results.items()
                        if k not in ["original", "frequency_analysis"]
                    }
                    best_method = max(
                        methods_only.items(), key=lambda x: x[1]["quality"]
                    )

                    batch_results.append(
                        {
                            "filename": filename,
                            "best_method": best_method[1]["name"],
                            "quality_improvement": best_method[1]["quality"]
                            - results["original"]["quality"],
                            "original_quality": results["original"]["quality"],
                            "best_quality": best_method[1]["quality"],
                        }
                    )

                    print(f"  Best method: {best_method[1]['name']}")
                    print(
                        f"  Quality improvement: {best_method[1]['quality'] - results['original']['quality']:+.1f}"
                    )

                except Exception as e:
                    print(f"  Error processing {filename}: {e}")

        self.print_batch_summary(batch_results)
        return batch_results

    def print_batch_summary(self, batch_results):
        """
        Print summary of batch processing results
        """
        print(f"\n" + "=" * 80)
        print("FREQUENCY DOMAIN BATCH PROCESSING SUMMARY")
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

        # Method effectiveness statistics
        method_counts = {}
        for result in batch_results:
            method = result["best_method"].split("(")[0].strip()
            method_counts[method] = method_counts.get(method, 0) + 1

        print(f"\nMost effective frequency domain methods:")
        for method, count in sorted(
            method_counts.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {method}: {count} images ({count/len(batch_results)*100:.1f}%)")


def main():
    """
    Main function demonstrating the frequency domain enhancement system
    """
    print("Frequency Domain Image Enhancement System")
    print("=" * 60)
    print("This system applies frequency domain enhancement methods:")
    print("1. Ideal Low-pass Filter (sharp cutoff, may cause ringing)")
    print("2. Butterworth Low-pass Filter (smooth transition)")
    print("3. Gaussian Low-pass Filter (smoothest, no ringing)")
    print("=" * 60)

    try:
        # Create enhancement system
        enhancer = FrequencyDomainImageEnhancement()

        # Batch process all images in MV01 folder
        print("Starting batch processing of MV01 folder...")
        batch_results = enhancer.batch_process_images("MV01")

        print(f"\nBatch processing completed successfully!")
        print(f"Results saved in: {enhancer.output_dir}")
        print(f"\nFor each image, the following files were generated:")
        print(f"  - 01_original.jpg: Original image")
        print(f"  - 02_ideal_lowpass.jpg: Ideal low-pass filter result")
        print(f"  - 03_butterworth_lowpass.jpg: Butterworth low-pass filter result")
        print(f"  - 04_gaussian_lowpass.jpg: Gaussian low-pass filter result")
        print(f"  - frequency_domain_comparison.png: Comprehensive comparison")

        # Uncomment to process a single image instead
        # image_path = "MV01/filter_test01.jpg"
        # results = enhancer.process_single_image(image_path)

    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Verify the MV01 folder exists in the current directory")
        print("2. Ensure the folder contains readable image files")
        print("3. Check that required libraries are installed:")
        print("   pip install opencv-python numpy matplotlib")


if __name__ == "__main__":
    main()