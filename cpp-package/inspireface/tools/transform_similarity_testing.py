import numpy as np
import matplotlib.pyplot as plt


class SimilarityConverter:
    def __init__(self, 
                 threshold=0.48, 
                 middle_score=0.6, 
                 steepness=8.0, 
                 output_range=(0.01, 0.99)):
        self.threshold = threshold
        self.middle_score = middle_score
        self.steepness = steepness
        self.output_min, self.output_max = output_range
        self.output_scale = self.output_max - self.output_min
        
        self.bias = -np.log((self.output_max - self.middle_score)/(self.middle_score - self.output_min))
        
    def convert(self, cosine):
        shifted_input = self.steepness * (cosine - self.threshold)
        sigmoid = 1 / (1 + np.exp(-shifted_input - self.bias))
        return sigmoid * self.output_scale + self.output_min
    
    def plot(self, test_points=None):
        x = np.linspace(-1, 1, 200)
        y = self.convert(x)
        
        plt.figure(figsize=(12, 6))
        plt.plot(x, y, '-', label='Conversion Function', color='blue')
        
        if test_points is not None:
            test_scores = self.convert(test_points)
            plt.plot(test_points, test_scores, 'ro', label='Test Points')
            
            for point, score in zip(test_points, test_scores):
                plt.annotate(f'({point:.2f}, {score:.3f})', 
                            xy=(point, score), 
                            xytext=(10, 10),
                            textcoords='offset points')
        
        plt.axvline(x=self.threshold, color='gray', linestyle='--', alpha=0.5, 
                   label=f'Threshold ({self.threshold:.2f})')
        plt.axhline(y=self.middle_score, color='gray', linestyle='--', alpha=0.5)
        
        plt.plot(self.threshold, self.middle_score, 'r*', markersize=15, 
                label='Passing Point', color='red', zorder=5)
        
        plt.annotate(f'Passing Point\n({self.threshold:.2f}, {self.middle_score:.2f})',
                    xy=(self.threshold, self.middle_score),
                    xytext=(30, 20),
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', 
                                  color='red', lw=2))
        
        
        plt.fill_between(x, y, 0, 
                        where=(x >= self.threshold),
                        alpha=0.1, color='green',
                        label='Passing Region')
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Converted Score')
        plt.title('Similarity Score Conversion')
        plt.legend()
        plt.xlim(-1.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Set the threshold to 0.42
    converter1 = SimilarityConverter(
        threshold=0.42,
        middle_score=0.6,
        steepness=8.0,
        output_range=(0.01, 1.0)
    )
    test_points1 = np.array([-0.8, -0.2, 0.02, 0.1, 0.25, 0.3, 0.48, 0.7, 0.8, 0.9, 1.0])
    print("\nModel 1 (Threshold = 0.48):")
    converter1.plot(test_points1)
    
    # Set the threshold to 0.32
    converter2 = SimilarityConverter(
        threshold=0.32,
        middle_score=0.6,
        steepness=10.0,
        output_range=(0.02, 1.0)
    )
    test_points2 = np.array([-0.8, -0.2, 0.02, 0.1, 0.25,0.32, 0.5, 0.7, 0.8, 0.9, 1.0])
    print("\nModel 2 (Threshold = 0.32):")
    converter2.plot(test_points2)
    
    # Print the results
    print("\nTest Results for Model 1 (threshold=0.48):")
    for point in test_points1:
        score = converter1.convert(point)
        print(f"Cosine: {point:6.2f} -> Score: {score:.4f}")
        
    print("\nTest Results for Model 2 (threshold=0.32):")
    for point in test_points2:
        score = converter2.convert(point)
        print(f"Cosine: {point:6.2f} -> Score: {score:.4f}")