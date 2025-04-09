#include <vector>
#include <iostream>
#include <string>

// Polynomial approximation for square root without range adjustment
double polyApproxSqrt(double x) {
    // Handle special cases
    if (x <= 0) return 0;
    if (x == 1) return 1;

    // Initial approximation: a0 + a1*x + a2*x^2 + a3*x^3
    double a0 = 0.1215;
    double a1 = 0.9318;
    double a2 = -0.0942;
    double a3 = 0.0409;

    // Calculate numerator using Horner's method
    double num = a3;
    num = num * x + a2;
    num = num * x + a1;
    num = num * x + a0;

    // For very large numbers, improve accuracy with a single Newton-Raphson step
    if (x > 10000) {
        return 0.5 * (num + x/num);
    }

    return num;
}

// Custom implementation of absolute value
double customAbs(double x) {
    return x < 0 ? -x : x;
}

// Custom implementation of exponential function using polynomial approximation
double customExp(double x) {
    // Handle edge cases
    if (x > 700) return 1e308; // Prevent overflow
    if (x < -700) return 0;    // Prevent underflow

    // For exp(-x) where x is positive (common case in RBF kernel)
    // we'll use a rational polynomial approximation for better accuracy
    if (x <= 0) {
        // Range reduction: exp(x) = exp(n + f) = exp(n) * exp(f)
        // where n is an integer and f is in [0, 1)
        int n = (int)x;
        double f = x - n;

        // Polynomial approximation for exp(f) where f is in [0, 1)
        // P(f) = 1 + f * (1 + f * (1/2 + f * (1/6 + f * (1/24 + f/120))))
        double p = 1.0 + f * (1.0 + f * (0.5 + f * (1.0/6.0 + f * (1.0/24.0 + f/120.0))));

        // Scale by 2^n (shift binary exponent)
        // This works because exp(n) = 2^(n/ln(2))
        double result = p;
        for (int i = 0; i > n; i--) {
            result *= 0.36787944117; // approximation of 1/e
        }

        return result;
    } else {
        // For positive x, use a rational polynomial approximation
        // exp(x) ≈ 1 + 2x/(2-x) for small x
        // Range reduction for larger values

        // Break x into integer and fractional part
        int n = (int)x;
        double f = x - n;

        // Rational approximation for exp(f) where f is in [0, 1)
        double num = 2.0 * f;
        double den = 2.0 - f;
        double p = 1.0 + num/den;

        // Scale by e^n
        double result = p;
        for (int i = 0; i < n; i++) {
            result *= 2.718281828459; // approximation of e
        }

        return result;
    }
}

// Custom random number generator (linear congruential generator)
unsigned int seed = 123456789;
unsigned int customRand() {
    seed = (1103515245 * seed + 12345) % 2147483647;
    return seed;
}

// Structure to represent a kernel-based polynomial
struct KernelPolynomial {
    int kernelType;                    // 0: Linear, 1: Polynomial, 2: RBF
    std::vector<double> sampleCoeffs;  // Coefficients for each sample point
    std::vector<std::vector<double>> supportVectors; // For kernel evaluation
    double gamma;                      // For RBF kernel
    double coef0;                      // For polynomial kernel
    int degree;                        // For polynomial kernel
};

// Kernel functions
double linearKernel(const std::vector<double>& x, const std::vector<double>& y) {
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); i++) {
        sum += x[i] * y[i];
    }
    return sum;
}

double polynomialKernel(const std::vector<double>& x, const std::vector<double>& y, double gamma, double coef0, int degree) {
    double dot = 0.0;
    for (size_t i = 0; i < x.size(); i++) {
        dot += x[i] * y[i];
    }
    double result = gamma * dot + coef0;

    // Compute power using repeated multiplication
    double power = 1.0;
    for (int i = 0; i < degree; i++) {
        power *= result;
    }
    return power;
}

// Polynomial approximation of RBF kernel
double rbfKernel(const std::vector<double>& x, const std::vector<double>& y, double gamma) {
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); i++) {
        double diff = x[i] - y[i];
        sum += diff * diff;
    }

    // Calculate exp(-gamma * sum) using our polynomial approximation
    return customExp(-gamma * sum);
}

// Generate sample data points
std::vector<std::vector<double>> generateSampleData(int numSamples, int numFeatures) {
    std::vector<std::vector<double>> data(numSamples, std::vector<double>(numFeatures, 0.0));

    for (int i = 0; i < numSamples; i++) {
        for (int j = 0; j < numFeatures; j++) {
            data[i][j] = (double)customRand() / 2147483647 * 2.0 - 1.0; // Range -1 to 1
        }
    }

    return data;
}

// Calculate kernel matrix
std::vector<std::vector<double>> calculateKernelMatrix(
        const std::vector<std::vector<double>>& data,
        int kernelType,
        double gamma = 1.0,
        double coef0 = 0.0,
        int degree = 3) {

    int numSamples = data.size();
    std::vector<std::vector<double>> kernelMatrix(numSamples, std::vector<double>(numSamples, 0.0));

    for (int i = 0; i < numSamples; i++) {
        for (int j = 0; j <= i; j++) { // Take advantage of symmetry
            double k;
            if (kernelType == 0) { // Linear
                k = linearKernel(data[i], data[j]);
            } else if (kernelType == 1) { // Polynomial
                k = polynomialKernel(data[i], data[j], gamma, coef0, degree);
            } else { // RBF
                k = rbfKernel(data[i], data[j], gamma);
            }
            kernelMatrix[i][j] = k;
            kernelMatrix[j][i] = k; // Symmetry
        }
    }

    return kernelMatrix;
}

// Center the kernel matrix
void centerKernelMatrix(std::vector<std::vector<double>>& kernelMatrix) {
    int n = kernelMatrix.size();
    std::vector<double> rowMeans(n, 0.0);
    double totalMean = 0.0;

    // Calculate row means
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            rowMeans[i] += kernelMatrix[i][j];
        }
        rowMeans[i] /= n;
        totalMean += rowMeans[i];
    }
    totalMean /= n;

    // Center the kernel matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            kernelMatrix[i][j] = kernelMatrix[i][j] - rowMeans[i] - rowMeans[j] + totalMean;
        }
    }
}

// Perform Kernel PCA
std::vector<KernelPolynomial> performKernelPCA(
        const std::vector<std::vector<double>>& data,
        int numComponents,
        int kernelType,
        double gamma = 1.0,
        double coef0 = 0.0,
        int degree = 3) {

    int numSamples = data.size();

    // Calculate kernel matrix
    std::vector<std::vector<double>> kernelMatrix =
            calculateKernelMatrix(data, kernelType, gamma, coef0, degree);

    // Center the kernel matrix
    centerKernelMatrix(kernelMatrix);

    // Power iteration to find eigenvectors of the kernel matrix
    std::vector<std::vector<double>> eigenvectors(numComponents, std::vector<double>(numSamples, 0.0));
    std::vector<double> eigenvalues(numComponents, 0.0);

    // Temporary matrix for deflation
    std::vector<std::vector<double>> tempMatrix = kernelMatrix;

    for (int c = 0; c < numComponents; c++) {
        // Initialize random vector
        std::vector<double> vector(numSamples, 0.0);
        for (int i = 0; i < numSamples; i++) {
            vector[i] = (double)customRand() / 2147483647;
        }

        // Normalize
        double norm = 0.0;
        for (int i = 0; i < numSamples; i++) {
            norm += vector[i] * vector[i];
        }
        norm = polyApproxSqrt(norm);
        for (int i = 0; i < numSamples; i++) {
            vector[i] /= norm;
        }

        // Power iteration
        for (int iter = 0; iter < 100; iter++) {
            std::vector<double> newVector(numSamples, 0.0);

            // v = Kv
            for (int i = 0; i < numSamples; i++) {
                for (int j = 0; j < numSamples; j++) {
                    newVector[i] += tempMatrix[i][j] * vector[j];
                }
            }

            // Calculate eigenvalue
            double eigenvalue = 0.0;
            for (int i = 0; i < numSamples; i++) {
                eigenvalue += vector[i] * newVector[i];
            }

            // Normalize
            norm = 0.0;
            for (int i = 0; i < numSamples; i++) {
                norm += newVector[i] * newVector[i];
            }
            norm = polyApproxSqrt(norm);

            // Update vector
            for (int i = 0; i < numSamples; i++) {
                vector[i] = newVector[i] / norm;
            }
        }

        // Calculate eigenvalue using Rayleigh quotient
        eigenvalues[c] = 0.0;
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numSamples; j++) {
                eigenvalues[c] += vector[i] * tempMatrix[i][j] * vector[j];
            }
        }

        // Store eigenvector
        for (int i = 0; i < numSamples; i++) {
            eigenvectors[c][i] = vector[i];
        }

        // Deflate the matrix
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numSamples; j++) {
                tempMatrix[i][j] -= eigenvalues[c] * vector[i] * vector[j];
            }
        }
    }

    // Normalize eigenvectors by eigenvalues
    for (int c = 0; c < numComponents; c++) {
        double factor = polyApproxSqrt(eigenvalues[c]);
        for (int i = 0; i < numSamples; i++) {
            eigenvectors[c][i] /= factor;
        }
    }

    // Create kernel polynomials
    std::vector<KernelPolynomial> kernelPolynomials(numComponents);

    for (int c = 0; c < numComponents; c++) {
        kernelPolynomials[c].kernelType = kernelType;
        kernelPolynomials[c].sampleCoeffs = eigenvectors[c];
        kernelPolynomials[c].supportVectors = data;
        kernelPolynomials[c].gamma = gamma;
        kernelPolynomials[c].coef0 = coef0;
        kernelPolynomials[c].degree = degree;
    }

    return kernelPolynomials;
}

// Format polynomial representation of kernel projection
std::string formatKernelPolynomial(const KernelPolynomial& poly, int componentIndex) {
    std::string result = "PC" + std::to_string(componentIndex+1) + "(x) = ";

    // Different descriptions based on kernel type
    if (poly.kernelType == 0) {
        result += "Linear kernel projection";
    } else if (poly.kernelType == 1) {
        result += "Polynomial kernel (degree " + std::to_string(poly.degree) +
                  ", gamma=" + std::to_string(poly.gamma) +
                  ", coef0=" + std::to_string(poly.coef0) + ") projection";
    } else {
        result += "RBF kernel (gamma=" + std::to_string(poly.gamma) + ") projection";
    }

    result += " using " + std::to_string(poly.supportVectors.size()) + " support vectors";

    return result;
}

// Project a new data point using kernel PCA
double projectPoint(const std::vector<double>& point, const KernelPolynomial& poly, int componentIndex) {
    double projection = 0.0;
    int numSamples = poly.supportVectors.size();

    for (int i = 0; i < numSamples; i++) {
        double k;
        if (poly.kernelType == 0) { // Linear
            k = linearKernel(point, poly.supportVectors[i]);
        } else if (poly.kernelType == 1) { // Polynomial
            k = polynomialKernel(point, poly.supportVectors[i], poly.gamma, poly.coef0, poly.degree);
        } else { // RBF
            k = rbfKernel(point, poly.supportVectors[i], poly.gamma);
        }

        projection += poly.sampleCoeffs[i] * k;
    }

    return projection;
}

int main(int argc, char* argv[]) {
    /**
   if (argc < 5) {
       std::cerr << "Usage: " << argv[0] << " <num_samples> <num_features> <num_components> <kernel_type>" << std::endl;
       std::cerr << "Kernel types: 0=Linear, 1=Polynomial, 2=RBF" << std::endl;
       return 1;
   }**/

// Parse command line arguments
    int numSamples = 200;
    int numFeatures = 16;
    int numComponents = 6;
    int kernelType = 2;

/**
// Custom string to int conversion
for (int i = 0; argv[1][i] != '\0'; ++i) {
    numSamples = numSamples * 10 + (argv[1][i] - '0');
}

for (int i = 0; argv[2][i] != '\0'; ++i) {
    numFeatures = numFeatures * 10 + (argv[2][i] - '0');
}

for (int i = 0; argv[3][i] != '\0'; ++i) {
    numComponents = numComponents * 10 + (argv[3][i] - '0');
}

for (int i = 0; argv[4][i] != '\0'; ++i) {
    kernelType = kernelType * 10 + (argv[4][i] - '0');
}

if (kernelType < 0 || kernelType > 2) {
    std::cerr << "Invalid kernel type. Use 0=Linear, 1=Polynomial, 2=RBF" << std::endl;
    return 1;
}**/


    // Generate sample data
    std::vector<std::vector<double>> data = generateSampleData(numSamples, numFeatures);

    // Set kernel parameters
    double gamma = 1.0 / numFeatures; // Default scaling
    double coef0 = 1.0;               // Default for polynomial kernel
    int degree = 3;                   // Default for polynomial kernel

    // Perform kernel PCA
    std::vector<KernelPolynomial> kernelPolynomials =
            performKernelPCA(data, numComponents, kernelType, gamma, coef0, degree);

    // Display results
    std::cout << "Kernel PCA with "
              << numSamples << " samples, "
              << numFeatures << " features, reduced to "
              << numComponents << " components" << std::endl;

    std::cout << "Kernel type: ";
    if (kernelType == 0) {
        std::cout << "Linear" << std::endl;
    } else if (kernelType == 1) {
        std::cout << "Polynomial (degree=" << degree << ", gamma=" << gamma << ", coef0=" << coef0 << ")" << std::endl;
    } else {
        std::cout << "RBF (gamma=" << gamma << ")" << std::endl;
    }

    // Show polynomial representations
    for (int i = 0; i < numComponents; i++) {
        std::cout << formatKernelPolynomial(kernelPolynomials[i], i) << std::endl;
    }

    // Show sample projections for the first few data points
    int maxSamplesToShow = std::min(3, numSamples);
    for (int i = 0; i < maxSamplesToShow; i++) {
        std::cout << "Sample " << (i+1) << " projections:" << std::endl;
        for (int j = 0; j < numComponents; j++) {
            double proj = projectPoint(data[i], kernelPolynomials[j], j);
            std::cout << "  PC" << (j+1) << ": " << proj << std::endl;
        }
    }

    // Generate a new test point and project it
    std::vector<double> testPoint(numFeatures);
    for (int i = 0; i < numFeatures; i++) {
        testPoint[i] = (double)customRand() / 2147483647 * 2.0 - 1.0;
    }

    std::cout << "New test point projections:" << std::endl;
    for (int j = 0; j < numComponents; j++) {
        double proj = projectPoint(testPoint, kernelPolynomials[j], j);
        std::cout << "  PC" << (j+1) << ": " << proj << std::endl;
    }

    return 0;
}