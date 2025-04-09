#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

// Custom implementation of square root using Newton's method
double customSqrt(double x) {
    if (x <= 0) return 0;
    double guess = x / 2.0;
    double epsilon = 1e-10;

    while (true) {
        double newGuess = 0.5 * (guess + x / guess);
        if ((guess - newGuess) * (guess - newGuess) < epsilon)
            return newGuess;
        guess = newGuess;
    }
}

// Custom implementation of absolute value
double customAbs(double x) {
    return x < 0 ? -x : x;
}

// Custom random number generator (linear congruential generator)
unsigned int seed = 123456789;
unsigned int customRand() {
    seed = (1103515245 * seed + 12345) % 2147483647;
    return seed;
}

// Custom division of double by integer
double customDiv(double a, int b) {
    return a / b;
}

std::vector<std::vector<double>> performPCA(const std::vector<std::vector<double>>& data, int numComponents) {
    // Get dimensions
    int numSamples = data.size();
    int numFeatures = data[0].size();

    if (numComponents > numFeatures) {
        numComponents = numFeatures;
    }

    // Step 1: Compute the mean of each feature
    std::vector<double> means(numFeatures, 0.0);
    for (int i = 0; i < numSamples; i++) {
        for (int j = 0; j < numFeatures; j++) {
            means[j] += data[i][j];
        }
    }
    for (int j = 0; j < numFeatures; j++) {
        means[j] = customDiv(means[j], numSamples);
    }

    // Step 2: Center the data (subtract mean)
    std::vector<std::vector<double>> centeredData = data;
    for (int i = 0; i < numSamples; i++) {
        for (int j = 0; j < numFeatures; j++) {
            centeredData[i][j] -= means[j];
        }
    }

    // Step 3: Compute the covariance matrix
    std::vector<std::vector<double>> covMatrix(numFeatures, std::vector<double>(numFeatures, 0.0));
    for (int i = 0; i < numFeatures; i++) {
        for (int j = 0; j < numFeatures; j++) {
            for (int k = 0; k < numSamples; k++) {
                covMatrix[i][j] += centeredData[k][i] * centeredData[k][j];
            }
            covMatrix[i][j] = customDiv(covMatrix[i][j], (numSamples - 1));
        }
    }

    // Step 4: Power iteration method to find eigenvectors and eigenvalues
    std::vector<std::vector<double>> eigenvectors(numComponents, std::vector<double>(numFeatures, 0.0));
    std::vector<double> eigenvalues(numComponents, 0.0);

    // Temporary matrix to deflate the covariance matrix
    std::vector<std::vector<double>> tempCovMatrix = covMatrix;

    for (int c = 0; c < numComponents; c++) {
        // Initialize random vector
        std::vector<double> vector(numFeatures, 0.0);
        for (int i = 0; i < numFeatures; i++) {
            vector[i] = customDiv((double)customRand(), 2147483647);
        }

        // Normalize
        double norm = (0.0);
        for (int i = 0; i < numFeatures; i++) {
            norm += vector[i] * vector[i];
        }
        norm = customSqrt(norm);
        for (int i = 0; i < numFeatures; i++) {
            vector[i] /= norm;
        }

        // Power iteration
        for (int iter = 0; iter < 100; iter++) {
            // v = Av
            std::vector<double> newVector(numFeatures, 0.0);
            for (int i = 0; i < numFeatures; i++) {
                for (int j = 0; j < numFeatures; j++) {
                    newVector[i] += tempCovMatrix[i][j] * vector[j];
                }
            }

            // Calculate eigenvalue (Rayleigh quotient)
            double eigenvalue = 0.0;
            for (int i = 0; i < numFeatures; i++) {
                eigenvalue += vector[i] * newVector[i];
            }

            // Normalize new vector
            norm = 0.0;
            for (int i = 0; i < numFeatures; i++) {
                norm += newVector[i] * newVector[i];
            }
            norm = customSqrt(norm);

            // Check for convergence
            bool converged = true;
            for (int i = 0; i < numFeatures; i++) {
                double newValue = newVector[i] / norm;
                if (customAbs(customAbs(newValue) - customAbs(vector[i])) > 1e-10) {
                    converged = false;
                    break;
                }
            }

            // Update vector
            for (int i = 0; i < numFeatures; i++) {
                vector[i] = newVector[i] / norm;
            }

            if (converged) {
                break;
            }
        }

        // Store eigenvector and eigenvalue
        eigenvalues[c] = 0.0;
        for (int i = 0; i < numFeatures; i++) {
            for (int j = 0; j < numFeatures; j++) {
                eigenvalues[c] += vector[i] * tempCovMatrix[i][j] * vector[j];
            }
        }

        for (int i = 0; i < numFeatures; i++) {
            eigenvectors[c][i] = vector[i];
        }

        // Deflate the covariance matrix (subtract the component)
        for (int i = 0; i < numFeatures; i++) {
            for (int j = 0; j < numFeatures; j++) {
                tempCovMatrix[i][j] -= eigenvalues[c] * vector[i] * vector[j];
            }
        }
    }

    // Step 5: Project the data onto the principal components
    std::vector<std::vector<double>> projectedData(numSamples, std::vector<double>(numComponents, 0.0));
    for (int i = 0; i < numSamples; i++) {
        for (int c = 0; c < numComponents; c++) {
            for (int j = 0; j < numFeatures; j++) {
                projectedData[i][c] += centeredData[i][j] * eigenvectors[c][j];
            }
        }
    }

    return projectedData;
}

int main(int argc, char* argv[]) {
//if (argc < 3) {
//    std::cerr << "Usage: " << argv[0] << " <filename> <num_components>" << std::endl;
//    return 1;
//}


// Parse command line arguments
    std::string filename = "../data_pca_200x16.csv";
    int numComponents = 6;


    // Read data from file
    std::vector<std::vector<double>> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return 1;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string value;

        // Parse comma-separated values
        while (std::getline(ss, value, ';')) {
            try {
                // Custom string to double conversion
                double val = 0.0;
                bool negative = false;
                int decimalPos = -1;

                for (int i = 0; i < value.length(); i++) {
                    if (value[i] == '-' && i == 0) {
                        negative = true;
                    }
                    else if (value[i] == ',') {
                        decimalPos = i;
                    }
                    else if (value[i] >= '0' && value[i] <= '9') {
                        val = val * 10 + (value[i] - '0');
                        if (decimalPos >= 0) {
                            decimalPos++;
                        }
                    }
                }

                if (decimalPos > 0) {
                    for (int i = 0; i < (decimalPos - 1); i++) {
                        val /= 10.0;
                    }
                }

                if (negative) {
                    val = -val;
                }

                row.push_back(val);
            } catch (...) {
                std::cerr << "Error parsing value: " << value << std::endl;
                continue;
            }
        }

        if (!row.empty()) {
            data.push_back(row);
        }
    }

    file.close();

    if (data.empty()) {
        std::cerr << "Error: No data read from file" << std::endl;
        return 1;
    }

    // Display data dimensions
    std::cout << "Read " << data.size() << " samples with " << data[0].size() << " features each." << std::endl;

    // Perform PCA
    std::vector<std::vector<double>> projectedData = performPCA(data, numComponents);

    // Output results
    std::cout << "Projected data (" << projectedData.size() << " samples x " << projectedData[0].size() << " components):" << std::endl;

    for (size_t i = 0; i < projectedData.size(); i++) {
        for (size_t j = 0; j < projectedData[i].size(); j++) {
            std::cout << projectedData[i][j];
            if (j < projectedData[i].size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << std::endl;
    }

    return 0;
}