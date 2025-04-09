#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <openfhe.h>

using namespace lbcrypto;
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

// Custom random number generator (linear congruential generator)
unsigned int seed = 123456789;
unsigned int customRand() {
    seed = (1103515245 * seed + 12345) % 2147483647;
    return seed;
}

// Structure to represent a polynomial as a function of input dimensions
struct PolynomialCoefficients {
    // Coefficients for the polynomial: a + b*x + c*y + d*x^2 + e*xy + f*y^2 + ...
    std::vector<double> coefficients;
    std::vector<std::vector<int>> exponents; // Powers of each input dimension
};

// Generate PCA polynomial coefficients for a given number of samples and features
std::vector<std::vector<PolynomialCoefficients>> generatePCAPolynomials(int numSamples, int numFeatures, int numComponents) {
    if (numComponents > numFeatures) {
        numComponents = numFeatures;
    }

    // This will hold our result - polynomial coefficients for each output element
    // Each element of the output (i,j) will have its own polynomial
    std::vector<std::vector<PolynomialCoefficients>> polynomials(
            numSamples,
            std::vector<PolynomialCoefficients>(numComponents)
    );

    // Generate random eigenvectors to simulate PCA results
    // In real PCA, these would come from covariance matrix eigendecomposition
    std::vector<std::vector<double>> eigenvectors(numComponents, std::vector<double>(numFeatures, 0.0));

    for (int c = 0; c < numComponents; c++) {
        // Initialize random std::vector for this component
        std::vector<double> vector(numFeatures, 0.0);
        for (int i = 0; i < numFeatures; i++) {
            vector[i] = (double)customRand() / 2147483647;
        }

        // Normalize
        double norm = 0.0;
        for (int i = 0; i < numFeatures; i++) {
            norm += vector[i] * vector[i];
        }
        norm = polyApproxSqrt(norm);
        for (int i = 0; i < numFeatures; i++) {
            eigenvectors[c][i] = vector[i] / norm;
        }

        // Make orthogonal to previous std::vectors (Gram-Schmidt process)
        for (int prev = 0; prev < c; prev++) {
            // Calculate dot product
            double dotProduct = 0.0;
            for (int i = 0; i < numFeatures; i++) {
                dotProduct += eigenvectors[c][i] * eigenvectors[prev][i];
            }

            // Subtract projection
            for (int i = 0; i < numFeatures; i++) {
                eigenvectors[c][i] -= dotProduct * eigenvectors[prev][i];
            }
        }

        // Normalize again after orthogonalization
        norm = 0.0;
        for (int i = 0; i < numFeatures; i++) {
            norm += eigenvectors[c][i] * eigenvectors[c][i];
        }
        norm = polyApproxSqrt(norm);
        for (int i = 0; i < numFeatures; i++) {
            eigenvectors[c][i] /= norm;
        }
    }

    // For each sample and component, create a polynomial function
    for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++) {
        for (int componentIdx = 0; componentIdx < numComponents; componentIdx++) {
            // For PCA projection, the polynomial is simply a linear combination of input features
            PolynomialCoefficients poly;

            // Linear terms - one coefficient per input feature
            poly.coefficients.resize(numFeatures);
            poly.exponents.resize(numFeatures, std::vector<int>(numFeatures, 0));

            for (int featureIdx = 0; featureIdx < numFeatures; featureIdx++) {
                // The coefficient is just the eigenvector component
                poly.coefficients[featureIdx] = eigenvectors[componentIdx][featureIdx];

                // This term has exponent 1 for this feature and 0 for others
                poly.exponents[featureIdx][featureIdx] = 1;
            }

            polynomials[sampleIdx][componentIdx] = poly;
        }
    }

    return polynomials;
}

// Format polynomial coefficients as a string for display
std::string formatPolynomial(const PolynomialCoefficients& poly, int numFeatures) {
    std::string result;

    for (size_t i = 0; i < poly.coefficients.size(); i++) {
        double coef = poly.coefficients[i];
        if (customAbs(coef) < 1e-10) continue; // Skip near-zero terms

        if (!result.empty() && coef > 0) {
            result += " + ";
        } else if (!result.empty() && coef < 0) {
            result += " - ";
            coef = -coef;
        }

        if (result.empty() && coef < 0) {
            result += "-";
            coef = -coef;
        }

        // Add coefficient if not 1 (or if it's a constant term)
        bool allZero = true;
        for (int j = 0; j < numFeatures; j++) {
            if (poly.exponents[i][j] != 0) {
                allZero = false;
                break;
            }
        }

        if (customAbs(coef - 1.0) > 1e-10 || allZero) {
            result += std::to_string(coef);
        }

        // Add variables with their exponents
        for (int j = 0; j < numFeatures; j++) {
            int exp = poly.exponents[i][j];
            if (exp > 0) {
                result += "x" + std::to_string(j+1);
                if (exp > 1) {
                    result += "^" + std::to_string(exp);
                }
            }
        }
    }

    if (result.empty()) {
        result = "0";
    }

    return result;
}

// Function to parse a CSV file into integers
std::vector<std::vector<double>> parseCSVasDoubles(const std::string &filename, char delimiter = ',') {
    std::vector<std::vector<double>> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }

    std::string line;
    int lineNumber = 0;
    while (getline(file, line)) {
        ++lineNumber;
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;

        while (getline(ss, cell, delimiter)) {
            try {
                int value = stod(cell);
                row.push_back(value);
            } catch (const std::invalid_argument &e) {
                std::cerr << "Warning: Non-integer value '" << cell << "' on line " << lineNumber << " skipped." << std::endl;
            } catch (const std::out_of_range &e) {
                std::cerr << "Warning: Out-of-range integer value '" << cell << "' on line " << lineNumber << " skipped." << std::endl;
            }
        }

        if (!row.empty()) {
            data.push_back(row);
        }
    }

    file.close();
    return data;
}

// Format polynomial coefficients as a string for display
std::string calculatePolynomial(const PolynomialCoefficients& poly, int numFeatures, std::vector<double> data) {
    std::string result;
    double final;

    for (size_t i = 0; i < poly.coefficients.size(); i++) {
        double coef = poly.coefficients[i];
        coef = coef*data[i];
        final += coef;
    }

    return result+= std::to_string(final);
}

// Format polynomial coefficients as a string for display
std::string calculatePolynomial(const PolynomialCoefficients& poly, int numFeatures, std::vector<Ciphertext<DCRTPoly>> data, CryptoContext<DCRTPoly> cc, KeyPair<DCRTPoly> keyPair) {
    std::string result;
    double final;
    Ciphertext<DCRTPoly> res;

    for (size_t i = 0; i < 16; i++) {
        cc->EvalMultInPlace(data[i], poly.coefficients[i]);
        if (i == 0) res = data[i];
        else res += data[i];
    }
    Plaintext plain;
    cc->Decrypt(keyPair.secretKey, res, &plain);
    auto finvec = plain->GetCKKSPackedValue();
    //std::cout << "Full value " << finvec << std::endl;
    final = finvec[0].real();

    return result+= std::to_string(final);
}


int main(){
    /**
    if (argc < 4) {
        std::std::cerr << "Usage: " << argv[0] << " <num_samples> <num_features> <num_components>" << std::endl;
        return 1;
    }**/

    // Parse command line arguments
    int numSamples = 200;
    int numFeatures = 16;
    int numComponents = 6;

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
    }**/

    // Generate polynomials
    std::vector<std::vector<PolynomialCoefficients>> polynomials =
            generatePCAPolynomials(numSamples, numFeatures, numComponents);

    std::vector<std::vector<double>> inputData = parseCSVasDoubles("../data_pca_200x16.csv");

    //Encode Data
    uint32_t multDepth = 1;
    uint32_t scaleModSize = 50;
    uint32_t batchSize = 32;
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << std::endl << std::endl;

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalRotateKeyGen(keys.secretKey, {1, -2});

    std::vector<std::vector<Ciphertext<DCRTPoly>>> encryptedData;
    for (int i =  0; i < inputData.size(); i++){
        std::vector<Ciphertext<DCRTPoly>> row;
        for (int j = 0; j < inputData[0].size(); j++){
            std::vector<double> ct(32,0);
            ct[0] = inputData[i][j];
            Plaintext pt = cc->MakeCKKSPackedPlaintext(ct);
            auto add = cc->Encrypt(keys.publicKey, pt);
            row.push_back(add);
        }
        std::cout << "Making a plaintext " << i << std::endl;
        encryptedData.push_back(row);
    }


    // Display results
    std::cout << "Generated PCA polynomial coefficients for "
              << numSamples << " samples, "
              << numFeatures << " features, reduced to "
              << numComponents << " components:" << std::endl;

    // Output the first few sample polynomials as example
    int maxSamplesToShow = std::min(200, numSamples);
    int maxComponentsToShow = std::min(6, numComponents);

    for (int i = 0; i < maxSamplesToShow; i++) {
        std::cout << "Sample " << (i+1) << ":" << std::endl;
        for (int j = 0; j < maxComponentsToShow; j++) {
            std::cout << "  Component " << (j+1) << ": "
                      << formatPolynomial(polynomials[i][j], numFeatures) << " = "
                      << calculatePolynomial(polynomials[i][j], numFeatures, inputData[i]) << " "
                      << calculatePolynomial(polynomials[i][j], numFeatures, encryptedData[i], cc, keys)
            << std::endl;
        }
    }

    if (numSamples > maxSamplesToShow || numComponents > maxComponentsToShow) {
        std::cout << "... (showing " << maxSamplesToShow << " samples out of " << numSamples
                  << " and " << maxComponentsToShow << " components out of " << numComponents << ")" << std::endl;
    }

    return 0;
}