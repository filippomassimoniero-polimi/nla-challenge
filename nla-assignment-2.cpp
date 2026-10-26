#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// COMPILE WITH `g++ -O2 -I${mkEigenInc} nla-assignment-2.cpp -o challenge2`
// RUN WITH `./challenge2`

MatrixXd compute_C(const MatrixXd &U, const int k) {
	MatrixXd C(U.rows(), k);

	for (int i = 0; i < k; ++i) {
		C.col(i) = U.col(i);
	}

	return C;
}

MatrixXd compute_D(const MatrixXd &V, const VectorXd &sigma, const int k) {
	MatrixXd D(V.rows(), k);

	for (int i = 0; i < k; ++i) {
		D.col(i) = sigma[i] * V.col(i);
	}

	return D;
}

MatrixXd compressed_matrix(const MatrixXd &C, const MatrixXd &D) {
	assert(C.cols() == D.cols());
	return C * D.transpose();
}

RowMatrixXd load_image(const char *path) {
	int desired_channels = 1;

	int x = 0;
	int y = 0;
	int channels = 0;

	auto *image = stbi_load(path, &x, &y, &channels, desired_channels);

	RowMatrixXd img(y, x);
	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			img(i, j) = static_cast<double>(image[i * x + j]);
		}
	}

	stbi_image_free(image);

	return img;
}

void save_image(const RowMatrixXd &img, const char *path) {
	Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> output_matrix(
		img.rows(), img.cols());

	output_matrix = img.unaryExpr([](double val) -> unsigned char {
		return static_cast<unsigned char>(std::clamp(val, 0.0, 255.0));
	});

	if (stbi_write_png(path, img.cols(), img.rows(), 1, output_matrix.data(), img.cols()) == 0) {
		std::cerr << "Error: Could not save grayscale image\n";
	}
}

int main() {
	// Task 1
	RowMatrixXd input_image = load_image("./artifacts/deer.jpg");

	RowMatrixXd simm_A = input_image.transpose() * input_image;
	std::cout << "- Task 1: norm of A^T A " << simm_A.norm() << "\n";

	// Task 2
	Eigen::SelfAdjointEigenSolver<MatrixXd> eigensolver(simm_A);
	VectorXd eigenvalues = eigensolver.eigenvalues();
	const auto n_eig = eigenvalues.size();

	std::cout << "- Task 2: The two biggest eigenvalues of A^T A are " << eigenvalues[n_eig - 1] << "\t" << eigenvalues[n_eig - 2] << std::endl;

	std::cout << "----------The two biggest singular values of A are " << std::sqrt(eigenvalues[n_eig - 1]) << "\t" << std::sqrt(eigenvalues[n_eig - 2]) << std::endl;

	// Task 3
	// mpirun -n 4 ./eigen1 simm_A.mtx eigvec.txt hist.txt -e pi -etol 1.0e-10
	// -> eigenvalue = 1.200286e+09 in 10 iterations
	Eigen::saveMarket(simm_A, "simm_A.mtx");

	// Task 4
	// Optimal shift for the power method: mu = (lambda_2 + lambda_n) / 2 ~ lambda_2 / 2 ~ 5.64e7
	// mpirun -n 4 ./eigen1 simm_A.mtx eigvec.txt hist.txt -e pi -etol 1.0e-10 -shift 5.64e7
	// -> 9 iterations (vs 10 without shift)

	// Task 5
	Eigen::BDCSVD<Eigen::MatrixXd> svd_original(input_image, Eigen::ComputeThinU | Eigen::ComputeThinV);
	VectorXd singular_values_original = svd_original.singularValues();

	std::cout << "- Task 5: Norm of the singular values: " << singular_values_original.norm() << std::endl;

	// Task 6
	MatrixXd c_30 = compute_C(svd_original.matrixU(), 30);
	MatrixXd d_30 = compute_D(svd_original.matrixV(), singular_values_original, 30);
	MatrixXd c_90 = compute_C(svd_original.matrixU(), 90);
	MatrixXd d_90 = compute_D(svd_original.matrixV(), singular_values_original, 90);

	std::cout << "- Task 6: k = 30 -> nonzeros of C: " << c_30.nonZeros() << ", nonzeros of D: " << d_30.nonZeros() << std::endl;
	std::cout << "- Task 6: k = 90 -> nonzeros of C: " << c_90.nonZeros() << ", nonzeros of D: " << d_90.nonZeros() << std::endl;

	// Task 7
	MatrixXd compressed_30 = compressed_matrix(c_30, d_30);
	MatrixXd compressed_90 = compressed_matrix(c_90, d_90);

	save_image(compressed_30, "compressed_30.png");
	save_image(compressed_90, "compressed_90.png");

	// Task 8
	const int board_size = 400;
	const int square_size = 40;
	RowMatrixXd checkerboard = RowMatrixXd::Zero(board_size, board_size);
	for (int i = 0; i < board_size; ++i) {
		for (int j = 0; j < board_size; ++j) {
			if ((i / square_size + j / square_size) % 2 == 1) {
				checkerboard(i, j) = 255.0;
			}
		}
	}
	save_image(checkerboard, "checkerboard.png");

	std::cout << "- Task 8: Norm of the checkerboard: " << checkerboard.norm() << std::endl;

	// Task 9
	RowMatrixXd noise = 60.0 * RowMatrixXd::Random(board_size, board_size);
	RowMatrixXd noisy_checkerboard = checkerboard + noise;
	save_image(noisy_checkerboard, "noisy_checkerboard.png");

	// Task 10
	Eigen::BDCSVD<Eigen::MatrixXd> svd_noisy(noisy_checkerboard, Eigen::ComputeThinU | Eigen::ComputeThinV);
	VectorXd singular_values_noisy = svd_noisy.singularValues();

	std::cout << "- Task 10: The two biggest singular values are " << singular_values_noisy[0] << "\t" << singular_values_noisy[1] << std::endl;

	// Task 11
	MatrixXd c_2 = compute_C(svd_noisy.matrixU(), 2);
	MatrixXd d_2 = compute_D(svd_noisy.matrixV(), singular_values_noisy, 2);
	MatrixXd c_6 = compute_C(svd_noisy.matrixU(), 6);
	MatrixXd d_6 = compute_D(svd_noisy.matrixV(), singular_values_noisy, 6);
	MatrixXd c_12 = compute_C(svd_noisy.matrixU(), 12);
	MatrixXd d_12 = compute_D(svd_noisy.matrixV(), singular_values_noisy, 12);

	std::cout << "- Task 11: k = 6 -> C is " << c_6.rows() << "x" << c_6.cols() << ", D is " << d_6.rows() << "x" << d_6.cols() << std::endl;
	std::cout << "- Task 11: k = 12 -> C is " << c_12.rows() << "x" << c_12.cols() << ", D is " << d_12.rows() << "x" << d_12.cols() << std::endl;

	// Task 12
	
	MatrixXd compressed_2 = compressed_matrix(c_2,d_2);
	MatrixXd compressed_6 = compressed_matrix(c_6, d_6);
	MatrixXd compressed_12 = compressed_matrix(c_12, d_12);

	save_image(compressed_2, "compressed_checkerboard_2.png");
	save_image(compressed_6, "compressed_checkerboard_6.png");
	save_image(compressed_12, "compressed_checkerboard_12.png");

	// Task 13 - Helper
	std::cout << "- Task 13: Error with the original checkerboard" << std::endl;
	std::cout << "\tnoisy: " << (noisy_checkerboard - checkerboard).norm() << std::endl;
	std::cout << "\tk = 2: " << (compressed_2 - checkerboard).norm() << std::endl;
	std::cout << "\tk = 6: " << (compressed_6 - checkerboard).norm() << std::endl;
	std::cout << "\tk = 12: " << (compressed_12 - checkerboard).norm() << std::endl;

	return 0;
}
