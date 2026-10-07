#include <algorithm>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <tuple>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

#include <lis.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>; 

// COMPILE WITH `mpicxx -DUSE_MPI -I${mkLisInc} -I${mkEigenInc} nla-assignment-1.cpp -o challenge -L${mkLisLib} -llis -O2`
// RUN WITH `mpirun -n 1 challenge` or `./challenge`
RowMatrixXd load_image(const char *path) { 
	int desired_channels = 1;

	int x = 0;
	int y = 0;
	int channels = 0;

	auto *image = stbi_load(path, &x, &y, &channels, desired_channels);

	RowMatrixXd img(y, x); 
	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			int index = (i * x + j) * channels;
			img(i, j) = static_cast<double>(image[index]); 
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

	if (stbi_write_png(path, img.cols(), img.rows(), 1, output_matrix.data(), 
		img.cols()) == 0) { 
		std::cerr << "Error: Could not save grayscale image\n";
	}
}

void save_image_vec(const VectorXd &img, const char *path,
	Eigen::Index stride) {
	Eigen::Matrix<unsigned char, Eigen::Dynamic, 1> output_data =
		img.unaryExpr([](double val) -> unsigned char {
			return static_cast<unsigned char>(std::clamp(val, 0.0, 255.0));
		});

	if (stbi_write_png(path, stride, img.size() / stride, 1, output_data.data(),
		stride) == 0) {
		std::cerr << "Error: Could not save grayscale image\n";
	}
}

Eigen::SparseMatrix<double> build_convolution_matrix(const MatrixXd &filter, Eigen::Index size, Eigen::Index stride) {
	auto A = Eigen::SparseMatrix<double>(size, size);
	A.reserve(Eigen::VectorXi::Constant(size, (filter.array() != 0).count()));

	auto row_mid = filter.rows() / 2;
	auto col_mid = filter.cols() / 2;

	Eigen::Index cols = stride;
	Eigen::Index rows = size / stride;

	for (Eigen::Index k = 0; k < size; k++) {
		for (auto i = -row_mid; i <= row_mid; i++) {
			for (auto j = -col_mid; j <= col_mid; j++) {

				Eigen::Index nr = k / cols + i; 
				Eigen::Index nc = k % cols + j; 

				if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) 
					continue;

				double tmp = filter(i + row_mid, j + col_mid);
				if (tmp != 0.0)
					A.insert(k, k + (i * stride) + j) = tmp; 
			}
		}
	}

	return A;
}

// Returns <iteration_count, residual>. Options is passed to lis_solver_set_option, for example "-i bicgstab -p ilu".
std::pair<int,double> solve_lis_file_output(const char *matrix_path, const char *b_path, const char *solution_path, char *options, double tol) {
	// Bad workaround to avoid mixing argc/argv, should work for this challenge
	int argc = 1;
	char program_name[] = "nla";
	char *argv_data[] = {program_name, nullptr};
	char **argv = argv_data;
	lis_initialize(&argc, &argv);

	LIS_MATRIX A;
	LIS_VECTOR b, x;
	LIS_SOLVER solver;

	lis_matrix_create(LIS_COMM_WORLD, &A);
	if (lis_input_matrix(A, const_cast<char *>(matrix_path)) != LIS_SUCCESS)
		std::cerr << "LIS: Could not read matrix " << matrix_path << std::endl;

	lis_vector_create(LIS_COMM_WORLD, &b);
	if (lis_input_vector(b, const_cast<char *>(b_path)) != LIS_SUCCESS)
		std::cerr << "LIS: Could not read vector " << b_path << std::endl;

	lis_vector_duplicate(A, &x);

	lis_solver_create(&solver);
	lis_solver_set_option(options, solver);

	std::ostringstream tol_option; 
	tol_option << "-tol " << std::scientific << tol; 
	lis_solver_set_option(tol_option.str().data(), solver); 

	lis_solve(A, b, x, solver);

	LIS_INT iterations = 0;
	LIS_REAL residual = 0.0;
	lis_solver_get_iter(solver, &iterations);
	lis_solver_get_residualnorm(solver, &residual);

	lis_output_vector(x, LIS_FMT_MM, const_cast<char *>(solution_path));

	lis_solver_destroy(solver);
	lis_vector_destroy(x);
	lis_vector_destroy(b);
	lis_matrix_destroy(A);
	lis_finalize();

	return {static_cast<int>(iterations), static_cast<double>(residual)};
}

// Solves A x = b for SPD A with Conjugate Gradient
// Returns <solution, iteration_count, relative residual>
// std::tuple<VectorXd, int, double> solve_eigen_spd(const Eigen::SparseMatrix<double> &A,
// 	const VectorXd &b, double tol) {
// 	Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> solver;
// 	solver.setTolerance(tol);
// 	solver.compute(A);
// 	VectorXd x = solver.solve(b);

// 	return {x, static_cast<int>(solver.iterations()), solver.error()};
// }

// Solves A x = b with BiCGSTAB, preconditioned with the default preconditioner
// Returns <solution, iteration_count, relative residual>
std::tuple<VectorXd, int, double> solve_eigen_bicgstab(const Eigen::SparseMatrix<double> &A,
	const VectorXd &b, double tol) {
	Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
	solver.setTolerance(tol);
	solver.compute(A);
	VectorXd x = solver.solve(b);

	return {x, static_cast<int>(solver.iterations()), solver.error()};
}

VectorXd read_mtx_vector_lis(const char *path) {
	std::ifstream in(path);
	if (!in)
		std::cerr <<"Could not open " << path << std::endl;

	std::string header;
	std::getline(in, header);
	if (header.find("vector coordinate") == std::string::npos)
		std::cerr <<"File has an invalid format: " << path << std::endl;

	Eigen::Index size = 0;
	in >> size;

	VectorXd vec = VectorXd::Zero(size);
	Eigen::Index index = 0;
	double value = 0.0;
	while (in >> index >> value)
		vec(index - 1) = value;

	return vec;
}

void write_mtx_vector_lis(const VectorXd &vec, const char *path) {
	std::ofstream out(path);

	if(!out)
		std::cerr <<"Could not open " << path << std::endl;

	out << "%%MatrixMarket vector coordinate real general\n";
	out << vec.size() << "\n";
	out << std::scientific << std::setprecision(17);

	Eigen::Index index = 1;
	for (Eigen::Index i = 0; i < vec.size(); ++i) {
		out << index++ << " " << vec(i) << "\n";
	}
}

int main() {
	std::cout << std::boolalpha;

	// Task 1
	RowMatrixXd input_image = load_image("./artifacts/deer.jpg"); 
	const auto rows = input_image.rows();
	const auto cols = input_image.cols();

	std::cout << "- Task 1: matrix size " << rows << "x" << cols << "\n";

	// Task 2
	constexpr auto noise_scale = 50.0; 
	RowMatrixXd noise = RowMatrixXd::Random(rows, cols) * noise_scale; 

	RowMatrixXd noisy_image = input_image + noise; 

	noisy_image = noisy_image.unaryExpr([](double val) -> double {
		return static_cast<unsigned char>(std::clamp(val, 0.0, 255.0));
	});
	constexpr auto noisy_image_path = "./artifacts/noisy.png";
	save_image(noisy_image, noisy_image_path);
	std::cout << "- Task 2: saved to " << noisy_image_path << "\n";

	// Task 3
	VectorXd input_image_vec = Eigen::Map<VectorXd>(input_image.data(), input_image.size());
	VectorXd noisy_image_vec = Eigen::Map<VectorXd>(noisy_image.data(), noisy_image.size());

	assert(input_image_vec.size() == rows * cols);
	assert(input_image_vec.size() == noisy_image_vec.size());

	auto norm = input_image_vec.norm();
	std::cout << "- Task 3: norm of v " << norm << "\n";

	// Task 4
	MatrixXd av1(3, 3);
	av1 << 1, 1, 1, 1, 4, 1, 1, 1, 1;
	av1 /= 12;

	auto matrix_av1 = build_convolution_matrix(av1, input_image_vec.size(), cols);
	std::cout << "- Task 4: nnz(av1) = " << matrix_av1.nonZeros() << "\n";
	

	// Task 5
	auto smoothed_noisy = matrix_av1 * noisy_image_vec;
	constexpr auto smoothed_noisy_path = "./artifacts/noisy_filtered.png";
	save_image_vec(smoothed_noisy, smoothed_noisy_path, cols);
	std::cout << "- Task 5: saved to " << smoothed_noisy_path << "\n";

	// Task 6
	MatrixXd sh1(3, 3);
	sh1 << 0, -3, 0, -1, 9, -3, 0, -1, 0;

	auto matrix_sh1 = build_convolution_matrix(sh1, input_image_vec.size(), cols);
	std::cout << "- Task 6: nnz(sh1) = " << matrix_sh1.nonZeros()
		<< ", symmetric: " << matrix_sh1.isApprox(matrix_sh1.transpose()) << "\n";

	// matrix_sh1.prune(0.0); // Check that it outputs the same number
	// std::cout << "- Task 6: check nnz(sh1) = " << matrix_sh1.nonZeros() << "\n";

	// Task 7
	auto sharpened_original = matrix_sh1 * input_image_vec;
	constexpr auto sharpened_original_path = "./artifacts/sharpened_original.png";
	save_image_vec(sharpened_original, sharpened_original_path, cols);
	std::cout << "- Task 7: saved to " << sharpened_original_path << "\n";

	// Task 8
	constexpr auto sh1_mtx_path = "./artifacts/sh1.mtx";
	constexpr auto w_mtx_path = "./artifacts/w.mtx";
	Eigen::saveMarket(matrix_sh1, sh1_mtx_path);
	Eigen::saveMarketVector(noisy_image_vec, w_mtx_path);
	std::cout << "- Task 8: saved to " << sh1_mtx_path << " and " << w_mtx_path
		<< "\n";

	// A2 is not symmetric (Task 6), so it can't be SPD: no CG, use BiCGSTAB
	constexpr auto w_lis_path = "./artifacts/w_lis.mtx";
	constexpr auto x_lis_path = "./artifacts/x_lis.mtx";
	write_mtx_vector_lis(noisy_image_vec, w_lis_path);

	char lis_options[] = "-i bicgstab -p ilu";
	auto [lis_iterations, lis_residual] =
		solve_lis_file_output(sh1_mtx_path, w_lis_path, x_lis_path, lis_options, 1e-12);
	std::cout << "- Task 8: LIS bicgstab + ilu, iterations: " << lis_iterations << ", residual: " << lis_residual << "\n";

	// Task 9
	VectorXd x = read_mtx_vector_lis(x_lis_path);
	constexpr auto x_image_path = "./artifacts/x.png";
	save_image_vec(x, x_image_path, cols);
	std::cout << "- Task 9: saved to " << x_image_path << "\n";

	// Task 10
	MatrixXd ed2(3, 3);
	ed2 << -1, 0, 1, -2, 0, 2, -1, 0, 1; 

	auto matrix_ed2 = build_convolution_matrix(ed2, input_image_vec.size(), cols);
	std::cout << "- Task 10: symmetric: " << matrix_ed2.isApprox(matrix_ed2.transpose()) << "\n";
	std::cout << "- Task 10: nnz(ed2) = " << matrix_av1.nonZeros() << "\n";
	

	// Task 11
	auto edge_detected_original = matrix_ed2 * input_image_vec;
	constexpr auto edge_detected_original_path = "./artifacts/edge_detected_original.png";
	save_image_vec(edge_detected_original, edge_detected_original_path, cols);
	std::cout << "- Task 11: saved to " << edge_detected_original_path << "\n";

	// Task 12
	Eigen::SparseMatrix<double> identity(input_image_vec.size(), input_image_vec.size());
	identity.setIdentity();
	Eigen::SparseMatrix<double> matrix_task12 = 4.0 * identity + matrix_ed2;

	// CG needs A SPD, but 4I + A3 is never symmetric.
	std::cout << "- Task 12: 4I + A3 symmetric: " << matrix_task12.isApprox(matrix_task12.transpose()) << " -> not SPD, no CG\n";
	
	auto [y, eigen_iterations, eigen_residual] = solve_eigen_bicgstab(matrix_task12, noisy_image_vec, 1e-10);
	std::cout << "- Task 12: Eigen bicgstab, iterations: " << eigen_iterations << ", residual: " << eigen_residual << "\n";
	// With the default preconditioner it does more iterations but in the end it is faster because of the reduced 



	constexpr auto y_mtx_path = "./artifacts/y.mtx";
	Eigen::saveMarketVector(y, y_mtx_path);

	// Task 13
	VectorXd y_loaded;
	Eigen::loadMarketVector(y_loaded, y_mtx_path); // Here we could even use y directly
	constexpr auto y_image_path = "./artifacts/y.png";
	save_image_vec(y_loaded, y_image_path, cols);
	std::cout << "- Task 13: saved to " << y_image_path << "\n";

	return 0;
}
