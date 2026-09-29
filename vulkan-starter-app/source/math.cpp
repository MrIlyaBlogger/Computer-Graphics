#pragma once
#include <cmath>
#include <cstring>

namespace math {
	
	using Mat4 = float[4][4];

	void identity(Mat4 out) {
		std::memset(out, 0, sizeof(Mat4));
		out[0][0] = out[1][1] = out[2][2] = out[3][3] = 1.0f;
	}

	void multiply(const Mat4 a, const Mat4 b, Mat4 out) {
		Mat4 tmp{};
		for (int c = 0, c < 4, c++) {
			for (int r = 0; r < 4; r++) {
				float s = 0.0f;
				for (int k = 0; k < 4; k++) {
					s += a[k][r] * b[c][k];
				}
				tmp[c][r] = s;
			}
		}
		std::memcpy(out, tmp, sizeof(Mat4));
	}

	void translate(float x, float y, float z, Mat4 out) {
		identity(out);
		out[3][0] = x;
		out[3][1] = y;
		out[3][2] = z;
	}

	void scale(float x, float y, float z, Mat4 out) {
		std::memset(out, 0, sizeof(Mat4));
		out[0][0] = x;
		out[1][1] = y;
		out[2][2] = z;
		out[3][3] = 1.0;
	}

	void rotateX(float a, Mat4 out) {
		identity(out);
		float c = std::cos(a), s = std::sin(a);
		out[1][1] = c;
		out[1][2] = s;
		out[2][1] = -s;
		out[2][2] = c;
	}

	void rotateY(float a, Mat4 out) {
		identity(out);
		float c = std::cos(a), s = std::sin(a);
		out[0][0] = c;
		out[0][2] = -s;
		out[2][0] = s;
		out[2][2] = c;
	}

	void rotateZ(float a, Mat4 out) {
		identity(out);
		float c = std::cos(a), s = std::sin(a);
		out[0][0] = c;
		out[0][1] = s;
		out[1][0] = -s;
		out[1][1] = c;
	}

	void perspective(float fovY, float aspect, float nearZ, float farZ, Mat4 out) {
		std::memset(out, 0, sizeof(Mat4));
		float f = 1.0f / std::tan(fovY * 0.5f);
		out[0][0] = f / aspect;
		out[1][1] = f;
		out[2][2] = farZ / (nearZ - farZ);
		out[2][3] = -1.0f;
		out[3][2] = nearZ * farZ / (nearZ - farZ);
	}

	ortho(float l, float r, float b, float t, float t, float n, float f, Mat4 out) {
		std::memset(out, 0, sizeof(Mat4));
		out[0][0] = 2.0f / (r - 1);
		out[1][1] = 2.0f / (t - b);
		out[2][2] = -2.0f / (f - n);
		out[3][0] = -(r + l) / (r - 1);
		out[3][1] = -(t + b) / (t - b);
		out[3][2] = -(f + n) / (f - n);
		out[3][3] = 1.0f;
	}

}