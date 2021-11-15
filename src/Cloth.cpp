#include <iostream>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Cloth.h"
#include "Particle.h"
#include "Spring.h"
#include "MatrixStack.h"
#include "Program.h"
#include "GLSL.h"

using namespace std;
using namespace Eigen;

shared_ptr<Spring> createSpring(const shared_ptr<Particle> p0, const shared_ptr<Particle> p1, double E) {
	auto s = make_shared<Spring>(p0, p1);
	s->E = E;
	Vector3d x0 = p0->x;
	Vector3d x1 = p1->x;
	Vector3d dx = x1 - x0;
	s->L = dx.norm();
	return s;
}

Cloth::Cloth(int rows, int cols,
	const Vector3d& x00,
	const Vector3d& x01,
	const Vector3d& x10,
	const Vector3d& x11,
	double mass,
	double stiffness,
	double collisionStiffness) {
	assert(rows > 1);
	assert(cols > 1);
	assert(mass > 0.0);
	assert(stiffness > 0.0);

	this->rows = rows;
	this->cols = cols;
	this->collisionStiffness = collisionStiffness;

	// Create particles
	n = 0;
	double r = 0.01; // Used for collisions
	int nVerts = rows * cols;
	for (int i = 0; i < rows; ++i) {
		double u = i / (rows - 1.0);
		Vector3d x0 = (1 - u) * x00 + u * x10;
		Vector3d x1 = (1 - u) * x01 + u * x11;
		for (int j = 0; j < cols; ++j) {
			double v = j / (cols - 1.0);
			Vector3d x = (1 - v) * x0 + v * x1;
			auto p = make_shared<Particle>();
			particles.push_back(p);
			p->r = r;
			p->x = x;
			p->v << 0.0, 0.0, 0.0;
			p->m = mass / (nVerts);
			// Pin two particles
			if (i == 0 && (j == 0 || j == cols - 1)) {
				p->fixed = true;
				p->i = -1;
			} else {
				p->fixed = false;
				p->i = n;
				n += 3;
			}
		}
	}

	// Create x springs
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols - 1; ++j) {
			int k0 = i * cols + j;
			int k1 = k0 + 1;
			springs.push_back(createSpring(particles[k0], particles[k1], stiffness));
		}
	}

	// Create y springs
	for (int j = 0; j < cols; ++j) {
		for (int i = 0; i < rows - 1; ++i) {
			int k0 = i * cols + j;
			int k1 = k0 + cols;
			springs.push_back(createSpring(particles[k0], particles[k1], stiffness));
		}
	}

	// Create shear springs
	for (int i = 0; i < rows - 1; ++i) {
		for (int j = 0; j < cols - 1; ++j) {
			int k00 = i * cols + j;
			int k10 = k00 + 1;
			int k01 = k00 + cols;
			int k11 = k01 + 1;
			springs.push_back(createSpring(particles[k00], particles[k11], stiffness));
			springs.push_back(createSpring(particles[k10], particles[k01], stiffness));
		}
	}

	// Create x bending springs
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols - 2; ++j) {
			int k0 = i * cols + j;
			int k2 = k0 + 2;
			springs.push_back(createSpring(particles[k0], particles[k2], stiffness));
		}
	}

	// Create y bending springs
	for (int j = 0; j < cols; ++j) {
		for (int i = 0; i < rows - 2; ++i) {
			int k0 = i * cols + j;
			int k2 = k0 + 2 * cols;
			springs.push_back(createSpring(particles[k0], particles[k2], stiffness));
		}
	}

	// Build system matrices and vectors
	M.resize(n, n);
	K.resize(n, n);
	v.resize(n);
	f.resize(n);

	// Build vertex buffers
	posBuf.clear();
	norBuf.clear();
	texBuf.clear();
	eleBuf.clear();
	posBuf.resize(nVerts * 3);
	norBuf.resize(nVerts * 3);
	updatePosNor();

	// Texture coordinates (don't change)
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			texBuf.push_back((float)(i / (rows - 1.0)));
			texBuf.push_back((float)(j / (cols - 1.0)));
		}
	}

	// Elements (don't change)
	for (int i = 0; i < rows - 1; ++i) {
		for (int j = 0; j < cols; ++j) {
			int k0 = i * cols + j;
			int k1 = k0 + cols;
			// Triangle strip
			eleBuf.push_back(k0);
			eleBuf.push_back(k1);
		}
	}
}

Cloth::~Cloth() {
}

void Cloth::tare() {
	for (int k = 0; k < (int)particles.size(); ++k) {
		particles[k]->tare();
	}
}

void Cloth::reset() {
	for (int k = 0; k < (int)particles.size(); ++k) {
		particles[k]->reset();
	}
	updatePosNor();
}

void Cloth::updatePosNor() {
	// Position
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			int k = i * cols + j;
			Vector3d x = particles[k]->x;
			posBuf[3 * k + 0] = (float)x(0);
			posBuf[3 * k + 1] = (float)x(1);
			posBuf[3 * k + 2] = (float)x(2);
		}
	}

	// Normal
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			// Each particle has four neighbors
			//
			//      v1
			//     /|\
			// u0 /_|_\ u1
			//    \ | /
			//     \|/
			//      v0
			//
			// Use these four triangles to compute the normal
			int k = i * cols + j;
			int ku0 = k - 1;
			int ku1 = k + 1;
			int kv0 = k - cols;
			int kv1 = k + cols;
			Vector3d x = particles[k]->x;
			Vector3d xu0, xu1, xv0, xv1, dx0, dx1, c;
			Vector3d nor(0.0, 0.0, 0.0);
			int count = 0;
			// Top-right triangle
			if (j != cols - 1 && i != rows - 1) {
				xu1 = particles[ku1]->x;
				xv1 = particles[kv1]->x;
				dx0 = xu1 - x;
				dx1 = xv1 - x;
				c = dx0.cross(dx1);
				nor += c.normalized();
				++count;
			}
			// Top-left triangle
			if (j != 0 && i != rows - 1) {
				xu1 = particles[kv1]->x;
				xv1 = particles[ku0]->x;
				dx0 = xu1 - x;
				dx1 = xv1 - x;
				c = dx0.cross(dx1);
				nor += c.normalized();
				++count;
			}
			// Bottom-left triangle
			if (j != 0 && i != 0) {
				xu1 = particles[ku0]->x;
				xv1 = particles[kv0]->x;
				dx0 = xu1 - x;
				dx1 = xv1 - x;
				c = dx0.cross(dx1);
				nor += c.normalized();
				++count;
			}
			// Bottom-right triangle
			if (j != cols - 1 && i != 0) {
				xu1 = particles[kv0]->x;
				xv1 = particles[ku1]->x;
				dx0 = xu1 - x;
				dx1 = xv1 - x;
				c = dx0.cross(dx1);
				nor += c.normalized();
				++count;
			}
			nor /= count;
			nor.normalize();
			norBuf[3 * k + 0] = (float)nor(0);
			norBuf[3 * k + 1] = (float)nor(1);
			norBuf[3 * k + 2] = (float)nor(2);
		}
	}
}

void Cloth::blockAddK(Matrix3d& m, int r, int c) {
	double tol = 1e-6;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if (m(i, j) < -tol || m(i, j) > tol) { // nonzero
				K_.push_back(T(r + i, c + j, m(i, j)));
			}
		}
	}
}

void Cloth::step(double h, const Vector3d& grav, const vector< shared_ptr<Particle> > spheres) {
	M.setZero();
	K.setZero();
	v.setZero();
	f.setZero();
	M_.clear();
	K_.clear();

	// fill M, v, and f
	for (auto p : particles) {
		if (!p->fixed) {
			int i = p->i;
			M_.push_back(T(i, i, p->m));
			M_.push_back(T(i + 1, i + 1, p->m));
			M_.push_back(T(i + 2, i + 2, p->m));
			v.segment<3>(i) = p->v;
			f.segment<3>(i) = grav * p->m;
		}
	}
	// compute spring force and fill in K
	for (auto s : springs) {
		int i0 = s->p0->i;
		int i1 = s->p1->i;
		Vector3d deltaX = s->p1->x - s->p0->x;
		double l = deltaX.norm();
		double lDiff = (l - s->L) / l;
		Vector3d fs = s->E * deltaX * lDiff; // f = E(l - L)(x1 - x0)/l
		Matrix3d ks = (s->E / (l * l)) * ((1 - lDiff) * (deltaX * deltaX.transpose()) + lDiff * (double)(deltaX.transpose() * deltaX) * Matrix3d::Identity());
		Matrix3d ksNeg = -ks;

		if (!s->p0->fixed) {
			f.segment<3>(i0) += fs;
			blockAddK(ksNeg, i0, i0);
		}
		if (!s->p1->fixed) {
			f.segment<3>(i1) -= fs;
			blockAddK(ksNeg, i1, i1);
		}
		if (!s->p0->fixed && !s->p1->fixed) {
			blockAddK(ks, i0, i1);
			blockAddK(ks, i1, i0);
		}
	}
	// check collisions and add appropriate force
	for (auto p : particles) {
		if (!p->fixed) {
			for (auto s : spheres) {
				double l = (p->x - s->x).norm();
				Vector3d n = (p->x - s->x) / l; // collision normal
				double d = p->r + s->r - l; // penetration depth
				if (d > 0) { // collision
					Vector3d fc = collisionStiffness * d * n;
					Matrix3d kc = collisionStiffness * d * Matrix3d::Identity();
					f.segment<3>(p->i) += fc;
					blockAddK(kc, p->i, p->i);
				}
			}
		}
	}
	// convert triplet matrices into compressed row
	M.setFromTriplets(M_.begin(), M_.end());
	K.setFromTriplets(K_.begin(), K_.end());

	// solve: Ax = b to get new velocity
	// A = M - h^2 * K
	// b = Mv + hf;
	SparseMatrix<double> A(n, n);
	A = M - h * h * K;
	VectorXd b = M * v + h * f;
	// solve with Conjugate Gradients
	ConjugateGradient< SparseMatrix<double> > cg;
	cg.setMaxIterations(25);
	cg.setTolerance(1e-6);
	cg.compute(A);
	VectorXd x = cg.solveWithGuess(b, v);

	for (auto p : particles) {
		if (!p->fixed) {
			p->v = x.segment<3>(p->i);
			p->x += p->v * h;
		}
	}

	if (false) {
		cout << "M:" << endl << M << endl;
		cout << "v:" << endl << v << endl;
		cout << "f:" << endl << f << endl;
		cout << "A:" << endl << A << endl;
		cout << "b:" << endl << b << endl;
		cout << "x:" << endl << x << endl;
	}
	// Update position and normal buffers
	updatePosNor();
}

void Cloth::init() {
	glGenBuffers(1, &posBufID);
	glBindBuffer(GL_ARRAY_BUFFER, posBufID);
	glBufferData(GL_ARRAY_BUFFER, posBuf.size() * sizeof(float), &posBuf[0], GL_DYNAMIC_DRAW);

	glGenBuffers(1, &norBufID);
	glBindBuffer(GL_ARRAY_BUFFER, norBufID);
	glBufferData(GL_ARRAY_BUFFER, norBuf.size() * sizeof(float), &norBuf[0], GL_DYNAMIC_DRAW);

	glGenBuffers(1, &texBufID);
	glBindBuffer(GL_ARRAY_BUFFER, texBufID);
	glBufferData(GL_ARRAY_BUFFER, texBuf.size() * sizeof(float), &texBuf[0], GL_STATIC_DRAW);

	glGenBuffers(1, &eleBufID);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eleBufID);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, eleBuf.size() * sizeof(unsigned int), &eleBuf[0], GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	assert(glGetError() == GL_NO_ERROR);
}

void Cloth::draw(shared_ptr<MatrixStack> MV, const shared_ptr<Program> p) const {
	// Draw mesh
	glUniform3fv(p->getUniform("kdFront"), 1, Vector3f(1.0, 0.0, 0.0).data());
	glUniform3fv(p->getUniform("kdBack"), 1, Vector3f(1.0, 1.0, 0.0).data());
	MV->pushMatrix();
	glUniformMatrix4fv(p->getUniform("MV"), 1, GL_FALSE, glm::value_ptr(MV->topMatrix()));
	int h_pos = p->getAttribute("aPos");
	glEnableVertexAttribArray(h_pos);
	glBindBuffer(GL_ARRAY_BUFFER, posBufID);
	glBufferData(GL_ARRAY_BUFFER, posBuf.size() * sizeof(float), &posBuf[0], GL_DYNAMIC_DRAW);
	glVertexAttribPointer(h_pos, 3, GL_FLOAT, GL_FALSE, 0, (const void*)0);
	int h_nor = p->getAttribute("aNor");
	glEnableVertexAttribArray(h_nor);
	glBindBuffer(GL_ARRAY_BUFFER, norBufID);
	glBufferData(GL_ARRAY_BUFFER, norBuf.size() * sizeof(float), &norBuf[0], GL_DYNAMIC_DRAW);
	glVertexAttribPointer(h_nor, 3, GL_FLOAT, GL_FALSE, 0, (const void*)0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eleBufID);
	for (int i = 0; i < rows; ++i) {
		glDrawElements(GL_TRIANGLE_STRIP, 2 * cols, GL_UNSIGNED_INT, (const void*)(2 * cols * i * sizeof(unsigned int)));
	}
	glDisableVertexAttribArray(h_nor);
	glDisableVertexAttribArray(h_pos);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	MV->popMatrix();
}
