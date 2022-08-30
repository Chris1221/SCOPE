#ifndef PROPCA_MAILMAN_H_
#define PROPCA_MAILMAN_H_


#include <assert.h>
#include <vector>

#include "storage.h"
#include "genotype.h"


#if SSE_SUPPORT == 1
	#define fastmultiply fastmultiply_sse
	#define fastmultiply_pre fastmultiply_pre_sse
#else
	#define fastmultiply fastmultiply_normal
	#define fastmultiply_pre fastmultiply_pre_normal
#endif


namespace mailman {

	/* Compute Y=A*X where A is a m X n matrix represented in mailman form. X is a n X k matrix
	 * m : number of rows
	 * n : number of columns
	 * k : batching factor: how many vectors to operate on 
	 * p : matrix A represented in mailman form
	 * x : matrix X
	 * yint : intermediate computation
	 * c : intermediate computation
	 * y : result
	 */
	void fastmultiply_normal(int m, int n , int k, std::vector<int> &p,
							 MatrixXdr &x,
							 double *yint, double *c, double **y) {
		for (int i = 0; i < n; i++) {
			int l = p[i];
			for (int j = 0; j < k; j ++) {
				yint[l * k + j] += x(i, j);
			}
		}

		int d = pow(3, m);
		for (int j  = 0; j < m; j++) {
			d = d / 3;
			for (int l = 0; l < k ; l++)
				c[l] = 0;
			for (int i = 0; i < d; i++) {
				for (int l = 0; l < k; l++) {
					double z1 = yint[l + (i + d) * k];
					double z2 = yint[l + (i + 2 * d) * k];
					yint[l + (i + d) * k] = 0;
					yint[l + (i + 2 * d) * k] = 0;
					yint[l + i * k] = yint[l + i * k] + z1 + z2;
					c[l] += (z1 + 2 * z2);
				}
			}
			for (int l = 0; l < k ; l++) {
				y[j][l] = c[l];
			}
		}
		for (int l = 0; l < k ; l++) {
			yint[l] = 0;
		}
	}

	/* Compute Y=X*A + Y_0 where A is a m X n matrix represented in mailman form. X is a k X m matrix. Y_0 is a k X n matrix.
	 * X is specified as a matrix k X m_0 matrix X_0 (m_0 > m) 
	 * and a index: start \in {1..m_0 }
	 * so that X = X_0 [start:(start+m-1),]
	 *
	 * m : number of rows
	 * n : number of columns
	 * k : batching factor: how many vectors to operate on 
	 * start: index into X_0
	 * p : matrix A represented in mailman format
	 * x : matrix X_0 
	 * yint : intermediate computation
	 * c : intermediate computation
	 * y : result. also contains Y_0 that is updated.
	 */
	void fastmultiply_pre_normal(int m, int n, int k, int start, std::vector<int> &p,
								 MatrixXdr &x,
								 double *yint, double *c, double **y) {
		int size1 = pow(3.0, m);
		memset (yint, 0, size1* sizeof(double));

		int prefix = 1;
		for (int i  = m - 1; i >= 0; i--) {
			int i1 = start + i;
			for (int j = 0; j < prefix; j++) {
				int offset0 = j * k;
				int offset1 = (prefix + j) * k;
				int offset2 = (2 * prefix + j) * k;
				for (int l = 0; l < k; l++) {
					yint[offset1 + l] = yint[offset0 + l] + x(i1, l);
					yint[offset2 + l] = yint[offset0 + l] + 2 * x(i1, l);
				}
			}
			prefix *= 3;
		}

		for (int i = 0; i < n; i++) {
			for (int l = 0; l < k  ; l++) {
				y[i][l] += yint[l + p[i] * k];
				// yint[l+ p[i]*k] = 0 ;
			}
		}
	}
}  // namespace mailman

#endif  // PROPCA_MAILMAN_H_
