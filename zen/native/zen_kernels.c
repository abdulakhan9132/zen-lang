/* Native C99 math kernels for ZEN runtime (optional acceleration; pure-Python fallback always exists). */
#include <math.h>
#include <stddef.h>

#ifdef _WIN32
#define ZEN_EXPORT __declspec(dllexport)
#else
#define ZEN_EXPORT __attribute__((visibility("default")))
#endif

#define ZEN_BLOCK 64

ZEN_EXPORT void zen_matmul_f64(
    const double *A,
    const double *B,
    double *C,
    int M,
    int K,
    int N
) {
    for (int i = 0; i < M * N; ++i) {
        C[i] = 0.0;
    }
    for (int ii = 0; ii < M; ii += ZEN_BLOCK) {
        int i_end = ii + ZEN_BLOCK < M ? ii + ZEN_BLOCK : M;
        for (int kk = 0; kk < K; kk += ZEN_BLOCK) {
            int k_end = kk + ZEN_BLOCK < K ? kk + ZEN_BLOCK : K;
            for (int jj = 0; jj < N; jj += ZEN_BLOCK) {
                int j_end = jj + ZEN_BLOCK < N ? jj + ZEN_BLOCK : N;
                for (int i = ii; i < i_end; ++i) {
                    const double *a_row = A + (size_t)i * K;
                    double *c_row = C + (size_t)i * N;
                    for (int k = kk; k < k_end; ++k) {
                        double a_ik = a_row[k];
                        const double *b_row = B + (size_t)k * N;
                        for (int j = jj; j < j_end; ++j) {
                            c_row[j] += a_ik * b_row[j];
                        }
                    }
                }
            }
        }
    }
}

ZEN_EXPORT void zen_add_f64(const double *a, const double *b, double *out, int n) {
    for (int i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

ZEN_EXPORT void zen_sub_f64(const double *a, const double *b, double *out, int n) {
    for (int i = 0; i < n; ++i) out[i] = a[i] - b[i];
}

ZEN_EXPORT void zen_mul_f64(const double *a, const double *b, double *out, int n) {
    for (int i = 0; i < n; ++i) out[i] = a[i] * b[i];
}

ZEN_EXPORT void zen_div_f64(const double *a, const double *b, double *out, int n) {
    for (int i = 0; i < n; ++i) out[i] = a[i] / b[i];
}

ZEN_EXPORT void zen_exp_f64(const double *a, double *out, int n) {
    for (int i = 0; i < n; ++i) out[i] = exp(a[i]);
}

ZEN_EXPORT void zen_log_f64(const double *a, double *out, int n) {
    for (int i = 0; i < n; ++i) out[i] = log(a[i]);
}

ZEN_EXPORT void zen_tanh_f64(const double *a, double *out, int n) {
    for (int i = 0; i < n; ++i) out[i] = tanh(a[i]);
}

ZEN_EXPORT void zen_relu_f64(const double *a, double *out, int n) {
    for (int i = 0; i < n; ++i) out[i] = a[i] > 0.0 ? a[i] : 0.0;
}

ZEN_EXPORT double zen_sum_f64(const double *a, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += a[i];
    return s;
}

ZEN_EXPORT double zen_max_f64(const double *a, int n) {
    double m = a[0];
    for (int i = 1; i < n; ++i) {
        if (a[i] > m) m = a[i];
    }
    return m;
}

ZEN_EXPORT double zen_min_f64(const double *a, int n) {
    double m = a[0];
    for (int i = 1; i < n; ++i) {
        if (a[i] < m) m = a[i];
    }
    return m;
}

ZEN_EXPORT void zen_softmax_2d_f64(const double *X, double *Out, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const double *x = X + (size_t)r * cols;
        double *o = Out + (size_t)r * cols;
        double max_v = x[0];
        for (int c = 1; c < cols; ++c) {
            if (x[c] > max_v) max_v = x[c];
        }
        double sum_e = 0.0;
        for (int c = 0; c < cols; ++c) {
            double e = exp(x[c] - max_v);
            o[c] = e;
            sum_e += e;
        }
        double inv = 1.0 / sum_e;
        for (int c = 0; c < cols; ++c) {
            o[c] *= inv;
        }
    }
}

ZEN_EXPORT void zen_layernorm_2d_f64(
    const double *X,
    const double *gamma,
    const double *beta,
    double *Out,
    int rows,
    int cols,
    double eps
) {
    for (int r = 0; r < rows; ++r) {
        const double *x = X + (size_t)r * cols;
        double *o = Out + (size_t)r * cols;
        double mean = 0.0;
        for (int c = 0; c < cols; ++c) mean += x[c];
        mean /= (double)cols;

        double var = 0.0;
        for (int c = 0; c < cols; ++c) {
            double d = x[c] - mean;
            var += d * d;
        }
        double inv_std = 1.0 / sqrt(var / (double)cols + eps);
        for (int c = 0; c < cols; ++c) {
            double norm = (x[c] - mean) * inv_std;
            double g = gamma ? gamma[c] : 1.0;
            double b = beta ? beta[c] : 0.0;
            o[c] = norm * g + b;
        }
    }
}

ZEN_EXPORT void zen_embedding_f64(
    const double *table,
    const int *indices,
    double *Out,
    int n_idx,
    int dim
) {
    for (int i = 0; i < n_idx; ++i) {
        const double *src = table + (size_t)indices[i] * dim;
        double *dst = Out + (size_t)i * dim;
        for (int d = 0; d < dim; ++d) {
            dst[d] = src[d];
        }
    }
}
