#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_RANGE 128
#define N_ANGLE 128
#define N_VEL 128
#define N_CHIRP 255
#define N_RX 8
#define N_SAMPLE 128
#define CFAR_GUARD_CELLS_ROW 20
#define CFAR_GUARD_CELLS_COL 2
#define CFAR_TRAINING_CELLS_ROW 8
#define CFAR_TRAINING_CELLS_COL 8
#define CFAR_THRESHOLD_SCALE_ROW 3.0f
#define CFAR_THRESHOLD_SCALE_COL 3.0f
#define CLUSTER_RADIUS 2
#define CLUSTER_MIN_SIZE 1
#define PI 3.14159265358979323846f

typedef struct {
    float re;
    float im;
} complex_f32_t;

static complex_f32_t complex_add(complex_f32_t a, complex_f32_t b) {
    complex_f32_t out = {a.re + b.re, a.im + b.im};
    return out;
}

static complex_f32_t complex_sub(complex_f32_t a, complex_f32_t b) {
    complex_f32_t out = {a.re - b.re, a.im - b.im};
    return out;
}

static complex_f32_t complex_mul(complex_f32_t a, complex_f32_t b) {
    complex_f32_t out = {
        a.re * b.re - a.im * b.im,
        a.re * b.im + a.im * b.re,
    };
    return out;
}

static size_t bit_reverse(size_t value, unsigned int bits) {
    size_t reversed = 0;
    for (unsigned int i = 0; i < bits; ++i) {
        reversed = (reversed << 1) | (value & 1u);
        value >>= 1;
    }
    return reversed;
}

static int is_power_of_two(size_t n) {
    return n != 0 && (n & (n - 1u)) == 0;
}

static void bit_reverse_reorder(complex_f32_t *data, size_t n) {
    unsigned int bits = 0;
    for (size_t tmp = n; tmp > 1u; tmp >>= 1u) {
        ++bits;
    }

    for (size_t i = 0; i < n; ++i) {
        size_t j = bit_reverse(i, bits);
        if (j > i) {
            complex_f32_t tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
        }
    }
}

static int fft_radix2(complex_f32_t *data, size_t n, int inverse) {
    if (!is_power_of_two(n)) {
        return -1;
    }

    bit_reverse_reorder(data, n);

    for (size_t size = 2; size <= n; size <<= 1u) {
        size_t half_size = size >> 1u;
        float sign = inverse ? 1.0f : -1.0f;
        float angle = sign * 2.0f * PI / (float)size;
        complex_f32_t twiddle_step = {cosf(angle), sinf(angle)};

        for (size_t start = 0; start < n; start += size) {
            complex_f32_t twiddle = {1.0f, 0.0f};
            for (size_t offset = 0; offset < half_size; ++offset) {
                size_t even_idx = start + offset;
                size_t odd_idx = even_idx + half_size;
                complex_f32_t even = data[even_idx];
                complex_f32_t odd = complex_mul(twiddle, data[odd_idx]);

                data[even_idx] = complex_add(even, odd);
                data[odd_idx] = complex_sub(even, odd);
                twiddle = complex_mul(twiddle, twiddle_step);
            }
        }
    }

    if (inverse) {
        float scale = 1.0f / (float)n;
        for (size_t i = 0; i < n; ++i) {
            data[i].re *= scale;
            data[i].im *= scale;
        }
    }

    return 0;
}

static size_t adc_idx(size_t sample, size_t rx, size_t chirp) {
    return (sample * N_RX + rx) * N_CHIRP + chirp;
}

static size_t cube_idx(size_t range, size_t rx, size_t chirp) {
    return (range * N_RX + rx) * N_CHIRP + chirp;
}

static size_t map_idx(size_t row, size_t col) {
    return row * N_ANGLE + col;
}

static void hamming_window(float *window, size_t n) {
    if (n == 1u) {
        window[0] = 1.0f;
        return;
    }

    for (size_t i = 0; i < n; ++i) {
        window[i] = 0.54f - 0.46f * cosf(2.0f * PI * (float)i / (float)(n - 1u));
    }
}

static float complex_abs(complex_f32_t value) {
    return sqrtf(value.re * value.re + value.im * value.im);
}

static void fftshift_128(const complex_f32_t *input, complex_f32_t *output) {
    for (size_t i = 0; i < 64u; ++i) {
        output[i] = input[i + 64u];
        output[i + 64u] = input[i];
    }
}

static int read_complex64_file(const char *path, complex_f32_t *data, size_t count) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "failed to open input '%s': %s\n", path, strerror(errno));
        return -1;
    }

    size_t read_count = fread(data, sizeof(*data), count, file);
    int failed = ferror(file);
    fclose(file);

    if (failed || read_count != count) {
        fprintf(stderr, "input '%s' has %zu complex samples, expected %zu\n", path, read_count, count);
        return -1;
    }

    return 0;
}

static int write_npy_float32(const char *path, const float *data, const size_t *shape, size_t ndim) {
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        fprintf(stderr, "failed to open output '%s': %s\n", path, strerror(errno));
        return -1;
    }

    char shape_text[128];
    size_t offset = 0;
    offset += (size_t)snprintf(shape_text + offset, sizeof(shape_text) - offset, "(");
    for (size_t i = 0; i < ndim; ++i) {
        offset += (size_t)snprintf(
            shape_text + offset,
            sizeof(shape_text) - offset,
            "%zu%s",
            shape[i],
            ndim == 1u ? "," : (i + 1u == ndim ? "" : ", ")
        );
    }
    snprintf(shape_text + offset, sizeof(shape_text) - offset, ")");

    char header[256];
    int header_base_len = snprintf(
        header,
        sizeof(header),
        "{'descr': '<f4', 'fortran_order': False, 'shape': %s, }",
        shape_text
    );
    if (header_base_len < 0 || (size_t)header_base_len >= sizeof(header)) {
        fclose(file);
        fprintf(stderr, "failed to build npy header for '%s'\n", path);
        return -1;
    }

    size_t header_len = (size_t)header_base_len;
    size_t preamble_len = 10u;
    size_t padding = 16u - ((preamble_len + header_len + 1u) % 16u);
    if (padding == 16u) {
        padding = 0u;
    }
    if (header_len + padding + 1u > sizeof(header)) {
        fclose(file);
        fprintf(stderr, "npy header too large for '%s'\n", path);
        return -1;
    }

    memset(header + header_len, ' ', padding);
    header_len += padding;
    header[header_len++] = '\n';

    const unsigned char magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    unsigned char len_bytes[2] = {
        (unsigned char)(header_len & 0xffu),
        (unsigned char)((header_len >> 8u) & 0xffu),
    };

    size_t count = 1u;
    for (size_t i = 0; i < ndim; ++i) {
        count *= shape[i];
    }

    int ok = fwrite(magic, 1u, sizeof(magic), file) == sizeof(magic)
             && fwrite(len_bytes, 1u, sizeof(len_bytes), file) == sizeof(len_bytes)
             && fwrite(header, 1u, header_len, file) == header_len
             && fwrite(data, sizeof(*data), count, file) == count;

    if (fclose(file) != 0) {
        ok = 0;
    }

    if (!ok) {
        fprintf(stderr, "failed to write complete npy output '%s'\n", path);
        return -1;
    }

    return 0;
}

static int write_npy_bool(const char *path, const uint8_t *data, const size_t *shape, size_t ndim) {
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        fprintf(stderr, "failed to open output '%s': %s\n", path, strerror(errno));
        return -1;
    }

    char shape_text[128];
    size_t offset = 0;
    offset += (size_t)snprintf(shape_text + offset, sizeof(shape_text) - offset, "(");
    for (size_t i = 0; i < ndim; ++i) {
        offset += (size_t)snprintf(
            shape_text + offset,
            sizeof(shape_text) - offset,
            "%zu%s",
            shape[i],
            ndim == 1u ? "," : (i + 1u == ndim ? "" : ", ")
        );
    }
    snprintf(shape_text + offset, sizeof(shape_text) - offset, ")");

    char header[256];
    int header_base_len = snprintf(
        header,
        sizeof(header),
        "{'descr': '|b1', 'fortran_order': False, 'shape': %s, }",
        shape_text
    );
    if (header_base_len < 0 || (size_t)header_base_len >= sizeof(header)) {
        fclose(file);
        fprintf(stderr, "failed to build npy header for '%s'\n", path);
        return -1;
    }

    size_t header_len = (size_t)header_base_len;
    size_t preamble_len = 10u;
    size_t padding = 16u - ((preamble_len + header_len + 1u) % 16u);
    if (padding == 16u) {
        padding = 0u;
    }
    if (header_len + padding + 1u > sizeof(header)) {
        fclose(file);
        fprintf(stderr, "npy header too large for '%s'\n", path);
        return -1;
    }

    memset(header + header_len, ' ', padding);
    header_len += padding;
    header[header_len++] = '\n';

    const unsigned char magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    unsigned char len_bytes[2] = {
        (unsigned char)(header_len & 0xffu),
        (unsigned char)((header_len >> 8u) & 0xffu),
    };

    size_t count = 1u;
    for (size_t i = 0; i < ndim; ++i) {
        count *= shape[i];
    }

    int ok = fwrite(magic, 1u, sizeof(magic), file) == sizeof(magic)
             && fwrite(len_bytes, 1u, sizeof(len_bytes), file) == sizeof(len_bytes)
             && fwrite(header, 1u, header_len, file) == header_len
             && fwrite(data, sizeof(*data), count, file) == count;

    if (fclose(file) != 0) {
        ok = 0;
    }

    if (!ok) {
        fprintf(stderr, "failed to write complete npy output '%s'\n", path);
        return -1;
    }

    return 0;
}

static int range_fft(const complex_f32_t *adc_data, complex_f32_t *range_data) {
    float window[N_SAMPLE];
    complex_f32_t line[N_RANGE];
    hamming_window(window, N_SAMPLE);

    for (size_t rx = 0; rx < N_RX; ++rx) {
        for (size_t chirp = 0; chirp < N_CHIRP; ++chirp) {
            for (size_t sample = 0; sample < N_SAMPLE; ++sample) {
                complex_f32_t value = adc_data[adc_idx(sample, rx, chirp)];
                line[sample].re = value.re * window[sample];
                line[sample].im = value.im * window[sample];
            }

            if (fft_radix2(line, N_RANGE, 0) != 0) {
                return -1;
            }

            for (size_t range = 0; range < N_RANGE; ++range) {
                range_data[cube_idx(range, rx, chirp)] = line[range];
            }
        }
    }

    return 0;
}

static int produce_rv_maps(const complex_f32_t *range_data, float *rv_maps) {
    float window[N_VEL];
    complex_f32_t line[N_VEL];
    complex_f32_t shifted[N_VEL];
    hamming_window(window, N_VEL);

    for (size_t range = 0; range < N_RANGE; ++range) {
        for (size_t rx = 0; rx < N_RX; ++rx) {
            for (size_t slice = 0; slice < 2u; ++slice) {
                size_t chirp_offset = slice == 0u ? 0u : (N_VEL - 1u);
                for (size_t vel = 0; vel < N_VEL; ++vel) {
                    complex_f32_t value = range_data[cube_idx(range, rx, chirp_offset + vel)];
                    line[vel].re = value.re * window[vel];
                    line[vel].im = value.im * window[vel];
                }

                if (fft_radix2(line, N_VEL, 0) != 0) {
                    return -1;
                }

                fftshift_128(line, shifted);
                for (size_t vel = 0; vel < N_VEL; ++vel) {
                    rv_maps[(slice * N_RANGE + range) * N_VEL + vel] += complex_abs(shifted[vel]) / (float)N_RX;
                }
            }
        }
    }

    return 0;
}

static int produce_ra_map(const complex_f32_t *range_data, float *ra_map) {
    float window[N_RX];
    complex_f32_t line[N_ANGLE];
    complex_f32_t shifted[N_ANGLE];
    hamming_window(window, N_RX);

    for (size_t range = 0; range < N_RANGE; ++range) {
        memset(line, 0, sizeof(line));
        for (size_t rx = 0; rx < N_RX; ++rx) {
            complex_f32_t value = range_data[cube_idx(range, rx, 0)];
            line[rx].re = value.re * window[rx];
            line[rx].im = value.im * window[rx];
        }

        if (fft_radix2(line, N_ANGLE, 0) != 0) {
            return -1;
        }

        fftshift_128(line, shifted);
        for (size_t angle = 0; angle < N_ANGLE; ++angle) {
            ra_map[map_idx(range, angle)] = complex_abs(shifted[angle]);
        }
    }

    return 0;
}

static void cfar_axis1(const float *data,
                       uint8_t *detections,
                       size_t rows,
                       size_t cols,
                       size_t guard_cells,
                       size_t training_cells,
                       float threshold_scale) {
    size_t start_cell = guard_cells + training_cells;
    size_t end_cell = cols - guard_cells - training_cells;

    for (size_t row = 0; row < rows; ++row) {
        for (size_t cell = start_cell; cell < end_cell; ++cell) {
            float training_sum = 0.0f;
            for (size_t col = cell - guard_cells - training_cells; col < cell - guard_cells; ++col) {
                training_sum += data[row * cols + col];
            }
            for (size_t col = cell + guard_cells + 1u; col < cell + guard_cells + training_cells + 1u; ++col) {
                training_sum += data[row * cols + col];
            }

            float threshold = (training_sum / (float)(2u * training_cells)) * threshold_scale;
            detections[row * cols + cell] = data[row * cols + cell] > threshold ? 1u : 0u;
        }
    }
}

static void cfar_axis0(const float *data,
                       uint8_t *detections,
                       size_t rows,
                       size_t cols,
                       size_t guard_cells,
                       size_t training_cells,
                       float threshold_scale) {
    size_t start_cell = guard_cells + training_cells;
    size_t end_cell = rows - guard_cells - training_cells;

    for (size_t col = 0; col < cols; ++col) {
        for (size_t cell = start_cell; cell < end_cell; ++cell) {
            float training_sum = 0.0f;
            for (size_t row = cell - guard_cells - training_cells; row < cell - guard_cells; ++row) {
                training_sum += data[row * cols + col];
            }
            for (size_t row = cell + guard_cells + 1u; row < cell + guard_cells + training_cells + 1u; ++row) {
                training_sum += data[row * cols + col];
            }

            float threshold = (training_sum / (float)(2u * training_cells)) * threshold_scale;
            detections[cell * cols + col] = data[cell * cols + col] > threshold ? 1u : 0u;
        }
    }
}

static void cfar2(const float *data, uint8_t *detections, size_t rows, size_t cols) {
    const size_t count = rows * cols;
    uint8_t *row_detections = calloc(count, sizeof(*row_detections));
    uint8_t *column_detections = calloc(count, sizeof(*column_detections));
    if (row_detections == NULL || column_detections == NULL) {
        free(row_detections);
        free(column_detections);
        memset(detections, 0, count * sizeof(*detections));
        return;
    }

    cfar_axis1(
        data,
        row_detections,
        rows,
        cols,
        CFAR_GUARD_CELLS_ROW,
        CFAR_TRAINING_CELLS_ROW,
        CFAR_THRESHOLD_SCALE_ROW
    );
    cfar_axis0(
        data,
        column_detections,
        rows,
        cols,
        CFAR_GUARD_CELLS_COL,
        CFAR_TRAINING_CELLS_COL,
        CFAR_THRESHOLD_SCALE_COL
    );

    for (size_t i = 0; i < count; ++i) {
        detections[i] = row_detections[i] && column_detections[i] ? 1u : 0u;
    }

    free(row_detections);
    free(column_detections);
}

static void dilate_detections(const uint8_t *detections, uint8_t *expanded, size_t rows, size_t cols, size_t radius) {
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            if (!detections[row * cols + col]) {
                continue;
            }

            size_t row_start = row > radius ? row - radius : 0u;
            size_t row_end = row + radius + 1u < rows ? row + radius + 1u : rows;
            size_t col_start = col > radius ? col - radius : 0u;
            size_t col_end = col + radius + 1u < cols ? col + radius + 1u : cols;

            for (size_t rr = row_start; rr < row_end; ++rr) {
                for (size_t cc = col_start; cc < col_end; ++cc) {
                    expanded[rr * cols + cc] = 1u;
                }
            }
        }
    }
}

static void label_component(const uint8_t *expanded,
                            int *labels,
                            size_t rows,
                            size_t cols,
                            size_t seed_row,
                            size_t seed_col,
                            int label_id) {
    const size_t count = rows * cols;
    size_t *queue = malloc(count * sizeof(*queue));
    if (queue == NULL) {
        return;
    }

    size_t head = 0;
    size_t tail = 0;
    size_t seed = seed_row * cols + seed_col;
    labels[seed] = label_id;
    queue[tail++] = seed;

    while (head < tail) {
        size_t idx = queue[head++];
        size_t row = idx / cols;
        size_t col = idx % cols;

        size_t row_start = row > 0u ? row - 1u : 0u;
        size_t row_end = row + 2u < rows ? row + 2u : rows;
        size_t col_start = col > 0u ? col - 1u : 0u;
        size_t col_end = col + 2u < cols ? col + 2u : cols;

        for (size_t rr = row_start; rr < row_end; ++rr) {
            for (size_t cc = col_start; cc < col_end; ++cc) {
                size_t neighbor = rr * cols + cc;
                if (expanded[neighbor] && labels[neighbor] == 0) {
                    labels[neighbor] = label_id;
                    queue[tail++] = neighbor;
                }
            }
        }
    }

    free(queue);
}

static void cluster_detections(const uint8_t *detections,
                               const float *values,
                               uint8_t *clustered,
                               size_t rows,
                               size_t cols,
                               size_t radius,
                               size_t min_size) {
    const size_t count = rows * cols;
    memset(clustered, 0, count * sizeof(*clustered));

    uint8_t *expanded = calloc(count, sizeof(*expanded));
    int *labels = calloc(count, sizeof(*labels));
    if (expanded == NULL || labels == NULL) {
        free(expanded);
        free(labels);
        return;
    }

    dilate_detections(detections, expanded, rows, cols, radius);

    int label_id = 0;
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            size_t idx = row * cols + col;
            if (expanded[idx] && labels[idx] == 0) {
                ++label_id;
                label_component(expanded, labels, rows, cols, row, col, label_id);
            }
        }
    }

    for (int label = 1; label <= label_id; ++label) {
        size_t cluster_size = 0;
        size_t best_idx = 0;
        float best_value = 0.0f;
        int has_best = 0;

        for (size_t idx = 0; idx < count; ++idx) {
            if (detections[idx] && labels[idx] == label) {
                ++cluster_size;
                if (!has_best || values[idx] > best_value) {
                    best_value = values[idx];
                    best_idx = idx;
                    has_best = 1;
                }
            }
        }

        if (has_best && cluster_size >= min_size) {
            clustered[best_idx] = 1u;
        }
    }

    free(expanded);
    free(labels);
}

static int process_radar(const complex_f32_t *adc_data, float *output_maps) {
    const size_t cube_count = (size_t)N_RANGE * N_RX * N_CHIRP;
    complex_f32_t *range_data = calloc(cube_count, sizeof(*range_data));
    if (range_data == NULL) {
        fprintf(stderr, "failed to allocate range_data\n");
        return -1;
    }

    float *ra_map = output_maps;
    float *rv_maps = output_maps + (size_t)N_RANGE * N_ANGLE;

    int status = 0;
    if (range_fft(adc_data, range_data) != 0) {
        status = -1;
    } else if (produce_ra_map(range_data, ra_map) != 0) {
        status = -1;
    } else if (produce_rv_maps(range_data, rv_maps) != 0) {
        status = -1;
    }

    free(range_data);
    return status;
}

static int write_outputs(const char *prefix, const float *output_maps) {
    char ra_path[512];
    char rv_map_path[512];
    char rv_path[512];
    char ra_cfar_path[512];
    char rv_cfar_path[512];
    char ra_clusters_path[512];
    char rv_clusters_path[512];
    int ra_len = snprintf(ra_path, sizeof(ra_path), "%s_ra_map.npy", prefix);
    int rv_map_len = snprintf(rv_map_path, sizeof(rv_map_path), "%s_rv_map.npy", prefix);
    int rv_len = snprintf(rv_path, sizeof(rv_path), "%s_rv_slice.npy", prefix);
    int ra_cfar_len = snprintf(ra_cfar_path, sizeof(ra_cfar_path), "%s_ra_cfar.npy", prefix);
    int rv_cfar_len = snprintf(rv_cfar_path, sizeof(rv_cfar_path), "%s_rv_cfar.npy", prefix);
    int ra_clusters_len = snprintf(ra_clusters_path, sizeof(ra_clusters_path), "%s_ra_clusters.npy", prefix);
    int rv_clusters_len = snprintf(rv_clusters_path, sizeof(rv_clusters_path), "%s_rv_clusters.npy", prefix);
    if (ra_len < 0 || rv_map_len < 0 || rv_len < 0
        || ra_cfar_len < 0 || rv_cfar_len < 0
        || ra_clusters_len < 0 || rv_clusters_len < 0
        || (size_t)ra_len >= sizeof(ra_path)
        || (size_t)rv_map_len >= sizeof(rv_map_path)
        || (size_t)rv_len >= sizeof(rv_path)
        || (size_t)ra_cfar_len >= sizeof(ra_cfar_path)
        || (size_t)rv_cfar_len >= sizeof(rv_cfar_path)
        || (size_t)ra_clusters_len >= sizeof(ra_clusters_path)
        || (size_t)rv_clusters_len >= sizeof(rv_clusters_path)) {
        fprintf(stderr, "output prefix is too long\n");
        return -1;
    }

    const float *ra_map = output_maps;
    const float *rv_maps = output_maps + (size_t)N_RANGE * N_ANGLE;
    const float *rv_map = rv_maps;
    float *rv_slice = malloc(2u * (size_t)N_RANGE * N_VEL * sizeof(*rv_slice));
    uint8_t *ra_cfar = calloc((size_t)N_RANGE * N_ANGLE, sizeof(*ra_cfar));
    uint8_t *rv_cfar = calloc((size_t)N_RANGE * N_VEL, sizeof(*rv_cfar));
    uint8_t *ra_clusters = calloc((size_t)N_RANGE * N_ANGLE, sizeof(*ra_clusters));
    uint8_t *rv_clusters = calloc((size_t)N_RANGE * N_VEL, sizeof(*rv_clusters));
    if (rv_slice == NULL || ra_cfar == NULL || rv_cfar == NULL || ra_clusters == NULL || rv_clusters == NULL) {
        fprintf(stderr, "failed to allocate output buffers\n");
        free(rv_slice);
        free(ra_cfar);
        free(rv_cfar);
        free(ra_clusters);
        free(rv_clusters);
        return -1;
    }

    for (size_t range = 0; range < N_RANGE; ++range) {
        for (size_t vel = 0; vel < N_VEL; ++vel) {
            for (size_t slice = 0; slice < 2u; ++slice) {
                rv_slice[(range * N_VEL + vel) * 2u + slice] =
                    rv_maps[(slice * N_RANGE + range) * N_VEL + vel];
            }
        }
    }

    cfar2(ra_map, ra_cfar, N_RANGE, N_ANGLE);
    cfar2(rv_map, rv_cfar, N_RANGE, N_VEL);
    cluster_detections(ra_cfar, ra_map, ra_clusters, N_RANGE, N_ANGLE, CLUSTER_RADIUS, CLUSTER_MIN_SIZE);
    cluster_detections(rv_cfar, rv_map, rv_clusters, N_RANGE, N_VEL, CLUSTER_RADIUS, CLUSTER_MIN_SIZE);

    size_t ra_shape[] = {N_RANGE, N_ANGLE};
    size_t rv_map_shape[] = {N_RANGE, N_VEL};
    size_t rv_shape[] = {N_RANGE, N_VEL, 2u};
    int status = 0;
    if (write_npy_float32(ra_path, ra_map, ra_shape, 2u) != 0) {
        status = -1;
    } else if (write_npy_float32(rv_map_path, rv_map, rv_map_shape, 2u) != 0) {
        status = -1;
    } else if (write_npy_float32(rv_path, rv_slice, rv_shape, 3u) != 0) {
        status = -1;
    } else if (write_npy_bool(ra_cfar_path, ra_cfar, ra_shape, 2u) != 0) {
        status = -1;
    } else if (write_npy_bool(rv_cfar_path, rv_cfar, rv_map_shape, 2u) != 0) {
        status = -1;
    } else if (write_npy_bool(ra_clusters_path, ra_clusters, ra_shape, 2u) != 0) {
        status = -1;
    } else if (write_npy_bool(rv_clusters_path, rv_clusters, rv_map_shape, 2u) != 0) {
        status = -1;
    }

    free(rv_slice);
    free(ra_cfar);
    free(rv_cfar);
    free(ra_clusters);
    free(rv_clusters);
    return status;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s input_adc_complex64.bin output_prefix\n", argv[0]);
        fprintf(stderr, "outputs: output_prefix_ra_map.npy, output_prefix_rv_map.npy, output_prefix_rv_slice.npy, output_prefix_ra_cfar.npy, output_prefix_rv_cfar.npy, output_prefix_ra_clusters.npy, output_prefix_rv_clusters.npy\n");
        return EXIT_FAILURE;
    }

    const size_t adc_count = (size_t)N_SAMPLE * N_RX * N_CHIRP;
    const size_t output_count = (size_t)N_RANGE * N_ANGLE + 2u * (size_t)N_RANGE * N_VEL;
    complex_f32_t *adc_data = malloc(adc_count * sizeof(*adc_data));
    float *output_maps = calloc(output_count, sizeof(*output_maps));
    if (adc_data == NULL || output_maps == NULL) {
        fprintf(stderr, "failed to allocate processing buffers\n");
        free(adc_data);
        free(output_maps);
        return EXIT_FAILURE;
    }

    int status = EXIT_SUCCESS;
    if (read_complex64_file(argv[1], adc_data, adc_count) != 0) {
        status = EXIT_FAILURE;
    } else if (process_radar(adc_data, output_maps) != 0) {
        status = EXIT_FAILURE;
    } else if (write_outputs(argv[2], output_maps) != 0) {
        status = EXIT_FAILURE;
    }

    free(adc_data);
    free(output_maps);
    return status;
}
