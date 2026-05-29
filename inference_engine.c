/* 
   Compile:
     cl /O2 /arch:AVX2 /nologo /Fe:inference_engine.exe inference_engine.c cJSON.c
   Run:
     .\inference_engine.exe weights.json archive\train_FD001.txt
*/

#define WIN32_LEAN_AND_MEAN // windows.h without rarely used APIs, faster compile time
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>// for memcpy
#include <math.h>// expf float, expf(x)-> euler^x  ; exp(X) double, expf(X) float  
#include "cJSON.h"


#define NUM_THREADS     4
#define ANOMALY_THRESH  0.5f // threshold for anomaly detection
#define MAX_ROWS    25000 // maximum rows to read from the file, needed for memory allocation

// model architecture and weights
int window_size, n_sensors, input_dim, l1_dim, l2_dim, l3_dim;
float *feat_min, *feat_max;
float *W1, *b1, *W2, *b2, *W3, *b3;

float  (*g_windows)[200] = NULL;/* Input data (block) -> adapted for sliding window --> (max 200 floats, 4 sensors x 50 window size)
                                    float (*g_windows)[200]; --> g_windows + 1 jumps 200 elements at once
                                    (arrays of 200-floats "jumps window by window"),
                                    equivalent to float *p; --> p + 200 */

int    *g_labels         = NULL;
int     g_total_rows     = 0;// total number of rows to be processed as sliding windows


volatile LONG g_anomaly_count = 0;// Atomically incremented via InterlockedIncrement for cross-thread synchronization
volatile LONG g_processed     = 0;// Using InterlockedIncrement instead of critical section enter/leave to avoid context switch overhead

// thread synchronization
CRITICAL_SECTION  queue_lock;
volatile int      g_next_row = 0;// The next row to be processed


/*
    Z = vec(A) * W^T + b
    Neuron formula: out[j] = b[j] + Σ (vec[k] * W_flatten[j * m + k])

    For each j-th neuron, m weights are read starting from the (j * m) index.
    b shape= [n] -> bias 
    out (Z) shape= [n] -> output vector

    matmul_vec processes 1 input vector when called.
    -> forward_pass pushes the vector to the end of the neural network
    -> All data is processed by calling forward_pass in a loop n times in main.

*/
static void matmul_vec(const float *vec, const float *W, const float *b, float *out, int m, int n){
    for (int j = 0; j < n; j++) {
        float sum = b[j];
        
    /* 
            W.shape = [m,n] -> W_flatten.shape = [m*n]
        Since W is a flat array,
        It is not (n,m) because transpose will be taken, meaning m weights for each neuron come consecutively.
        Starting address of the j-th neuron's weights = W + j * m
        (j * m) navigates to the first weight of the j-th neuron, then m weights are read.
        Thus, all weights of a neuron are kept contiguously, increasing cache performance.
    */
        const float *row = W + j * m; 
        
        for (int k = 0; k < m; k++)
            sum += vec[k] * row[k];
            
        out[j] = sum;
    }
}

static void relu_vec(float *v, int n) {
    for (int i = 0; i < n; i++) 
        if (v[i] < 0.0f) v[i] = 0.0f;
}


float forward_pass(const float *input) {
    float h1[128], h2[64], h3[1];

    //Layer 1
    matmul_vec(input, W1, b1, h1, input_dim, l1_dim);
    relu_vec(h1, l1_dim);

    // 2
    matmul_vec(h1, W2, b2, h2, l1_dim, l2_dim);
    relu_vec(h2, l2_dim);

    // 3
    matmul_vec(h2, W3, b3, h3, l2_dim, l3_dim);
    h3[0] = 1.0f / (1.0f + expf(-h3[0]));

    return h3[0];
}

// weights in json_doc are linked list of arrays,we need to flatten it to load into our weight arrays(compile time)
// If the dimensions of the loaded weights are known beforehand,can be defined as arr[][] without flattening
static void json_to_flat(cJSON *arr, float *dest) {
    int idx = 0;
    cJSON *row;
    cJSON_ArrayForEach(row, arr) {
        if (cJSON_IsArray(row)) {
            cJSON *val;
            cJSON_ArrayForEach(val, row) dest[idx++] = (float)val->valuedouble;
        } else if (cJSON_IsNumber(row)) {
            dest[idx++] = (float)row->valuedouble;
        }
    }
}

static int load_weights(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("Error: %s not found!\n", path); return 0; }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    rewind(f);
    char *buf = (char*)malloc(len + 1);
    fread(buf, 1, len, f);
    fclose(f);
    buf[len] = '\0';

    cJSON *js = cJSON_Parse(buf);
    free(buf);
    if (!js) { printf("JSON parse error!\n"); return 0; }

    cJSON *arch = cJSON_GetObjectItem(js, "architecture");
    input_dim   = cJSON_GetArrayItem(arch, 0)->valueint;
    l1_dim      = cJSON_GetArrayItem(arch, 1)->valueint;
    l2_dim      = cJSON_GetArrayItem(arch, 2)->valueint;
    l3_dim      = cJSON_GetArrayItem(arch, 3)->valueint;
    window_size = cJSON_GetObjectItem(js, "window_size")->valueint;
    n_sensors   = cJSON_GetObjectItem(js, "n_sensors")->valueint;

    feat_min = (float*)malloc(n_sensors * sizeof(float));
    feat_max = (float*)malloc(n_sensors * sizeof(float));
    W1 = (float*)malloc(l1_dim * input_dim * sizeof(float));
    b1 = (float*)malloc(l1_dim * sizeof(float));
    W2 = (float*)malloc(l2_dim * l1_dim * sizeof(float));
    b2 = (float*)malloc(l2_dim * sizeof(float));
    W3 = (float*)malloc(l3_dim * l2_dim * sizeof(float));
    b3 = (float*)malloc(l3_dim * sizeof(float));

    json_to_flat(cJSON_GetObjectItem(js, "feat_min"), feat_min);
    json_to_flat(cJSON_GetObjectItem(js, "feat_max"), feat_max);

    cJSON *layers = cJSON_GetObjectItem(js, "layers");
    json_to_flat(cJSON_GetObjectItem(cJSON_GetArrayItem(layers,0),"W"), W1);
    json_to_flat(cJSON_GetObjectItem(cJSON_GetArrayItem(layers,0),"b"), b1);
    json_to_flat(cJSON_GetObjectItem(cJSON_GetArrayItem(layers,1),"W"), W2);
    json_to_flat(cJSON_GetObjectItem(cJSON_GetArrayItem(layers,1),"b"), b2);
    json_to_flat(cJSON_GetObjectItem(cJSON_GetArrayItem(layers,2),"W"), W3);
    json_to_flat(cJSON_GetObjectItem(cJSON_GetArrayItem(layers,2),"b"), b3);

    cJSON_Delete(js);
    printf("Architecture: [%d, %d, %d, %d]  window=%d  sensors=%d\n",
           input_dim, l1_dim, l2_dim, l3_dim, window_size, n_sensors);
    return 1;
}

// Data + sliding window   
static int load_data(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { printf("Error: %s not found!\n", path); return 0; }

    // sensor indices used in the py
    static const int COL[] = {6, 7, 8, 11};

    
    float raw[MAX_ROWS][4];
    int units[MAX_ROWS];//Prevents boundary leaks, equivalent to the mask logic in python
    int row_count = 0;
    char line[1024];

    while (fgets(line, sizeof(line), f) && row_count < MAX_ROWS) {
        float vals[30]; int vi = 0;
        char *p = line;
        while (*p && vi < 30) {
            while (*p == ' ' || *p == '\t') p++;
            if (*p == '\0' || *p == '\n') break;
            vals[vi++] = strtof(p, &p);/* strtof converts the initial part of the string to a float
                                             and updates the pointer after the number,
                                             allowing sequential parsing of the line*/
        }
        if (vi < 12) continue;
        units[row_count] = (int)vals[0];

        for (int s = 0; s < 4; s++) {
            float v = vals[COL[s]];
            raw[row_count][s] = (v - feat_min[s]) / (feat_max[s] - feat_min[s] + 1e-8f);
        }
        row_count++;
    }
    fclose(f);

    /* Sliding window: create a 200-float input for each valid window */
    int max_windows = row_count - window_size + 1;
    /* number of windows = total rows - window size + 1 
       (to fit the exact last window)
       Off-by-one check:
       If the window size is 50, 51 windows can be created from 100 rows
       (1-50, 2-51, ..., 51-100).
    */
    if (max_windows <= 0) { printf("Error: Not enough rows!\n"); return 0; }

    g_windows   = (float(*)[200])malloc(max_windows * 200 * sizeof(float));
    g_labels    = (int*)malloc(max_windows * sizeof(int));
    g_total_rows = 0;

    for (int i = 0; i < max_windows; i++) {
        //Boundary leak check: the first and last row in the window must have the same Engine ID
        if (units[i] != units[i + window_size - 1]) continue;
        
        for (int w = 0; w < window_size; w++)
            memcpy(&g_windows[i][w * n_sensors], raw[i + w], n_sensors * sizeof(float));
            //copying the data block from raw to g_windows using memcpy instead of item-by-item assignment
        g_labels[i] = 0;   /*not real labels, simulated */
        g_total_rows++;
    }

    printf("[Data] %d sliding windows.\n\n", g_total_rows);
    return 1;
}

// single thread
static double run_single_thread(void) {
    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);

    long anomalies = 0;
    for (int i = 0; i < g_total_rows; i++) {
        float score = forward_pass(g_windows[i]);
        if (score >= ANOMALY_THRESH) anomalies++;
    }

    QueryPerformanceCounter(&t1);
    double elapsed = (double)(t1.QuadPart - t0.QuadPart) / freq.QuadPart;

    printf("  Anomaly    : %ld / %d  (%.1f%%)\n", anomalies, g_total_rows,
           100.0 * anomalies / g_total_rows);
    printf("  Time       : %.4f s\n", elapsed);
    printf("  Throughput : %.0f packet/s\n", g_total_rows / elapsed);
    printf("  Avg. latency: %.4f ms/packet\n\n", elapsed * 1000.0 / g_total_rows);
    return elapsed;
}

// worker thread
DWORD WINAPI worker_thread(LPVOID arg) {
    (void)arg;
    while (1) {
        EnterCriticalSection(&queue_lock);
        int row = g_next_row;
        if (row < g_total_rows) g_next_row++;
        LeaveCriticalSection(&queue_lock);

        if (row >= g_total_rows) break;

        float score = forward_pass(g_windows[row]);
        InterlockedIncrement(&g_processed);
        if (score >= ANOMALY_THRESH)
            InterlockedIncrement(&g_anomaly_count);
    }
    return 0;
}

// multi thread
static double run_multi_thread(int n_threads) {
    g_next_row      = 0;
    g_anomaly_count = 0;
    g_processed     = 0;

    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);

    HANDLE threads[16];
    for (int i = 0; i < n_threads; i++)
        threads[i] = CreateThread(NULL, 0, worker_thread, NULL, 0, NULL);
    WaitForMultipleObjects(n_threads, threads, TRUE, INFINITE);
    for (int i = 0; i < n_threads; i++) CloseHandle(threads[i]);

    QueryPerformanceCounter(&t1);
    double elapsed = (double)(t1.QuadPart - t0.QuadPart) / freq.QuadPart;

    printf("  Anomaly    : %ld / %d  (%.1f%%)\n",
           (long)g_anomaly_count, g_total_rows,
           100.0 * g_anomaly_count / g_total_rows);
    printf("  Time       : %.4f s\n", elapsed);
    printf("  Throughput : %.0f packet/s\n", g_total_rows / elapsed);
    printf("  Avg. latency: %.4f ms/packet\n\n", elapsed * 1000.0 / g_total_rows);
    return elapsed;
}


int main(int argc, char *argv[]) {
        
    const char *weights = (argc > 1) ? argv[1] : "weights.json";
    const char *data    = (argc > 2) ? argv[2] : "archive\\train_FD001.txt";

    if (!load_weights(weights)) return 1;
    if (!load_data(data))       return 1;

    InitializeCriticalSection(&queue_lock);

    printf(">>> [1] Single Thread\n");
    double t_single = run_single_thread();

    printf(">>> [2] Multi Thread (%d threads)\n", NUM_THREADS);
    double t_multi = run_multi_thread(NUM_THREADS);

    double speedup = t_single / t_multi;
    printf("====================================\n");
    printf("BENCHMARK REPORT\n");
    printf("====================================\n");
    printf("    Single thread time : %.4f s\n", t_single);
    printf("    Multi thread time  : %.4f s\n", t_multi);
    printf("    Speedup            : %.2fx\n",   speedup);
    printf("====================================\n\n");

    
    DeleteCriticalSection(&queue_lock);
    free(feat_min); free(feat_max);
    free(W1); free(b1); free(W2); free(b2); free(W3); free(b3);
    free(g_windows); free(g_labels);

    printf("END\n");
    return 0;
}