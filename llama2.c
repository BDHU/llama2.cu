#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdint.h>

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // ffn layer dimension
    int n_layers; // number of transformer layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of k/v heads
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int max_seq_len; // maximum sequence length to generate
} Config;

// RunState definition
typedef struct {
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but insize a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // bufffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for the scores/attention avlues (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float *key_cache; // (layer, seq_len, dim)
    float *value_cache; // (layer , seq_len, dim)
} RunState;

// Transformer definition
typedef struct {
    Config config;
    // TransformerWeights weights; // model weights
    RunState state; // buffer required to store intermediate values during forward pass
    int fd; // file descriptor required for memory mapping, explained later TODO
    float *data; //memory mapped data pointer, TODO
    uint64_t file_size; // size of the model checkpoint file in bytes
} Transformer;

void read_config(char *checkpoint, Transformer *transformer) {
    FILE *file = fopen(checkpoint, "rb");   // "rb" for openning binary file
    if (file == NULL) {fprintf(stderr, "Failed to checkpoint file %s\n", checkpoint); exit(EXIT_FAILURE);}
    // read in the config header
    if (fread(checkpoint, sizeof(Config), 1, file) != 1) {fprintf(stderr, "Read from checkpoint %s failed due to an error or EOF\n", checkpoint); exit(EXIT_FAILURE);}
}

void build_transformer(Transformer *transformer, char *checkpoint_path) {
    // read in Config and the Weights from the checkpoint
    read_config(checkpoint_path, transformer);
}

// long arguments
static struct option long_options[] = {
    {"model", required_argument, NULL, 'm'},
    {"tokenizer", optional_argument, NULL, 'z'},
    {"temperature", optional_argument, NULL, 't'},
    {"topp", optional_argument, NULL, 'p'},
    {"seed", optional_argument, NULL, 's'},
    {"step", optional_argument, NULL, 'n'},
    {"prompt", required_argument, NULL, 'i'},
    {"mode", optional_argument, NULL, 'M'},
    {"system-prompt", optional_argument, NULL, 'y'},
    {"ngl", optional_argument, NULL, 'l'},
    {"stream", no_argument, NULL, 'S'},
    {"help", optional_argument, NULL, 'h'},
};

void help_msg() {
    fprintf(stderr, "Usage: run main <mode_checkpoint> [options]\n");
    fprintf(stderr, "Example: ./main -i \"Tell me a story\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -m, --model <string> model checkpoint path\n");
    fprintf(stderr, "  -z, --tokenizer <string> tokenizer path\n");
    fprintf(stderr, "  -t, --temperature <float> temperatutre in [0,inf], default to 1.0\n");
    fprintf(stderr, "  -p, --topp <float> p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s, --seed <int> random seed, default time(NULL)\n");
    fprintf(stderr, "  -n, --step <int> number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i, --prompt <string> input prompt\n");
    fprintf(stderr, "  -M, --mode <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y, --system_prompt <string> (optional) system prompt in chat mode\n");
    fprintf(stderr, "  -l, --ngl <int> (optional) number of layers offload to CPU\n");
    fprintf(stderr, "  -S, --stream (optional) whether to stream outputs\n");
    fprintf(stderr, "  -y, --system_prompt <string> (optional) system prompt in chat mode\n");
    fprintf(stderr, "  -h, --help print this message\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    // Parameter setup
    char *checkpoint_path = NULL;   // e.g. models/llama2-7b.bin
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;   // higher temperature leads to more creative generations
    float topp = 0.9f;  // nucleas sampling.
    int steps = 256;
    char *prompt = NULL;
    unsigned long long rng_seed = 0;
    char *mode = "generate";    // generate|chat
    char *system_prompt = NULL;     // optional system prompt used in chat mode
    bool stream = false;
    int layers = -1;    // layers to offload to CPU

    // parse arguments
    int opt = 0;
    while ((opt = getopt_long(argc, argv, "m:z:t:p:s:n:i:M:y:l:Sh",
                    long_options, NULL)) != -1) {
        switch (opt) {
            case 'm':
                checkpoint_path = optarg;
                printf("checkpoint_path: %s\n", checkpoint_path);
                break;
            case 'z':
                tokenizer_path = optarg;
                printf("tokenizer path: %s\n", tokenizer_path);
                break;
            case 't':
                temperature = atoi(optarg);
                printf("temperature is %f\n", temperature);
                break;
            case 'p':
                topp = atoi(optarg);
                printf("topp is %f\n", topp);
                break;
            case 's':
                rng_seed = atoi(optarg);
                printf("rng seed %llu\n", rng_seed);
                break;
            case 'n':
                steps = atoi(optarg);
                printf("step is %d\n", steps);
                break;
            case 'i':
                prompt = optarg;
                break;
            case 'M':
                mode = optarg;
                break;
            case 'y':
                system_prompt = optarg;
                break;
            case 'l':
                layers = atoi(optarg);
                break;
            case 'S':
                stream = true;
                printf("stream is: %d\n", stream);
                break;
            case 'h':
                help_msg();
                break;
            case '?':
                help_msg();
                break;
            default:
                help_msg();
                break;
        }
    }

    // parameter validation/correction
    if (rng_seed <= 0) {rng_seed = (unsigned long long)time(NULL);}
    if (temperature < 0.0) {temperature = 0.0f;}
    if (topp < 0.0 || 1.0 <= topp) {topp = 0.9f;}
    if (steps < 0) {steps = 0;}

    // build Transformer from given model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);

    return 0;
}
