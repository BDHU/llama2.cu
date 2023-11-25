#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdint.h>
#include <assert.h>

#include "cuda.h"
#include "cuda_runtime.h"

#ifndef checkCudaErrors
#define checkCudaErrors(err) __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors( cudaError err, const char *file, const int line )
{
    if (cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}
#endif

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens
// ----------------------------------------------------------------------------

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char **vocab;
    float *vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling
// ----------------------------------------------------------------------------

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex *probindex;   // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;


// ----------------------------------------------------------------------------
// Transformer-related structs
// ----------------------------------------------------------------------------

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // ffn layer dimension
    int n_layers; // number of transformer layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of k/v heads
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int max_seq_len; // maximum sequence length to generate
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;

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
    TransformerWeights weights; // model weights
    RunState state; // buffer required to store intermediate values during forward pass
    int fd; // file descriptor required for memory mapping, explained later TODO
    float *data; // data pointer, TODO
    uint64_t file_size; // size of the model checkpoint file in bytes
} Transformer;

// ----------------------------------------------------------------------------
// Sampler
// ----------------------------------------------------------------------------

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = (ProbIndex *)malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler *sampler) {
    free(sampler->probindex);
}

// ----------------------------------------------------------------------------
// Tokenizer
// ----------------------------------------------------------------------------

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer *t, char *tokenizer_path, int vocab_size) {
    // should've written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // allocate space to hold the scores and the strings
    t->vocab = (char **)malloc(vocab_size * sizeof(char *));
    t->vocab_scores = (float *)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) {fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE);}
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) {fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
    int len = 0;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) {fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) {fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) {fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

void free_tokenizer(Tokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

int str_lookup(char *str, const TokenIndex *sorted_vocab, size_t vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    const TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = (TokenIndex *)bsearch((const void *)&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) {fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE);}

    printf("vocab_size is %d\n", t->vocab_size);
    if (t->sorted_vocab == NULL) {
        // lazily alloc and sort the vocabulary
        checkCudaErrors(cudaMallocManaged((void **)&t->sorted_vocab, t->vocab_size * sizeof(TokenIndex)));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char *str_buffer = (char *)malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    printf("max token len: %d\n", t->max_token_length);
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) {tokens[(*n_tokens)++] = 1;}

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        printf("dummy prefix %d\n", dummy_prefix);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    const int max_num_utf8_byte = 4;
    for (char *c = text; *c != '\0'; c++) {
        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0'; // write null char in case we have an ASCII character

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < max_num_utf8_byte) {
            continue;
        }

        // now str_buffer should contain a full UTF-8 character
        // c+1 is not a continuation byte, so we read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i = 0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }
}

// ----------------------------------------------------------------------------
// Transformer
// ----------------------------------------------------------------------------

void alloc_run_state(RunState *s, Config config) {
    int kv_dim = config.dim * config.n_kv_heads / config.n_heads;
    checkCudaErrors(cudaMallocManaged((void **)&s->x, config.dim * sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **)&s->xb, config.dim * sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **)&s->xb2, config.dim * sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **)&s->hb, config.hidden_dim * sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **)&s->hb2, config.hidden_dim * sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **)&s->q, config.dim * sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **)&s->key_cache, config.n_layers * config.max_seq_len * kv_dim * sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **)&s->value_cache, config.n_layers * config.max_seq_len * kv_dim * sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **)&s->att, config.n_heads * config.max_seq_len * sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **)&s->logits, config.vocab_size * sizeof(float)));
}

void free_run_state(RunState *s) {
    checkCudaErrors(cudaFree((void *)s->x));
    checkCudaErrors(cudaFree((void *)s->xb));
    checkCudaErrors(cudaFree((void *)s->xb2));
    checkCudaErrors(cudaFree((void *)s->hb));
    checkCudaErrors(cudaFree((void *)s->hb2));
    checkCudaErrors(cudaFree((void *)s->q));
    checkCudaErrors(cudaFree((void *)s->key_cache));
    checkCudaErrors(cudaFree((void *)s->value_cache));
    checkCudaErrors(cudaFree((void *)s->att));
    checkCudaErrors(cudaFree((void *)s->logits));
}

void memory_map_weights(TransformerWeights *w, Config config, float *ptr, int shared_weights) {
    int head_size = config.dim / config.n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = config.n_layers;
    w->token_embedding_table = ptr;
    ptr += config.vocab_size * config.dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * config.dim;
    w->wq = ptr;
    ptr += n_layers * config.dim * (config.n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * config.dim * (config.n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * config.dim * (config.n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (config.n_heads * head_size) * config.dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * config.dim;
    w->w1 = ptr;
    ptr += n_layers * config.dim * config.hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * config.hidden_dim * config.dim;
    w->w3 = ptr;
    ptr += n_layers * config.dim * config.hidden_dim;
    w->rms_final_weight = ptr;
    ptr += config.dim;
    ptr += config.max_seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += config.max_seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void read_checkpoint(char *checkpoint, Transformer *transformer) {
    Config *config = &(transformer->config);
    FILE *file = fopen(checkpoint, "rb");   // "rb" for openning binary file
    if (file == NULL) {fprintf(stderr, "Failed to open checkpoint file %s\n", checkpoint); exit(EXIT_FAILURE);}
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) {
        fprintf(stderr, "Read config from checkpoint %s failed due to an error or EOF\n", checkpoint); exit(EXIT_FAILURE);
    }
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    transformer->file_size = ftell(file);
    fclose(file);
    // memory map the Transformer weights into the data pointer
    transformer->fd = open(checkpoint, O_RDONLY);
    if (transformer->fd == -1) { fprintf(stderr, "open checkpoint failed!\n"); exit(EXIT_FAILURE); }
    checkCudaErrors(cudaMallocManaged((void **)&transformer->data, transformer->file_size+1));
    float *weights_ptr = transformer->data + sizeof(Config) / sizeof(float);
    memory_map_weights(&transformer->weights, transformer->config, weights_ptr, shared_weights);
    if (transformer->fd != -1) {close(transformer->fd);}
}

void build_transformer(Transformer *transformer, char *checkpoint_path) {
    // read in Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, transformer);
    // allocate the RunState buffers
    alloc_run_state(&transformer->state, transformer->config);
}

void free_transformer(Transformer* t) {
    // close the memory mapping
    checkCudaErrors(cudaFree(t->data));
    // free the RunState buffers
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// generation loop
// ----------------------------------------------------------------------------

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) {prompt = empty_prompt;}

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int *prompt_tokens = NULL;
    checkCudaErrors(cudaMallocManaged((void **)&prompt_tokens, sizeof(int) * strlen(prompt)+3)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
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
    char *tokenizer_path = (char *)"tokenizer.bin";
    float temperature = 1.0f;   // higher temperature leads to more creative generations
    float topp = 0.9f;  // nucleas sampling.
    int steps = 256;
    char *prompt = NULL;
    unsigned long long rng_seed = 0;
    char *mode = (char *)"generate";    // generate|chat
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
    if (steps == 0 || steps > transformer.config.max_seq_len) {steps = transformer.config.max_seq_len;}

    // build the tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // run!
    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        help_msg();
    }

    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);

    return 0;
}
