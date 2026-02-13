// Test suite for RPC server-to-server (peer) tensor transfer feature.
//
// Tests covered:
//   Test 1: Peer registration round-trip
//   Test 2: Direct push (data integrity)
//   Test 4: Fallback on failed direct transfer (wrong endpoint)
//
// These tests spawn rpc-server processes and communicate with them at the
// protocol level using raw sockets, verifying the new RPC commands:
//   RPC_CMD_REGISTER_PEER, RPC_CMD_PUSH_TENSOR_TO_PEER

#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <csignal>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "ws2_32.lib")
   typedef SOCKET sockfd_t;
#  define CLOSESOCK closesocket
#else
#  include <arpa/inet.h>
#  include <netdb.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <sys/wait.h>
#  include <unistd.h>
#  include <fcntl.h>
#  include <signal.h>
   typedef int sockfd_t;
#  define CLOSESOCK close
#  define INVALID_SOCKET (-1)
#endif

// ---- Protocol constants (must match ggml-rpc.h / ggml-rpc.cpp) ----

#define RPC_PROTO_MAJOR_VERSION    3
#define RPC_PROTO_MINOR_VERSION    7
#define RPC_PROTO_PATCH_VERSION    0

#define GGML_MAX_DIMS      4
#define GGML_MAX_SRC       10
#define GGML_MAX_OP_PARAMS 64
#define GGML_MAX_NAME      64

// ---- RPC command IDs ----
enum rpc_cmd {
    RPC_CMD_ALLOC_BUFFER = 0,
    RPC_CMD_GET_ALIGNMENT,
    RPC_CMD_GET_MAX_SIZE,
    RPC_CMD_BUFFER_GET_BASE,
    RPC_CMD_FREE_BUFFER,
    RPC_CMD_BUFFER_CLEAR,
    RPC_CMD_SET_TENSOR,
    RPC_CMD_SET_TENSOR_HASH,
    RPC_CMD_GET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    RPC_CMD_GRAPH_COMPUTE,
    RPC_CMD_GET_DEVICE_MEMORY,
    RPC_CMD_INIT_TENSOR,
    RPC_CMD_GET_ALLOC_SIZE,
    RPC_CMD_HELLO,
    RPC_CMD_DEVICE_COUNT,
    RPC_CMD_GRAPH_RECOMPUTE,
    RPC_CMD_REGISTER_PEER,
    RPC_CMD_PUSH_TENSOR_TO_PEER,
    RPC_CMD_PEER_TENSOR_DATA,
    RPC_CMD_COUNT,
};

// ---- Packed protocol structures ----
#pragma pack(push, 1)

struct rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    char     name[GGML_MAX_NAME];
    char     padding[4];
};

struct rpc_msg_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
};

struct rpc_msg_alloc_buffer_req {
    uint32_t device;
    uint64_t size;
};

struct rpc_msg_alloc_buffer_rsp {
    uint64_t remote_ptr;
    uint64_t remote_size;
};

struct rpc_msg_buffer_get_base_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_get_base_rsp {
    uint64_t base_ptr;
};

struct rpc_msg_free_buffer_req {
    uint64_t remote_ptr;
};

struct rpc_msg_get_tensor_req {
    rpc_tensor tensor;
    uint64_t   offset;
    uint64_t   size;
};

struct rpc_msg_register_peer_req {
    uint32_t peer_id;
    char     endpoint[128];
};

struct rpc_msg_register_peer_rsp {
    uint8_t result;
};

struct rpc_msg_push_tensor_to_peer_req {
    uint32_t   peer_id;
    rpc_tensor src;
    rpc_tensor dst;
    uint64_t   offset;
    uint64_t   size;
};

struct rpc_msg_push_tensor_to_peer_rsp {
    uint8_t result;
};

#pragma pack(pop)

// ---- Low-level socket helpers ----

static bool send_all(sockfd_t fd, const void * data, size_t size) {
    const char * ptr = (const char *)data;
    while (size > 0) {
        ssize_t n = send(fd, ptr, size, 0);
        if (n <= 0) return false;
        ptr  += n;
        size -= n;
    }
    return true;
}

static bool recv_all(sockfd_t fd, void * data, size_t size) {
    char * ptr = (char *)data;
    while (size > 0) {
        ssize_t n = recv(fd, ptr, size, 0);
        if (n <= 0) return false;
        ptr  += n;
        size -= n;
    }
    return true;
}

// Send an RPC command with request data and receive a fixed-size response.
// Wire format:
//   Request:  | cmd (1 byte) | request_size (8 bytes) | request_data |
//   Response: | response_size (8 bytes) | response_data |
static bool send_rpc_cmd(sockfd_t fd, uint8_t cmd,
                         const void * req, size_t req_size,
                         void * rsp, size_t rsp_size) {
    if (!send_all(fd, &cmd, 1)) return false;
    uint64_t sz = req_size;
    if (!send_all(fd, &sz, sizeof(sz))) return false;
    if (req_size > 0 && !send_all(fd, req, req_size)) return false;

    if (rsp != nullptr && rsp_size > 0) {
        uint64_t out_size = 0;
        if (!recv_all(fd, &out_size, sizeof(out_size))) return false;
        if (out_size != rsp_size) {
            fprintf(stderr, "send_rpc_cmd: expected rsp_size=%zu got %" PRIu64 "\n",
                    rsp_size, out_size);
            return false;
        }
        if (!recv_all(fd, rsp, rsp_size)) return false;
    }
    return true;
}

// Send an RPC command with request data, no response expected.
static bool send_rpc_cmd_no_rsp(sockfd_t fd, uint8_t cmd,
                                const void * req, size_t req_size) {
    if (!send_all(fd, &cmd, 1)) return false;
    uint64_t sz = req_size;
    if (!send_all(fd, &sz, sizeof(sz))) return false;
    if (req_size > 0 && !send_all(fd, req, req_size)) return false;
    return true;
}

// Send an RPC command with variable-length request, variable-length response.
static bool send_rpc_cmd_var(sockfd_t fd, uint8_t cmd,
                             const void * req, size_t req_size,
                             std::vector<uint8_t> & rsp) {
    if (!send_all(fd, &cmd, 1)) return false;
    uint64_t sz = req_size;
    if (!send_all(fd, &sz, sizeof(sz))) return false;
    if (req_size > 0 && !send_all(fd, req, req_size)) return false;

    uint64_t out_size = 0;
    if (!recv_all(fd, &out_size, sizeof(out_size))) return false;
    rsp.resize(out_size);
    if (out_size > 0 && !recv_all(fd, rsp.data(), out_size)) return false;
    return true;
}

// ---- Connect to a server ----

static sockfd_t connect_to(const char * host, int port) {
    struct addrinfo hints = {};
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", port);
    struct addrinfo * res = nullptr;
    if (getaddrinfo(host, port_str, &hints, &res) != 0 || res == nullptr) {
        return INVALID_SOCKET;
    }
    sockfd_t fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd == INVALID_SOCKET) {
        freeaddrinfo(res);
        return INVALID_SOCKET;
    }
    if (connect(fd, res->ai_addr, res->ai_addrlen) != 0) {
        CLOSESOCK(fd);
        freeaddrinfo(res);
        return INVALID_SOCKET;
    }
    freeaddrinfo(res);

    // Disable Nagle
    int flag = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char *)&flag, sizeof(flag));
    return fd;
}

// ---- HELLO handshake ----

static bool do_hello(sockfd_t fd, rpc_msg_hello_rsp * rsp = nullptr) {
    rpc_msg_hello_rsp response;
    if (!send_rpc_cmd(fd, RPC_CMD_HELLO, nullptr, 0, &response, sizeof(response))) {
        return false;
    }
    if (response.major != RPC_PROTO_MAJOR_VERSION) {
        fprintf(stderr, "do_hello: version mismatch %d.%d.%d\n",
                response.major, response.minor, response.patch);
        return false;
    }
    if (rsp) *rsp = response;
    return true;
}

// ---- Server process management ----

struct test_rpc_server {
#ifdef _WIN32
    PROCESS_INFORMATION pi = {};
#else
    pid_t pid = 0;
#endif
    int port = 0;
    std::string endpoint;

    // Start an rpc-server on the given port.
    // rpc_server_path: path to the rpc-server binary.
    bool start(const std::string & rpc_server_path, int p) {
        port = p;
        endpoint = "127.0.0.1:" + std::to_string(port);
#ifdef _WIN32
        STARTUPINFOA si = {};
        si.cb = sizeof(si);
        std::string cmd = rpc_server_path + " -H 127.0.0.1 -p " + std::to_string(port);
        if (!CreateProcessA(nullptr, (LPSTR)cmd.c_str(), nullptr, nullptr,
                            FALSE, 0, nullptr, nullptr, &si, &pi)) {
            fprintf(stderr, "Failed to start rpc-server on port %d\n", port);
            return false;
        }
#else
        pid = fork();
        if (pid < 0) {
            perror("fork");
            return false;
        }
        if (pid == 0) {
            // child
            // Redirect stdout/stderr to /dev/null to reduce noise
            int devnull = open("/dev/null", O_WRONLY);
            if (devnull >= 0) {
                dup2(devnull, STDOUT_FILENO);
                dup2(devnull, STDERR_FILENO);
                close(devnull);
            }
            execl(rpc_server_path.c_str(), "rpc-server",
                  "-H", "127.0.0.1", "-p", std::to_string(port).c_str(),
                  (char *)nullptr);
            perror("execl");
            _exit(1);
        }
#endif
        // Wait for the server to accept connections
        for (int i = 0; i < 100; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            sockfd_t fd = connect_to("127.0.0.1", port);
            if (fd != INVALID_SOCKET) {
                CLOSESOCK(fd);
                return true;
            }
        }
        fprintf(stderr, "rpc-server on port %d failed to become ready\n", port);
        stop();
        return false;
    }

    void stop() {
#ifdef _WIN32
        if (pi.hProcess) {
            TerminateProcess(pi.hProcess, 0);
            WaitForSingleObject(pi.hProcess, 5000);
            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
            pi = {};
        }
#else
        if (pid > 0) {
            kill(pid, SIGTERM);
            int status;
            waitpid(pid, &status, 0);
            pid = 0;
        }
#endif
    }

    ~test_rpc_server() { stop(); }
};

// ---- Helper: allocate a buffer on a server ----
// Returns the remote_ptr (handle) or 0 on failure.
static uint64_t alloc_buffer(sockfd_t fd, uint32_t device, uint64_t size, uint64_t * out_actual_size = nullptr) {
    rpc_msg_alloc_buffer_req req = {device, size};
    rpc_msg_alloc_buffer_rsp rsp = {};
    if (!send_rpc_cmd(fd, RPC_CMD_ALLOC_BUFFER, &req, sizeof(req), &rsp, sizeof(rsp))) {
        return 0;
    }
    if (out_actual_size) *out_actual_size = rsp.remote_size;
    return rsp.remote_ptr;
}

// ---- Helper: get buffer base pointer ----
static uint64_t buffer_get_base(sockfd_t fd, uint64_t remote_ptr) {
    rpc_msg_buffer_get_base_req req = {remote_ptr};
    rpc_msg_buffer_get_base_rsp rsp = {};
    if (!send_rpc_cmd(fd, RPC_CMD_BUFFER_GET_BASE, &req, sizeof(req), &rsp, sizeof(rsp))) {
        return 0;
    }
    return rsp.base_ptr;
}

// ---- Helper: free a buffer on a server ----
static bool free_buffer(sockfd_t fd, uint64_t remote_ptr) {
    rpc_msg_free_buffer_req req = {remote_ptr};
    // FREE_BUFFER expects zero-length response
    uint8_t cmd = RPC_CMD_FREE_BUFFER;
    if (!send_all(fd, &cmd, 1)) return false;
    uint64_t sz = sizeof(req);
    if (!send_all(fd, &sz, sizeof(sz))) return false;
    if (!send_all(fd, &req, sizeof(req))) return false;
    // Read empty response
    uint64_t out_size = 0;
    if (!recv_all(fd, &out_size, sizeof(out_size))) return false;
    return out_size == 0;
}

// ---- Helper: build an rpc_tensor for a 1D F32 tensor referencing a buffer ----
static rpc_tensor make_f32_tensor(uint64_t buffer_ptr, uint64_t base_ptr, uint32_t n_elements) {
    rpc_tensor t = {};
    t.id      = 0;
    t.type    = 0; // GGML_TYPE_F32
    t.buffer  = buffer_ptr;
    t.ne[0]   = n_elements;
    t.ne[1]   = 1;
    t.ne[2]   = 1;
    t.ne[3]   = 1;
    t.nb[0]   = 4;  // sizeof(float)
    t.nb[1]   = n_elements * 4;
    t.nb[2]   = n_elements * 4;
    t.nb[3]   = n_elements * 4;
    t.op      = 0;  // GGML_OP_NONE
    t.flags   = 0;
    t.view_src  = 0;
    t.view_offs = 0;
    t.data    = base_ptr;
    memset(t.name, 0, sizeof(t.name));
    strncpy(t.name, "test", sizeof(t.name) - 1);
    return t;
}

// ---- Helper: SET_TENSOR ----
// Wire format: | rpc_tensor | offset (8 bytes) | data (size bytes) |
// This is a no-response command.
static bool set_tensor_data(sockfd_t fd, const rpc_tensor & tensor, uint64_t offset,
                            const void * data, size_t size) {
    size_t payload_size = sizeof(rpc_tensor) + sizeof(uint64_t) + size;
    std::vector<uint8_t> payload(payload_size);
    memcpy(payload.data(), &tensor, sizeof(rpc_tensor));
    memcpy(payload.data() + sizeof(rpc_tensor), &offset, sizeof(offset));
    memcpy(payload.data() + sizeof(rpc_tensor) + sizeof(uint64_t), data, size);

    return send_rpc_cmd_no_rsp(fd, RPC_CMD_SET_TENSOR, payload.data(), payload.size());
}

// ---- Helper: GET_TENSOR ----
static bool get_tensor_data(sockfd_t fd, const rpc_tensor & tensor, uint64_t offset,
                            void * data, size_t size) {
    rpc_msg_get_tensor_req req;
    req.tensor = tensor;
    req.offset = offset;
    req.size   = size;
    return send_rpc_cmd(fd, RPC_CMD_GET_TENSOR, &req, sizeof(req), data, size);
}

// ---- Helper: REGISTER_PEER ----
static bool register_peer(sockfd_t fd, uint32_t peer_id, const char * endpoint) {
    rpc_msg_register_peer_req req = {};
    req.peer_id = peer_id;
    strncpy(req.endpoint, endpoint, sizeof(req.endpoint) - 1);
    req.endpoint[sizeof(req.endpoint) - 1] = '\0';

    rpc_msg_register_peer_rsp rsp = {};
    if (!send_rpc_cmd(fd, RPC_CMD_REGISTER_PEER, &req, sizeof(req), &rsp, sizeof(rsp))) {
        return false;
    }
    return rsp.result == 1;
}

// ---- Helper: REGISTER_PEER (returns raw result) ----
static int register_peer_raw(sockfd_t fd, uint32_t peer_id, const char * endpoint) {
    rpc_msg_register_peer_req req = {};
    req.peer_id = peer_id;
    strncpy(req.endpoint, endpoint, sizeof(req.endpoint) - 1);
    req.endpoint[sizeof(req.endpoint) - 1] = '\0';

    rpc_msg_register_peer_rsp rsp = {};
    if (!send_rpc_cmd(fd, RPC_CMD_REGISTER_PEER, &req, sizeof(req), &rsp, sizeof(rsp))) {
        return -1;  // communication failure
    }
    return rsp.result;
}

// ---- Helper: PUSH_TENSOR_TO_PEER ----
static bool push_tensor_to_peer(sockfd_t fd, uint32_t peer_id,
                                const rpc_tensor & src, const rpc_tensor & dst,
                                uint64_t offset, uint64_t size) {
    rpc_msg_push_tensor_to_peer_req req;
    req.peer_id = peer_id;
    req.src     = src;
    req.dst     = dst;
    req.offset  = offset;
    req.size    = size;

    rpc_msg_push_tensor_to_peer_rsp rsp = {};
    if (!send_rpc_cmd(fd, RPC_CMD_PUSH_TENSOR_TO_PEER, &req, sizeof(req), &rsp, sizeof(rsp))) {
        return false;
    }
    return rsp.result == 1;
}

// ---- Test helpers ----

static int  g_tests_run    = 0;
static int  g_tests_passed = 0;
static int  g_tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (line %d): %s\n", __func__, __LINE__, msg); \
        g_tests_failed++; \
        return false; \
    } \
} while(0)

#define RUN_TEST(test_func) do { \
    g_tests_run++; \
    printf("Running %s...\n", #test_func); \
    if (test_func) { \
        g_tests_passed++; \
        printf("  PASS: %s\n", #test_func); \
    } else { \
        fprintf(stderr, "  FAIL: %s\n", #test_func); \
    } \
} while(0)

// ---- Find the rpc-server binary ----
static std::string find_rpc_server_binary() {
    // Try common locations relative to the test binary
    const char * candidates[] = {
        "./bin/rpc-server",
        "./rpc-server",
        "../tools/rpc/rpc-server",
        "../bin/rpc-server",
    };
    for (const char * path : candidates) {
        if (access(path, X_OK) == 0) {
            return path;
        }
    }
    // Try the RPC_SERVER_PATH environment variable
    const char * env = getenv("RPC_SERVER_PATH");
    if (env && access(env, X_OK) == 0) {
        return env;
    }
    return "";
}

// Pick two free ports by binding to port 0 and reading back the assignment.
static int pick_free_port() {
    sockfd_t fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd == INVALID_SOCKET) return 0;
    struct sockaddr_in addr = {};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = 0;
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        CLOSESOCK(fd);
        return 0;
    }
    socklen_t len = sizeof(addr);
    if (getsockname(fd, (struct sockaddr *)&addr, &len) != 0) {
        CLOSESOCK(fd);
        return 0;
    }
    int port = ntohs(addr.sin_port);
    CLOSESOCK(fd);
    return port;
}

// ==========================================================================
// Test 1: Peer Registration Round-Trip
// ==========================================================================
static bool test_peer_registration(const std::string & rpc_server_path) {
    int port_a = pick_free_port();
    int port_b = pick_free_port();
    TEST_ASSERT(port_a > 0 && port_b > 0 && port_a != port_b, "failed to pick free ports");

    test_rpc_server server_a, server_b;
    TEST_ASSERT(server_a.start(rpc_server_path, port_a), "failed to start server A");
    TEST_ASSERT(server_b.start(rpc_server_path, port_b), "failed to start server B");

    // Connect to both
    sockfd_t fd_a = connect_to("127.0.0.1", port_a);
    sockfd_t fd_b = connect_to("127.0.0.1", port_b);
    TEST_ASSERT(fd_a != INVALID_SOCKET, "failed to connect to server A");
    TEST_ASSERT(fd_b != INVALID_SOCKET, "failed to connect to server B");

    // HELLO handshake
    TEST_ASSERT(do_hello(fd_a), "HELLO failed on server A");
    TEST_ASSERT(do_hello(fd_b), "HELLO failed on server B");

    // Register peer B on server A
    std::string ep_b = "127.0.0.1:" + std::to_string(port_b);
    TEST_ASSERT(register_peer(fd_a, 1, ep_b.c_str()), "register peer B on A failed");

    // Register peer A on server B
    std::string ep_a = "127.0.0.1:" + std::to_string(port_a);
    TEST_ASSERT(register_peer(fd_b, 0, ep_a.c_str()), "register peer A on B failed");

    // Edge case: empty endpoint → expect result == 0
    {
        int r = register_peer_raw(fd_a, 99, "");
        TEST_ASSERT(r == 0, "empty endpoint should fail");
    }

    // Edge case: overwrite same peer_id with different endpoint → expect success
    {
        std::string alt_ep = "127.0.0.1:" + std::to_string(port_b);
        TEST_ASSERT(register_peer(fd_a, 1, alt_ep.c_str()), "re-register same peer_id should succeed");
    }

    // Edge case: fill most of the 128-byte endpoint field
    // The colon must be within the first 127 bytes (field is 128 bytes, null-terminated)
    {
        // "a<120 a's>:12345" = 127 chars, fits in the 128-byte field
        std::string long_ep(120, 'a');
        long_ep += ":12345";  // total 126 chars, well within bounds
        int r = register_peer_raw(fd_a, 50, long_ep.c_str());
        TEST_ASSERT(r == 1, "long endpoint registration should succeed");
    }

    CLOSESOCK(fd_a);
    CLOSESOCK(fd_b);
    return true;
}

// ==========================================================================
// Test 2: Direct Push (Data Integrity)
// ==========================================================================
static bool test_direct_push(const std::string & rpc_server_path) {
    int port_a = pick_free_port();
    int port_b = pick_free_port();
    TEST_ASSERT(port_a > 0 && port_b > 0 && port_a != port_b, "failed to pick free ports");

    test_rpc_server server_a, server_b;
    TEST_ASSERT(server_a.start(rpc_server_path, port_a), "failed to start server A");
    TEST_ASSERT(server_b.start(rpc_server_path, port_b), "failed to start server B");

    sockfd_t fd_a = connect_to("127.0.0.1", port_a);
    sockfd_t fd_b = connect_to("127.0.0.1", port_b);
    TEST_ASSERT(fd_a != INVALID_SOCKET && fd_b != INVALID_SOCKET, "failed to connect");

    TEST_ASSERT(do_hello(fd_a), "HELLO A");
    TEST_ASSERT(do_hello(fd_b), "HELLO B");

    // Register peers
    std::string ep_a = "127.0.0.1:" + std::to_string(port_a);
    std::string ep_b = "127.0.0.1:" + std::to_string(port_b);
    TEST_ASSERT(register_peer(fd_a, 1, ep_b.c_str()), "register peer B on A");
    TEST_ASSERT(register_peer(fd_b, 0, ep_a.c_str()), "register peer A on B");

    const size_t DATA_SIZE = 4096;
    const uint32_t N_FLOATS = DATA_SIZE / sizeof(float);

    // Allocate buffer on server A
    uint64_t actual_a = 0;
    uint64_t buf_a = alloc_buffer(fd_a, 0, DATA_SIZE, &actual_a);
    TEST_ASSERT(buf_a != 0, "alloc buffer A");
    TEST_ASSERT(actual_a >= DATA_SIZE, "buffer A too small");

    // Allocate buffer on server B
    uint64_t actual_b = 0;
    uint64_t buf_b = alloc_buffer(fd_b, 0, DATA_SIZE, &actual_b);
    TEST_ASSERT(buf_b != 0, "alloc buffer B");
    TEST_ASSERT(actual_b >= DATA_SIZE, "buffer B too small");

    // Get base pointers
    uint64_t base_a = buffer_get_base(fd_a, buf_a);
    uint64_t base_b = buffer_get_base(fd_b, buf_b);
    TEST_ASSERT(base_a != 0, "base ptr A");
    TEST_ASSERT(base_b != 0, "base ptr B");

    // Create known data pattern
    std::vector<uint8_t> src_data(DATA_SIZE);
    for (size_t i = 0; i < DATA_SIZE; i++) {
        src_data[i] = (uint8_t)(i & 0xFF);
    }

    // Build tensor descriptors
    rpc_tensor tensor_a = make_f32_tensor(buf_a, base_a, N_FLOATS);
    rpc_tensor tensor_b = make_f32_tensor(buf_b, base_b, N_FLOATS);

    // Write data to server A
    TEST_ASSERT(set_tensor_data(fd_a, tensor_a, 0, src_data.data(), DATA_SIZE), "set_tensor A");

    // Push from A to B
    TEST_ASSERT(push_tensor_to_peer(fd_a, 1, tensor_a, tensor_b, 0, DATA_SIZE),
                "push_tensor_to_peer A→B");

    // Read data back from server B
    std::vector<uint8_t> dst_data(DATA_SIZE, 0);
    TEST_ASSERT(get_tensor_data(fd_b, tensor_b, 0, dst_data.data(), DATA_SIZE), "get_tensor B");

    // Compare
    TEST_ASSERT(memcmp(src_data.data(), dst_data.data(), DATA_SIZE) == 0,
                "data mismatch after direct push A→B");

    // ---- Variation: Partial push ----
    {
        // Clear buffer B first
        // Write zeros to B
        std::vector<uint8_t> zeros(DATA_SIZE, 0);
        TEST_ASSERT(set_tensor_data(fd_b, tensor_b, 0, zeros.data(), DATA_SIZE), "clear B");

        // Push only offset=1024, size=2048
        uint64_t partial_offset = 1024;
        uint64_t partial_size   = 2048;
        TEST_ASSERT(push_tensor_to_peer(fd_a, 1, tensor_a, tensor_b, partial_offset, partial_size),
                    "partial push A→B");

        // Read full buffer from B
        std::vector<uint8_t> partial_dst(DATA_SIZE, 0);
        TEST_ASSERT(get_tensor_data(fd_b, tensor_b, 0, partial_dst.data(), DATA_SIZE), "get partial B");

        // Bytes [0, 1024) should be zero (unchanged)
        bool pre_ok = true;
        for (size_t i = 0; i < partial_offset; i++) {
            if (partial_dst[i] != 0) { pre_ok = false; break; }
        }
        TEST_ASSERT(pre_ok, "pre-region should be zero after partial push");

        // Bytes [1024, 3072) should match src_data[1024..3072)
        TEST_ASSERT(memcmp(src_data.data() + partial_offset,
                           partial_dst.data() + partial_offset,
                           partial_size) == 0,
                    "partial push data mismatch");

        // Bytes [3072, 4096) should be zero (unchanged)
        bool post_ok = true;
        for (size_t i = partial_offset + partial_size; i < DATA_SIZE; i++) {
            if (partial_dst[i] != 0) { post_ok = false; break; }
        }
        TEST_ASSERT(post_ok, "post-region should be zero after partial push");
    }

    // ---- Variation: Bidirectional push (B → A) ----
    {
        // Write different data to B
        std::vector<uint8_t> b_data(DATA_SIZE);
        for (size_t i = 0; i < DATA_SIZE; i++) {
            b_data[i] = (uint8_t)(0xFF - (i & 0xFF));
        }
        TEST_ASSERT(set_tensor_data(fd_b, tensor_b, 0, b_data.data(), DATA_SIZE), "set_tensor B for reverse");

        // Push B → A
        TEST_ASSERT(push_tensor_to_peer(fd_b, 0, tensor_b, tensor_a, 0, DATA_SIZE),
                    "push_tensor_to_peer B→A");

        // Read from A
        std::vector<uint8_t> a_result(DATA_SIZE, 0);
        TEST_ASSERT(get_tensor_data(fd_a, tensor_a, 0, a_result.data(), DATA_SIZE), "get_tensor A after B→A");

        TEST_ASSERT(memcmp(b_data.data(), a_result.data(), DATA_SIZE) == 0,
                    "data mismatch after reverse push B→A");
    }

    // Cleanup
    free_buffer(fd_a, buf_a);
    free_buffer(fd_b, buf_b);
    CLOSESOCK(fd_a);
    CLOSESOCK(fd_b);
    return true;
}

// ==========================================================================
// Test 3: Large Tensor Push (16 MB)
// ==========================================================================
static bool test_large_push(const std::string & rpc_server_path) {
    int port_a = pick_free_port();
    int port_b = pick_free_port();
    TEST_ASSERT(port_a > 0 && port_b > 0 && port_a != port_b, "failed to pick free ports");

    test_rpc_server server_a, server_b;
    TEST_ASSERT(server_a.start(rpc_server_path, port_a), "failed to start server A");
    TEST_ASSERT(server_b.start(rpc_server_path, port_b), "failed to start server B");

    sockfd_t fd_a = connect_to("127.0.0.1", port_a);
    sockfd_t fd_b = connect_to("127.0.0.1", port_b);
    TEST_ASSERT(fd_a != INVALID_SOCKET && fd_b != INVALID_SOCKET, "connect");
    TEST_ASSERT(do_hello(fd_a), "HELLO A");
    TEST_ASSERT(do_hello(fd_b), "HELLO B");

    std::string ep_b = "127.0.0.1:" + std::to_string(port_b);
    TEST_ASSERT(register_peer(fd_a, 1, ep_b.c_str()), "register peer");

    const size_t LARGE_SIZE = 16 * 1024 * 1024;  // 16 MB
    const uint32_t N_FLOATS = LARGE_SIZE / sizeof(float);

    uint64_t buf_a = alloc_buffer(fd_a, 0, LARGE_SIZE);
    uint64_t buf_b = alloc_buffer(fd_b, 0, LARGE_SIZE);
    TEST_ASSERT(buf_a != 0 && buf_b != 0, "alloc buffers");

    uint64_t base_a = buffer_get_base(fd_a, buf_a);
    uint64_t base_b = buffer_get_base(fd_b, buf_b);
    TEST_ASSERT(base_a != 0 && base_b != 0, "base ptrs");

    // Generate data
    std::vector<uint8_t> data(LARGE_SIZE);
    for (size_t i = 0; i < LARGE_SIZE; i++) {
        data[i] = (uint8_t)((i * 7 + 13) & 0xFF);
    }

    rpc_tensor ta = make_f32_tensor(buf_a, base_a, N_FLOATS);
    rpc_tensor tb = make_f32_tensor(buf_b, base_b, N_FLOATS);

    TEST_ASSERT(set_tensor_data(fd_a, ta, 0, data.data(), LARGE_SIZE), "set large tensor");
    TEST_ASSERT(push_tensor_to_peer(fd_a, 1, ta, tb, 0, LARGE_SIZE), "push large tensor");

    std::vector<uint8_t> result(LARGE_SIZE, 0);
    TEST_ASSERT(get_tensor_data(fd_b, tb, 0, result.data(), LARGE_SIZE), "get large tensor");
    TEST_ASSERT(memcmp(data.data(), result.data(), LARGE_SIZE) == 0, "large tensor data mismatch");

    free_buffer(fd_a, buf_a);
    free_buffer(fd_b, buf_b);
    CLOSESOCK(fd_a);
    CLOSESOCK(fd_b);
    return true;
}

// ==========================================================================
// Test 4: Fallback on Failed Direct Transfer (wrong endpoint)
// ==========================================================================
static bool test_fallback_on_failed_push(const std::string & rpc_server_path) {
    int port_a = pick_free_port();
    int port_b = pick_free_port();
    TEST_ASSERT(port_a > 0 && port_b > 0 && port_a != port_b, "failed to pick free ports");

    test_rpc_server server_a, server_b;
    TEST_ASSERT(server_a.start(rpc_server_path, port_a), "failed to start server A");
    TEST_ASSERT(server_b.start(rpc_server_path, port_b), "failed to start server B");

    sockfd_t fd_a = connect_to("127.0.0.1", port_a);
    TEST_ASSERT(fd_a != INVALID_SOCKET, "connect A");
    TEST_ASSERT(do_hello(fd_a), "HELLO A");

    // Register a WRONG endpoint for peer B on server A (unreachable port)
    TEST_ASSERT(register_peer(fd_a, 1, "127.0.0.1:59999"), "register wrong peer on A");

    const size_t DATA_SIZE = 1024;
    const uint32_t N_FLOATS = DATA_SIZE / sizeof(float);

    uint64_t buf_a = alloc_buffer(fd_a, 0, DATA_SIZE);
    TEST_ASSERT(buf_a != 0, "alloc A");
    uint64_t base_a = buffer_get_base(fd_a, buf_a);

    // We also need a valid buffer on server B for the dst tensor descriptor,
    // but the push will happen through server A → wrong endpoint, so it should fail.
    sockfd_t fd_b = connect_to("127.0.0.1", port_b);
    TEST_ASSERT(fd_b != INVALID_SOCKET, "connect B");
    TEST_ASSERT(do_hello(fd_b), "HELLO B");

    uint64_t buf_b = alloc_buffer(fd_b, 0, DATA_SIZE);
    TEST_ASSERT(buf_b != 0, "alloc B");
    uint64_t base_b = buffer_get_base(fd_b, buf_b);

    // Write data to A
    std::vector<uint8_t> data(DATA_SIZE);
    for (size_t i = 0; i < DATA_SIZE; i++) data[i] = (uint8_t)(i & 0xFF);

    rpc_tensor ta = make_f32_tensor(buf_a, base_a, N_FLOATS);
    rpc_tensor tb = make_f32_tensor(buf_b, base_b, N_FLOATS);

    TEST_ASSERT(set_tensor_data(fd_a, ta, 0, data.data(), DATA_SIZE), "set tensor A");

    // Attempt direct push — should FAIL (result == 0)
    rpc_msg_push_tensor_to_peer_req req;
    req.peer_id = 1;
    req.src     = ta;
    req.dst     = tb;
    req.offset  = 0;
    req.size    = DATA_SIZE;

    rpc_msg_push_tensor_to_peer_rsp rsp = {};
    bool comm_ok = send_rpc_cmd(fd_a, RPC_CMD_PUSH_TENSOR_TO_PEER,
                                &req, sizeof(req), &rsp, sizeof(rsp));
    TEST_ASSERT(comm_ok, "push command communication should succeed");
    TEST_ASSERT(rsp.result == 0, "push to wrong endpoint should return result == 0");

    // The client should fall back to get-then-set relay.
    // Simulate that: GET from A, then SET on B.
    std::vector<uint8_t> relay_buf(DATA_SIZE, 0);
    TEST_ASSERT(get_tensor_data(fd_a, ta, 0, relay_buf.data(), DATA_SIZE), "get from A (relay)");
    TEST_ASSERT(set_tensor_data(fd_b, tb, 0, relay_buf.data(), DATA_SIZE), "set on B (relay)");

    // Verify data on B
    std::vector<uint8_t> verify(DATA_SIZE, 0);
    TEST_ASSERT(get_tensor_data(fd_b, tb, 0, verify.data(), DATA_SIZE), "verify B");
    TEST_ASSERT(memcmp(data.data(), verify.data(), DATA_SIZE) == 0, "relay data should match");

    free_buffer(fd_a, buf_a);
    free_buffer(fd_b, buf_b);
    CLOSESOCK(fd_a);
    CLOSESOCK(fd_b);
    return true;
}

// ==========================================================================
// Test 5: Rapid Connect/Disconnect Stress
// ==========================================================================
static bool test_rapid_connect_disconnect(const std::string & rpc_server_path) {
    int port = pick_free_port();
    TEST_ASSERT(port > 0, "pick free port");

    test_rpc_server server;
    TEST_ASSERT(server.start(rpc_server_path, port), "start server");

    const int ITERATIONS = 100;
    int successes = 0;

    for (int i = 0; i < ITERATIONS; i++) {
        sockfd_t fd = connect_to("127.0.0.1", port);
        if (fd == INVALID_SOCKET) continue;
        if (do_hello(fd)) {
            successes++;
        }
        CLOSESOCK(fd);
    }

    TEST_ASSERT(successes >= ITERATIONS * 9 / 10,
                "at least 90% of rapid connect/disconnect should succeed");

    // Verify server is still responsive
    sockfd_t fd = connect_to("127.0.0.1", port);
    TEST_ASSERT(fd != INVALID_SOCKET, "connect after stress");
    TEST_ASSERT(do_hello(fd), "HELLO after stress");
    CLOSESOCK(fd);

    return true;
}

// ==========================================================================
// main
// ==========================================================================
int main() {
#ifdef _WIN32
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif

    std::string rpc_server_path = find_rpc_server_binary();
    if (rpc_server_path.empty()) {
        fprintf(stderr, "ERROR: Cannot find rpc-server binary.\n"
                "Set RPC_SERVER_PATH env var or run from the build directory.\n");
        return 1;
    }
    printf("Using rpc-server: %s\n\n", rpc_server_path.c_str());

    RUN_TEST(test_peer_registration(rpc_server_path));
    RUN_TEST(test_direct_push(rpc_server_path));
    RUN_TEST(test_large_push(rpc_server_path));
    RUN_TEST(test_fallback_on_failed_push(rpc_server_path));
    RUN_TEST(test_rapid_connect_disconnect(rpc_server_path));

    printf("\n========================================\n");
    printf("Results: %d/%d passed, %d failed\n", g_tests_passed, g_tests_run, g_tests_failed);
    printf("========================================\n");

#ifdef _WIN32
    WSACleanup();
#endif

    return g_tests_failed > 0 ? 1 : 0;
}
