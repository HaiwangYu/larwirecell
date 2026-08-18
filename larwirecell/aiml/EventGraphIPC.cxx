#include "EventGraphIPC.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <vector>

namespace WireCell {
namespace AIML {

// ---- byte-order helpers (target is always little-endian x86_64) -----------

static void write_le16(uint8_t* p, uint16_t v)
{
    p[0] = static_cast<uint8_t>(v);
    p[1] = static_cast<uint8_t>(v >> 8);
}
static void write_le32(uint8_t* p, uint32_t v)
{
    for (int i = 0; i < 4; ++i) { p[i] = static_cast<uint8_t>(v >> (8 * i)); }
}
static void write_le64(uint8_t* p, uint64_t v)
{
    for (int i = 0; i < 8; ++i) { p[i] = static_cast<uint8_t>(v >> (8 * i)); }
}
static uint16_t read_le16(const uint8_t* p)
{
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}
static uint32_t read_le32(const uint8_t* p)
{
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) { v |= static_cast<uint32_t>(p[i]) << (8 * i); }
    return v;
}
static uint64_t read_le64(const uint8_t* p)
{
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) { v |= static_cast<uint64_t>(p[i]) << (8 * i); }
    return v;
}

// ---- envelope (32 bytes) ---------------------------------------------------

static constexpr size_t ENV_SIZE = 32;
static const uint8_t MAGIC[8]   = {'S', 'B', 'N', 'D', 'I', 'P', 'C', '\0'};

static void fill_envelope(uint8_t* env, uint16_t msg_type, uint64_t payload_len,
                           uint32_t run, uint32_t sub, uint32_t evt)
{
    std::memcpy(env, MAGIC, 8);
    write_le16(env + 8,  SBNDIPC_PROTOCOL_VERSION);
    write_le16(env + 10, msg_type);
    write_le64(env + 12, payload_len);
    write_le32(env + 20, run);
    write_le32(env + 24, sub);
    write_le32(env + 28, evt);
}

// ---- I/O helpers -----------------------------------------------------------

static bool write_all(int fd, const void* buf, size_t n)
{
    const uint8_t* p = static_cast<const uint8_t*>(buf);
    while (n > 0) {
        ssize_t r = ::write(fd, p, n);
        if (r < 0) {
            if (errno == EINTR) { continue; }
            return false;
        }
        p += r;
        n -= static_cast<size_t>(r);
    }
    return true;
}

static bool read_all(int fd, void* buf, size_t n)
{
    uint8_t* p = static_cast<uint8_t*>(buf);
    while (n > 0) {
        ssize_t r = ::read(fd, p, n);
        if (r < 0) {
            if (errno == EINTR) { continue; }
            return false;
        }
        if (r == 0) { return false; } // peer closed
        p += r;
        n -= static_cast<size_t>(r);
    }
    return true;
}

// ---- payload builder -------------------------------------------------------

static void append_u8(std::vector<uint8_t>& v, uint8_t x) { v.push_back(x); }
static void append_u16(std::vector<uint8_t>& v, uint16_t x)
{
    uint8_t t[2]; write_le16(t, x); v.insert(v.end(), t, t + 2);
}
static void append_u32(std::vector<uint8_t>& v, uint32_t x)
{
    uint8_t t[4]; write_le32(t, x); v.insert(v.end(), t, t + 4);
}
static void append_u64(std::vector<uint8_t>& v, uint64_t x)
{
    uint8_t t[8]; write_le64(t, x); v.insert(v.end(), t, t + 8);
}
static void append_bytes(std::vector<uint8_t>& v, const void* d, size_t n)
{
    const uint8_t* p = static_cast<const uint8_t*>(d);
    v.insert(v.end(), p, p + n);
}

// ---- public API ------------------------------------------------------------

bool send_event_graph(int fd, const char* sample_name,
                      int run, int sub, int evt, bool has_truth,
                      const EventGraphMemberView* members, int n_members)
{
    std::vector<uint8_t> payload;
    payload.reserve(1 << 20); // 1 MB initial reservation

    append_u32(payload, EG_SCHEMA_VERSION);
    append_u32(payload, has_truth ? 1u : 0u);

    const uint32_t sname_len = static_cast<uint32_t>(std::strlen(sample_name));
    append_u32(payload, sname_len);
    append_bytes(payload, sample_name, sname_len);

    append_u32(payload, static_cast<uint32_t>(n_members));

    for (int i = 0; i < n_members; ++i) {
        const EventGraphMemberView& m = members[i];
        const uint32_t fname_len = static_cast<uint32_t>(std::strlen(m.name));
        append_u32(payload, fname_len);
        append_bytes(payload, m.name, fname_len);
        append_u8(payload, m.is_float ? EG_DTYPE_FLOAT32_LE : EG_DTYPE_INT64_LE);
        append_u8(payload, static_cast<uint8_t>(m.rank));
        append_u16(payload, 0u); // reserved
        for (int d = 0; d < m.rank; ++d) {
            append_u64(payload, m.dims[d]);
        }
        append_u64(payload, static_cast<uint64_t>(m.byte_count));
        append_bytes(payload, m.data, m.byte_count);
    }

    uint8_t env[ENV_SIZE];
    fill_envelope(env, SBNDIPC_EVENT_GRAPH, static_cast<uint64_t>(payload.size()),
                  static_cast<uint32_t>(run), static_cast<uint32_t>(sub),
                  static_cast<uint32_t>(evt));

    if (!write_all(fd, env, ENV_SIZE)) { return false; }
    if (!write_all(fd, payload.data(), payload.size())) { return false; }
    return true;
}

bool wait_for_ack(int fd, int run, int sub, int evt)
{
    uint8_t env[ENV_SIZE];
    if (!read_all(fd, env, ENV_SIZE)) { return false; }
    if (std::memcmp(env, MAGIC, 8) != 0) { return false; }
    if (read_le16(env + 8) != SBNDIPC_PROTOCOL_VERSION) { return false; }

    const uint16_t msg_type    = read_le16(env + 10);
    const uint64_t payload_len = read_le64(env + 12);
    const uint32_t r           = read_le32(env + 20);
    const uint32_t s           = read_le32(env + 24);
    const uint32_t e           = read_le32(env + 28);

    // Drain ERROR payload before returning failure
    if (msg_type == SBNDIPC_ERROR) {
        if (payload_len > 0 && payload_len <= 8192) {
            std::vector<uint8_t> tmp(payload_len);
            read_all(fd, tmp.data(), payload_len);
        }
        return false;
    }
    if (msg_type != SBNDIPC_ACK)                 { return false; }
    if (payload_len != 0)                         { return false; }
    if (r != static_cast<uint32_t>(run) ||
        s != static_cast<uint32_t>(sub) ||
        e != static_cast<uint32_t>(evt))          { return false; }
    return true;
}

bool send_end_of_stream(int fd, int run, int sub, int evt)
{
    uint8_t env[ENV_SIZE];
    fill_envelope(env, SBNDIPC_END_OF_STREAM, 0,
                  static_cast<uint32_t>(run), static_cast<uint32_t>(sub),
                  static_cast<uint32_t>(evt));
    return write_all(fd, env, ENV_SIZE);
}

int connect_unix_socket(const std::string& path, int max_retries, int retry_delay_ms)
{
    int sock = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) { return -1; }

    struct sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);

    const struct timespec delay_ts {
        0, static_cast<long>(retry_delay_ms) * 1'000'000L
    };
    for (int attempt = 0; attempt < max_retries; ++attempt) {
        if (::connect(sock, reinterpret_cast<struct sockaddr*>(&addr),
                      sizeof(addr)) == 0) {
            return sock;
        }
        ::nanosleep(&delay_ts, nullptr);
    }
    ::close(sock);
    return -1;
}

} // namespace AIML
} // namespace WireCell
