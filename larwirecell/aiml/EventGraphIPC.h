// EventGraphIPC.h — SBNDIPC transport helper for the EventGraph memory bridge.
//
// Sends one EventGraph per ART event over a pre-connected POSIX socket using
// the SBNDIPC wire protocol (EVENT_GRAPH=8).  The caller (TensorSetLabeler)
// holds the socket fd and calls send_event_graph() + wait_for_ack() after
// assembling each EventGraph, before moving it into m_events.
//
// Wire format (all little-endian):
//
//   Envelope (32 bytes):
//     uint8[8]  magic       = "SBNDIPC\0"
//     uint16    version     = 1
//     uint16    msg_type    = 8 (EVENT_GRAPH) | 5 (ACK) | 6 (ERROR) | 7 (EOS)
//     uint64    payload_len
//     uint32    run, subrun, event
//
//   EVENT_GRAPH payload (schema version 1):
//     uint32    schema_version = 1
//     uint32    flags          (bit 0 = has_truth)
//     uint32    sample_name_len
//     char[]    sample_name    (ASCII, no null terminator)
//     uint32    member_count
//     per member:
//       uint32  name_len
//       char[]  field_name    (e.g. "sp/pos")
//       uint8   dtype         (1 = float32 LE, 2 = int64 LE)
//       uint8   rank          (0 = scalar, 1 = 1D, 2 = 2D)
//       uint16  reserved      = 0
//       uint64  dims[rank]
//       uint64  byte_count
//       char[]  raw_bytes     (C-contiguous, little-endian)
//
//   ACK payload:    empty (payload_len = 0)
//   ERROR payload:  uint32 msg_len + char msg[msg_len]

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace WireCell {
namespace AIML {

// Protocol constants (shared with Python eventgraph_ipc.py).
static constexpr uint16_t SBNDIPC_PROTOCOL_VERSION = 1;
static constexpr uint16_t SBNDIPC_EVENT_GRAPH       = 8;
static constexpr uint16_t SBNDIPC_ACK               = 5;
static constexpr uint16_t SBNDIPC_ERROR             = 6;
static constexpr uint16_t SBNDIPC_END_OF_STREAM     = 7;
static constexpr uint8_t  EG_DTYPE_FLOAT32_LE       = 1;
static constexpr uint8_t  EG_DTYPE_INT64_LE         = 2;
static constexpr uint32_t EG_SCHEMA_VERSION         = 1;

// Lightweight view into one H5Member's data — no ownership, valid only while
// the source H5Member lives (i.e. while ev is in scope in TensorSetLabeler).
struct EventGraphMemberView {
    const char*     name;       // null-terminated field name, e.g. "sp/pos"
    bool            is_float;   // true = float32, false = int64
    int             rank;       // 0 = scalar, 1 = 1D, 2 = 2D
    const uint64_t* dims;       // rank dimension values (may be nullptr if rank==0)
    const void*     data;       // raw C-contiguous little-endian bytes
    size_t          byte_count; // total bytes (elements * sizeof(element))
};

// Send one EVENT_GRAPH message with the given members.
// Returns true if the send completed without error.
// The caller must follow with wait_for_ack() before reusing the socket.
bool send_event_graph(int fd,
                      const char* sample_name,
                      int run, int sub, int evt,
                      bool has_truth,
                      const EventGraphMemberView* members,
                      int n_members);

// Block until the peer sends ACK (type 5) for the given run/sub/evt.
// Returns false if ERROR (type 6) or a protocol violation is received.
bool wait_for_ack(int fd, int run, int sub, int evt);

// Send END_OF_STREAM (type 7) to signal finalize().
bool send_end_of_stream(int fd, int run, int sub, int evt);

// Connect (blocking with retries) to a SOCK_STREAM Unix-domain socket.
// Returns the connected fd, or -1 on failure after max_retries attempts.
int connect_unix_socket(const std::string& path, int max_retries = 30,
                        int retry_delay_ms = 200);

} // namespace AIML
} // namespace WireCell
