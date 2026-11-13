#include "utils.hpp"
#include "dayan-intf.h"
#include "core_context.hpp"
#include <atomic>
#include <thread>
#include <optional>
#include <signal.h>
#include <sys/prctl.h>
#include <unistd.h>

static constexpr size_t kUramBytesPerCore = 1536 * 1024;

#ifndef DAYAN_DEFAULT_QEMU
#define DAYAN_DEFAULT_QEMU "deps/QEMU/build/dl-riscv64"
#endif

/*
 * ClusterMMU config sync (qemu-MMU extension).  The QEMU front end is
 * authoritative for the scalar ClusterMMU and pushes every accepted
 * translation-semantic APB write as {0x20, addr, value} and every
 * flush trigger as {0x21, mode | asid << 8} on the dedicated
 * DAYAN_MMU_SOCKET_PATH socket (8-byte word records).  This loop owns the
 * listening end and applies each event to all cores' mmu_t so the
 * tensor ports translate against the same config image.  Status
 * registers are never mirrored: FLUSH_DONE/ACK stay QEMU-authoritative.
 *
 * Non-fatal by design: with an older QEMU (no config-sync peer) the
 * thread simply blocks in accept() and the emulator runs unchanged.
 *
 * Startup handshake: while a peer connection is being established and
 * until its initial config replay (which always ends in a flush) has
 * been applied, mmu_config_synced stays false and the main loop holds
 * tensor instruction fetch, so no tensor access can run against the
 * reset or partially-replayed config image.
 */
static std::atomic<bool> mmu_config_synced{true};

static void mmu_config_sync_loop(std::vector<std::unique_ptr<core_context_t>>* cores,
                                  DAYAN::resource_cluster_t* resources) {
    for(;;) {
        int fd = dayan_socket_server_init(DAYAN_MMU_SOCKET_PATH);
        if(fd < 0) {
            fprintf(stderr, "MMU config-sync: cannot listen on %s (%s); "
                            "QEMU register mirroring disabled\n",
                    DAYAN_MMU_SOCKET_PATH, strerror(errno));
            return;
        }
        printf("MMU config-sync: QEMU front end connected\n");
        bool peer_lost = false;
        bool initial_replay_done = false;

        mmu_config_synced.store(false, std::memory_order_release);
        while(!peer_lost) {
            uint64_t opcode;
            if(dayan_socket_pop(fd, &opcode) != 0) {
                peer_lost = true;
                break;
            }
            switch(opcode) {
            case 0x20: { // {addr, value} translation-semantic register write
                uint64_t addr, value;
                if(dayan_socket_pop(fd, &addr) != 0 ||
                   dayan_socket_pop(fd, &value) != 0) {
                    peer_lost = true;
                    break;
                }
                for(auto& core : *cores) {
                    core->mmu->write_register((uint16_t)addr, (uint32_t)value);
                }
                break;
            }
            case 0x21: { // {mode | asid << 8} TLB flush trigger
                uint64_t payload;
                if(dayan_socket_pop(fd, &payload) != 0) {
                    peer_lost = true;
                    break;
                }
                for(auto& core : *cores) {
                    core->mmu->flush((payload & 1) != 0,
                                     (uint8_t)(payload >> 8));
                }
                if(!initial_replay_done) {
                    // The initial replay ends with a full flush; tensor
                    // execution may resume once it has been applied.
                    initial_replay_done = true;
                    mmu_config_synced.store(true, std::memory_order_release);
                }
                break;
            }
            case 0x22: { // Shared Resource clear mask; acknowledge before Host launch.
                uint64_t mask;
                if(dayan_socket_pop(fd, &mask) != 0) {
                    peer_lost = true;
                    break;
                }
                const bool accepted = !(mask >> 32) &&
                    resources->clear(static_cast<uint32_t>(mask));
                if(dayan_socket_push(fd, accepted ? 0 : 1) != 0) {
                    peer_lost = true;
                }
                break;
            }
            default:
                fprintf(stderr, "MMU config-sync: unknown opcode 0x%llx ignored\n",
                        (unsigned long long)opcode);
                break;
            }
        }

        fprintf(stderr, "MMU config-sync: QEMU peer lost; re-listening\n");
        close(fd);
    }
}

int main(int argc, char** argv) {
    int fdMsg, fdInstReq, fdInstRsp;
    // Note: fdMsg and the instruction request/response pair are shared IPC channels.
    // Serialize each instruction fetch transaction so request/response ordering stays intact.
    std::mutex fdMsgMutex;  // Protects fdMsg (used by backend_instruction_commit)
    std::mutex fdInstMutex; // Protects fdInstReq/fdInstRsp (used by backend_instruction_fetch)
    // Parse command line arguments to get core count
    int num_cores = 1;
    int qemu_arg_start = 1; // Starting position of QEMU arguments
    bool has_c_option = false;

    // Check for -c or --cores argument
    for(int i = 1; i < argc; i++) {
        if(strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--cores") == 0) {
            if(i + 1 >= argc) {
                fprintf(stderr, "Error: -c/--cores requires a value\n");
                fprintf(stderr, "Usage: %s [-c|--cores N] [QEMU arguments...]\n", argv[0]);
                return 1;
            }
            num_cores = atoi(argv[i + 1]);
            if(num_cores <= 0 || num_cores > 8) {
                fprintf(stderr, "Warning: Invalid core count %d (must be 1-8), using default 1\n", num_cores);
                num_cores = 1;
            }
            has_c_option = true;
            // Skip both the argument name and its value
            i++;
            qemu_arg_start = i + 1;
            break;
        } else if(strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [-c|--cores N] [QEMU arguments...]\n", argv[0]);
            printf("\nOptions:\n");
            printf("  -c, --cores N    Number of cores (1-8, default: 1)\n");
            printf("  -h, --help       Show this help message\n");
            printf("\nNote: If both -c and -smp are specified, -c takes precedence.\n");
            printf("\nExamples:\n");
            printf("  %s -c 4\n", argv[0]);
            printf("  %s -c 4 ./deps/QEMU/build/dl-riscv64 -monitor null ...\n", argv[0]);
            return 0;
        }
    }

    // Check for -smp parameter in QEMU arguments and validate consistency
    int smp_value = -1;
    int smp_index = -1;
    if(qemu_arg_start < argc) {
        for(int i = qemu_arg_start; i < argc; i++) {
            if(strcmp(argv[i], "-smp") == 0) {
                smp_index = i;
                if(i + 1 < argc) {
                    // Parse simple format: -smp N
                    smp_value = atoi(argv[i + 1]);
                    // Also check for cpus=N format (basic support)
                    if(smp_value == 0 && argv[i + 1][0] != '0') {
                        // Try to parse cpus=N format
                        if(strncmp(argv[i + 1], "cpus=", 5) == 0) {
                            smp_value = atoi(argv[i + 1] + 5);
                        }
                    }
                }
                break;
            }
        }
    }

    // Handle core count priority: -c takes precedence over -smp
    if(has_c_option && smp_index != -1 && smp_value > 0) {
        // Both -c and -smp are specified
        if(smp_value != num_cores) {
            fprintf(stderr, "Warning: -c specifies %d cores but -smp specifies %d. "
                    "Using -c value (%d) for consistency.\n",
                    num_cores, smp_value, num_cores);
            // Override -smp value with -c value
            // We'll handle this when constructing the QEMU command
        }
    } else if(!has_c_option && smp_index != -1 && smp_value > 0) {
        // Only -smp is specified, use its value
        num_cores = smp_value;
        if(num_cores <= 0 || num_cores > 8) {
            fprintf(stderr, "Warning: Invalid -smp core count %d (must be 1-8), using default 1\n", num_cores);
            num_cores = 1;
        }
    }

    printf("DAYAN Emulator: Using %d cores\n", num_cores);

    size_t dram_size = DAYAN::MMU_DEFAULT_DRAM_SIZE;
    if (const char *memory_mib = getenv("DAYAN_EMU_MEMORY_MIB")) {
        char *end = nullptr;
        errno = 0;
        const unsigned long long mib = strtoull(memory_mib, &end, 10);
        if (errno || memory_mib[0] < '0' || memory_mib[0] > '9' || *end ||
            mib == 0 || mib > SIZE_MAX / (1024UL * 1024UL)) {
            fprintf(stderr, "Error: invalid DAYAN_EMU_MEMORY_MIB\n");
            return 1;
        }
        dram_size = mib * 1024UL * 1024UL;
    }

    // Create DRAM (shared by all cores)
    uint8_t* dram = (uint8_t *)backend_get_shm_ptr("/dram_shm_0", dram_size);
    if (dram == nullptr) {
        fprintf(stderr, "Error: failed to create shared DRAM\n");
        return 1;
    }
    printf("Note: If the emulator fails to start with SIGBUS signal, it means Shared Memory is not properly created.\nPlease run: sudo mount -o remount,size=32G /dev/shm\n");
    dram[dram_size - 1] = 0x0; // Check Shared DRAM Status

    uint8_t* uram[num_cores];
    // Create URAM (Per-core)
    for(int i = 0 ; i < num_cores; i ++) {
        // Create per-core URAM
        char uram_shm_name[64];
        snprintf(uram_shm_name, sizeof(uram_shm_name), "/uram_shm_%d", i);
        uram[i] = (uint8_t *)backend_get_shm_ptr(uram_shm_name,
                                                kUramBytesPerCore);
        if(uram[i] == nullptr) {
            fprintf(stderr, "Error: Failed to create URAM for core %d\n", i);
            throw std::runtime_error("Failed to create URAM");
        }
        printf("Core %d: Created URAM '%s' at %p\n", i, uram_shm_name, uram[i]);
    }

    // Convert core count to string (for -smp parameter)
    std::string smp_arg = std::to_string(num_cores);

    // Create QEMU child process
    pid_t parentPid = getpid();
    pid_t qemuPid = fork();
    if (qemuPid == -1) {
        return -1;
    }
    else if (qemuPid == 0) {
        // QEMU may still be blocked in startup IPC when its parent exits, before
        // its normal signal handling can drain. Never leave that backend orphaned.
        if (prctl(PR_SET_PDEATHSIG, SIGKILL) == -1) {
            perror("prctl(PR_SET_PDEATHSIG) failed");
            _exit(EXIT_FAILURE);
        }
        if (getppid() != parentPid) {
            _exit(EXIT_FAILURE);
        }
        if(qemu_arg_start >= argc) {// Default debug mode (no QEMU arguments provided)
            // Default mode: automatically add -smp parameter
            execlp(DAYAN_DEFAULT_QEMU, "QEMU_THREAD", "-monitor", "null",
            "-bios", "../dayan-base/device/bl0/bootrom.elf",
            "-machine", "dayan-v0", "-nographic", "-m", "4096",
            "-smp", smp_arg.c_str(), "-S", "-s", NULL);
            perror("failed to exec default QEMU backend");
            _exit(EXIT_FAILURE);
        } else {
            // User-defined argument mode: construct argument array with correct -smp value
            std::vector<char*> new_argv;
            // Copy original QEMU arguments (skip -c/--cores parameters)
            for(int i = qemu_arg_start; i < argc; i++) {
                if(i == smp_index) {
                    // Skip the -smp argument and its value, we'll add our own with correct value
                    i++; // Skip the value too
                    continue;
                }
                new_argv.push_back(argv[i]);
            }
            // Always add -smp parameter with num_cores value
            // (from -c if specified, or from -smp if only -smp was specified, or default 1)
            static char smp_str[16];
            snprintf(smp_str, sizeof(smp_str), "%d", num_cores);
            new_argv.push_back(const_cast<char*>("-smp"));
            new_argv.push_back(smp_str);
            new_argv.push_back(nullptr); // NULL terminator
            execvp(new_argv[0], new_argv.data());
            perror("failed to exec QEMU backend");
            _exit(EXIT_FAILURE);
        }
    }
    
    // Parent process creates server socket
    backend_init((int*)&fdMsg, (int*)&fdInstReq, (int*)&fdInstRsp);
    debugf("Main: QEMU Ready!\n");
    
    // Create all cores
    DAYAN::resource_cluster_t resource_cluster;
    std::vector<std::unique_ptr<core_context_t>> cores;
    for(int i = 0; i < num_cores; i++) {
        try {
            cores.push_back(std::make_unique<core_context_t>(
                i, fdMsg, fdMsgMutex, dram, uram[i], &uram[0],
                (size_t)num_cores, resource_cluster, dram_size));
        } catch(const std::exception& e) {
            fprintf(stderr, "Error: Failed to create core %d: %s\n", i, e.what());
            return 1;
        }
    }

    debugf("Main: All %d cores initialized and started\n", num_cores);

    // Mirror QEMU's ClusterMMU config onto the tensor-side mmu_t
    // instances (additive; see mmu_config_sync_loop above).
    std::thread(mmu_config_sync_loop, &cores, &resource_cluster).detach();

    // Main loop: fetch instructions for all cores
    while(1) {
        // Hold tensor instruction fetch until the ClusterMMU config-sync
        // connection (if any) has replayed its initial image; see the
        // handshake note above mmu_config_sync_loop.
        while(!mmu_config_synced.load(std::memory_order_acquire)) {
            usleep(100);
        }
        // Fetch one instruction for each core
        for(auto& core : cores) {
            core->fetch_one(fdInstReq, fdInstRsp, fdInstMutex);
            }
    }
}
