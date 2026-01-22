#include "utils.hpp"
#include "rvtng-intf.h"
#include "core_context.hpp"
#include <thread>
#include <optional>


int main(int argc, char** argv) {
    int fdMsg, fdInst;
    // Note: fdMsg and fdInst have race condition issues, create mutexes to avoid related problems
    std::mutex fdMsgMutex;  // Protects fdMsg (used by backend_instruction_commit)
    std::mutex fdInstMutex; // Protects fdInst (used by backend_instruction_fetch)
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
            printf("  %s -c 4 /workspace/npu-qemu/build/dl-riscv64 -monitor null ...\n", argv[0]);
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

    printf("RVTNG Emulator: Using %d cores\n", num_cores);

    // Create DRAM (shared by all cores)
    uint8_t* dram = (uint8_t *)backend_get_shm_ptr("/dram_shm_0", 4ul * 1024ul * 1024ul * 1024ul);
    printf("Note: If the emulator fails to start with SIGBUS signal, it means Shared Memory is not properly created.\nPlease run: sudo mount -o remount,size=32G /dev/shm\n");
    dram[0x100000000ul - 1] = 0x0; // Check Shared DRAM Status

    uint8_t* uram[num_cores];
    // Create URAM (Per-core)
    for(int i = 0 ; i < num_cores; i ++) {
        // Create per-core URAM
        char uram_shm_name[64];
        snprintf(uram_shm_name, sizeof(uram_shm_name), "/uram_shm_%d", i);
        uram[i] = (uint8_t *)backend_get_shm_ptr(uram_shm_name, 1024 * 1024);
        if(uram[i] == nullptr) {
            fprintf(stderr, "Error: Failed to create URAM for core %d\n", i);
            throw std::runtime_error("Failed to create URAM");
        }
        printf("Core %d: Created URAM '%s' at %p\n", i, uram_shm_name, uram[i]);
    }

    // Convert core count to string (for -smp parameter)
    std::string smp_arg = std::to_string(num_cores);

    // Create QEMU child process
    pid_t qemuPid = fork();
    if (qemuPid == -1) {
        return -1;
    }
    else if (qemuPid == 0) {
        if(qemu_arg_start >= argc) {// Default debug mode (no QEMU arguments provided)
            // Default mode: automatically add -smp parameter
            execlp("/workspace/npu-qemu/build/dl-riscv64", "QEMU_THREAD", "-monitor", "null",
            "-bios", "/workspace/npu-firmware/bl0/bootrom.elf",
            "-machine", "rvtng-v0", "-nographic", "-m", "4096",
            "-smp", smp_arg.c_str(), "-S", "-s", NULL);
            exit(EXIT_FAILURE);
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
            exit(EXIT_FAILURE);
        }
    }
    
    // Parent process creates server socket
    backend_init((int*)&fdMsg, (int*)&fdInst);
    debugf("Main: QEMU Ready!\n");
    
    // Create all cores
    std::vector<std::unique_ptr<core_context_t>> cores;
    for(int i = 0; i < num_cores; i++) {
        try {
            cores.push_back(std::make_unique<core_context_t>(
                i, fdMsg, fdMsgMutex, dram, uram[i]));
        } catch(const std::exception& e) {
            fprintf(stderr, "Error: Failed to create core %d: %s\n", i, e.what());
            return 1;
        }
    }

    debugf("Main: All %d cores initialized and started\n", num_cores);

    // Main loop: fetch instructions for all cores
    while(1) {
        // Fetch one instruction for each core
        for(auto& core : cores) {
            core->fetch_one(fdInst, fdInstMutex);
            }
    }
}
