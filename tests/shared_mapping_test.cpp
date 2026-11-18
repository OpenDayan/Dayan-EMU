/* Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
 * Licensed under the Mulan PSL v2. */

#include "dayan-intf.h"

#include <sys/wait.h>

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "%s:%d: %s failed (errno=%d: %s)\n", \
                __FILE__, __LINE__, #expr, errno, strerror(errno)); \
        exit(1); \
    } \
} while (0)

static constexpr size_t mapping_size = 4096;

static void check_shared(const char *name) {
    auto *backend = static_cast<unsigned char *>(backend_get_shm_ptr(name, mapping_size));
    CHECK(backend != nullptr);
    backend[0] = 17;
    pid_t child = fork();
    CHECK(child >= 0);
    if (child == 0) {
        auto *frontend = static_cast<unsigned char *>(frontend_get_shm_ptr(name));
        if (!frontend || frontend[0] != 17) _exit(1);
        frontend[1] = 42;
        if (munmap(frontend, mapping_size) != 0) _exit(2);
        _exit(0);
    }
    int status = 0;
    CHECK(waitpid(child, &status, 0) == child);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    CHECK(backend[1] == 42);
    CHECK(munmap(backend, mapping_size) == 0);
}

int main() {
    char directory[] = "/tmp/dayan-mapping-test-XXXXXX";
    CHECK(mkdtemp(directory) != nullptr);
    CHECK(setenv("DAYAN_EMU_SHM_DIR", directory, 1) == 0);

    check_shared("/dram_shm_0");
    check_shared("/uram_shm_0");
    check_shared("/uram_shm_3");

    char path[512];
    CHECK(snprintf(path, sizeof(path), "%s/dram_shm_0", directory) > 0);
    struct stat before, after;
    CHECK(stat(path, &before) == 0);
    errno = 0;
    CHECK(dayan_mapping_open("/dram_shm_0", 1) == -1 && errno == EEXIST);
    CHECK(backend_get_shm_ptr("/dram_shm_0", mapping_size * 2) == nullptr);
    CHECK(frontend_get_shm_ptr_sized("/dram_shm_0", mapping_size * 2) == nullptr);
    CHECK(stat(path, &after) == 0 && before.st_size == after.st_size);
    errno = 0;
    CHECK(dayan_mapping_open("/../bad", 1) == -1 && errno == EINVAL);
    errno = 0;
    CHECK(dayan_mapping_open("/uram_shm_missing", 0) == -1 && errno == EINVAL);
    errno = 0;
    CHECK(dayan_mapping_open("/uram_shm_99", 0) == -1 && errno == ENOENT);
    CHECK(setenv("DAYAN_EMU_SHM_DIR", "/nonexistent/dayan-mapping-test", 1) == 0);
    CHECK(dayan_mapping_open("/dram_shm_0", 1) == -1);
    CHECK(unsetenv("DAYAN_EMU_SHM_DIR") == 0);

    char shm_name[128];
    CHECK(snprintf(shm_name, sizeof(shm_name), "/dayan_mapping_test_%ld", (long)getpid()) > 0);
    int fd = shm_open(shm_name, O_CREAT | O_EXCL | O_RDWR, 0600);
    CHECK(fd >= 0);
    CHECK(close(fd) == 0);
    check_shared(shm_name);
    CHECK(shm_unlink(shm_name) == 0);

    const char *names[] = {"dram_shm_0", "uram_shm_0", "uram_shm_3"};
    for (const char *name : names) {
        CHECK(snprintf(path, sizeof(path), "%s/%s", directory, name) > 0);
        CHECK(unlink(path) == 0);
    }
    CHECK(rmdir(directory) == 0);
    puts("shared mapping tests passed");
    return 0;
}
