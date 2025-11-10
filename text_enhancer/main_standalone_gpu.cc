#include "main_standalone_common.h"

int main(int argc, char** argv) {
    // Call the common main function, specifying "gpu" as the accelerator
    return RunStandaloneSession(argc, argv, "gpu");
}