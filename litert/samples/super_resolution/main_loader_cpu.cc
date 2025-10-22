#include <dlfcn.h>
#include <iostream>

typedef int (*SuperResFunc)(int, char**);

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <path_to.so> <model_path> <input_image_path> <output_image_path>"
                  << std::endl;
        return 1;
    }
    void* handle = dlopen(argv[1], RTLD_LAZY);
    if (!handle) { /* error handling */ }
    SuperResFunc super_res_func = (SuperResFunc)dlsym(handle, "run_super_resolution_cpu");
    if (!super_res_func) { /* error handling */ }
    int result = super_res_func(argc - 1, &argv[1]);
    dlclose(handle);
    return result;
}