import common.info as info

def main():
    setup.print_torch_version()
    print("CUDA is available:", setup.cuda_is_available())
    print("CUDA device count:", setup.cuda_device_count())

if __name__ == "__main__":
    main()
