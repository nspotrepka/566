import common.setup as setup

def main():
    print("PyTorch", setup.torch_version())
    print("CUDA is available:", setup.cuda_is_available())
    print("CUDA device count:", setup.cuda_device_count())

if __name__ == "__main__":
    main()
