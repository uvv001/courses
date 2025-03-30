from huggingface_hub import login
import os

def main():
    login(os.environ["HF_TOKEN"])
    print("Hello from smollagents!")


if __name__ == "__main__":
    main()
