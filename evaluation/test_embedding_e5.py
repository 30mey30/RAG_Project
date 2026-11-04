import sys

from evaluation.compare_embedding_models import main


if __name__ == "__main__":
    main(["--model", "e5", *sys.argv[1:]])
