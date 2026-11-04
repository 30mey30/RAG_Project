import sys

from evaluation.compare_answer_models import main


if __name__ == '__main__':
    main(['--model', 'qwen3:4b', *sys.argv[1:]])
