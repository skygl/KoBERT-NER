import argparse
import os

from fewshot_trainer import FewShotTrainer
from trainer import Trainer
from utils import init_logger, load_tokenizer, set_seed, MODEL_CLASSES, MODEL_PATH_MAP
from data_loader import load_and_cache_examples, get_loader


def main(args):
    init_logger()
    set_seed(args)

    train_dataset = None
    dev_dataset = None
    test_dataset = None

    if args.task == 'naver-ner':
        tokenizer = load_tokenizer(args)

        if args.do_train or args.do_eval:
            test_dataset = load_and_cache_examples(args, tokenizer, mode="test")
        if args.do_train:
            train_dataset = load_and_cache_examples(args, tokenizer, mode="train")

        trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)

    elif args.task == 'fsl':
        trainer = FewShotTrainer(args, train_dataset, dev_dataset, test_dataset)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("test", "eval")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default="naver-ner", type=str, help="The name of the task to train")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument("--pred_dir", default="./preds", type=str, help="The prediction file dir")

    parser.add_argument("--train_file", default="train.txt", type=str, help="Train file")
    parser.add_argument("--test_file", default="test.txt", type=str, help="Test file")
    parser.add_argument("--label_file", default="label.txt", type=str, help="Slot Label file")
    parser.add_argument("--write_pred", action="store_true", help="Write prediction during evaluation")

    parser.add_argument("--model_type", default="kobert", type=str, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", default=50, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=20.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=1000, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument('--cuda_number', type=int, default=0, help="number of CUDA to use")

    # Reference : https://github.com/thunlp/Few-NERD
    parser.add_argument('--trainN', default=2, type=int, help='N in train')
    parser.add_argument('--N', default=2, type=int, help='N way')
    parser.add_argument('--K', default=2, type=int, help='K shot')
    parser.add_argument('--Q', default=3, type=int, help='Num of query per class')
    parser.add_argument('--train_iter', default=600, type=int, help='num of iters in training')
    parser.add_argument('--valid_iter', default=100, type=int, help='num of iters in training')
    parser.add_argument('--test_iter', default=500, type=int, help='num of iters in training')
    parser.add_argument('--val_step', default=20, type=int, help='val after training how many iters')

    args = parser.parse_args()

    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]
    main(args)
