import copy
import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader

from fewshot_sampler import FewshotSampleBase, FewshotSampler
from utils import get_labels, get_labels_from_path

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The slot labels of the example.
    """

    def __init__(self, guid, words, labels):
        self.guid = guid
        self.words = words
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_ids = label_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class NaverNerProcessor(object):
    """Processor for the Naver NER data set """

    def __init__(self, args):
        self.args = args
        self.labels_lst = get_labels(args)
        print("labels_lst : ", self.labels_lst)

    @classmethod
    def _read_file(cls, input_file):
        """Read tsv file, and return words and label as list"""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, data) in enumerate(dataset):
            words, labels = data.split('\t')
            words = words.split()
            labels = labels.split()
            guid = "%s-%s" % (set_type, i)

            labels_idx = []
            for label in labels:
                labels_idx.append(
                    self.labels_lst.index(label) if label in self.labels_lst else self.labels_lst.index("UNK"))

            if len(words) != len(labels_idx):
                if len(words) + 1 == len(labels_idx):
                    labels_idx = labels_idx[:-1]

            assert len(words) == len(labels_idx)

            if i % 10000 == 0:
                logger.info(data)
            examples.append(InputExample(guid=guid, words=words, labels=labels_idx))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file
        elif mode == 'test':
            file_to_read = self.args.test_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, file_to_read)))
        return self._create_examples(self._read_file(os.path.join(self.args.data_dir, file_to_read)), mode)


class FewShotProcessor(object):
    def __init__(self, args):
        super(FewShotProcessor, self).__init__()

        self.args = args
        self.labels_lst = get_labels_from_path(os.path.join(args.data_dir, args.label_file))
        self.test_labels_lst = get_labels_from_path(os.path.join(args.data_dir, args.label_file))


processors = {
    "naver-ner": NaverNerProcessor,
    "few-shot": NaverNerProcessor,
    "fsl": FewShotProcessor,
}


def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 pad_token_label_id=-100,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # Tokenize word by word (for NER)
        tokens = []
        label_ids = []
        for word, slot_label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[: (max_seq_len - special_tokens_count)]
            label_ids = label_ids[: (max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        label_ids = label_ids + ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids),
                                                                                                  max_seq_len)
        assert len(label_ids) == max_seq_len, "Error with slot labels length {} vs {}".format(len(label_ids),
                                                                                              max_seq_len)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s " % " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label_ids=label_ids
                          ))

    return features


def load_and_cache_examples(args, tokenizer, mode):
    processor = processors[args.task](args)

    # Load data features from cache or dataset file
    cached_file_name = 'cached_{}_{}_{}_{}'.format(
        args.task, list(filter(None, args.model_name_or_path.split("/"))).pop(), args.max_seq_len, mode)

    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    cached_features_file = os.path.join(args.data_dir, cached_file_name)
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        features = convert_examples_to_features(examples, args.max_seq_len, tokenizer,
                                                pad_token_label_id=pad_token_label_id)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids)
    return dataset


def get_class_name(rawtag):
    # get (finegrained) class name
    if rawtag.startswith('B-') or rawtag.startswith('I-'):
        return rawtag[2:]
    else:
        return rawtag


class Sample(FewshotSampleBase):
    def __init__(self, filelines):
        filelines = [line.split('\t') for line in filelines]
        self.words, self.tags = zip(*filelines)
        self.words = [word.lower() for word in self.words]
        # strip B-, I-
        self.normalized_tags = list(map(get_class_name, self.tags))
        self.class_count = {}

    def __count_entities__(self):
        current_tag = self.normalized_tags[0]
        for tag in self.normalized_tags[1:]:
            if tag == current_tag:
                continue
            else:
                if current_tag != 'O':
                    if current_tag in self.class_count:
                        self.class_count[current_tag] += 1
                    else:
                        self.class_count[current_tag] = 1
                current_tag = tag
        if current_tag != 'O':
            if current_tag in self.class_count:
                self.class_count[current_tag] += 1
            else:
                self.class_count[current_tag] = 1

    def get_class_count(self):
        if self.class_count:
            return self.class_count
        else:
            self.__count_entities__()
            return self.class_count

    def set_class_count(self, class_count):
        self.class_count = class_count
        return self.class_count

    def get_tag_class(self):
        # strip 'B' 'I'
        tag_class = list(set(self.normalized_tags))
        if 'O' in tag_class:
            tag_class.remove('O')
        return tag_class

    def valid(self, target_classes):
        # 앞의 항 : 문장이 target class에 해당하는 클래스 토큰을 가지고 있음
        # 뒤의 항 (삭제) : 문장이 target class 이외 클래스 토큰을 가지고 있지 않아야 함
        return set(self.get_class_count().keys()).intersection(set(target_classes))

    def __str__(self):
        newlines = zip(self.words, self.tags)
        return '\n'.join(['\t'.join(line) for line in newlines])


class FewShotNERDatasetWithRandomSampling(Dataset):
    """
    Fewshot NER Dataset
    """

    def __init__(self, filepath, tokenizer, N, K, Q, max_length, ignore_label_id=-1):
        if not os.path.exists(filepath):
            print("[ERROR] Data file does not exist!")
            assert (0)
        self.class2sampleid = {}
        self.N = N
        self.K = K
        self.Q = Q
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples, self.classes = self.__load_data_from_file__(filepath)
        self.sampler = FewshotSampler(N, K, Q, self.samples, classes=self.classes)
        self.ignore_label_id = ignore_label_id

    def __insert_sample__(self, index, sample_classes):
        for item in sample_classes:
            if item in self.class2sampleid:
                self.class2sampleid[item].append(index)
            else:
                self.class2sampleid[item] = [index]

    def __load_data_from_file__(self, filepath):
        samples = []
        classes = []
        with open(filepath, 'r', encoding='utf-8')as f:
            lines = f.readlines()
        samplelines = []
        index = 0
        total_class_count = {}
        for line in lines:
            line = line.strip()
            if line:
                samplelines.append(line)
            else:
                sample = Sample(samplelines)
                class_count = self.set_sample_class_count(sample)
                for cls in class_count:
                    total_class_count[cls] = total_class_count.get(cls, 0) + class_count[cls]
                samples.append(sample)
                sample_classes = sample.get_tag_class()
                self.__insert_sample__(index, sample_classes)
                classes += sample_classes
                samplelines = []
                index += 1
        if samplelines:
            sample = Sample(samplelines)
            class_count = self.set_sample_class_count(sample)
            for cls in class_count:
                total_class_count[cls] = total_class_count.get(cls, 0) + class_count[cls]
            samples.append(sample)
            sample_classes = sample.get_tag_class()
            self.__insert_sample__(index, sample_classes)
            classes += sample_classes
            samplelines = []
            index += 1
        classes = list(set(classes))
        print(total_class_count)
        return samples, classes

    def set_sample_class_count(self, sample):
        def count_entity(normalized_tags):
            class_count = {}
            current_tag = normalized_tags[0]
            for tag in normalized_tags[1:]:
                if tag == current_tag:
                    continue
                else:
                    if current_tag != 'O':
                        if current_tag in class_count:
                            class_count[current_tag] += 1
                        else:
                            class_count[current_tag] = 1
                    current_tag = tag
            if current_tag != 'O':
                if current_tag in class_count:
                    class_count[current_tag] += 1
                else:
                    class_count[current_tag] = 1

            return class_count

        token_len = 0
        labels = []
        while token_len < self.max_length - 2:
            for word, tag in zip(sample.words, sample.normalized_tags):
                word_tokens = self.tokenizer.tokenize(word)
                if word_tokens:
                    token_len += len(word_tokens)
                    labels.append(tag)
                if token_len >= self.max_length - 2:
                    break

        normalized_labels = list(map(get_class_name, labels))
        class_count_of_sample = count_entity(normalized_labels)
        sample.set_class_count(class_count_of_sample)

        return class_count_of_sample

    def __get_token_label_list__(self, sample):
        tokens = []
        labels = []
        for word, tag in zip(sample.words, sample.normalized_tags):
            word_tokens = self.tokenizer.tokenize(word)
            if word_tokens:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                word_labels = [self.tag2label.get(tag, 0)] + [self.ignore_label_id] * (len(word_tokens) - 1)
                labels.extend(word_labels)
        return tokens, labels

    def __getraw__(self, tokens, labels):
        # get tokenized word list, attention mask, text mask (mask [CLS], [SEP] as well), tags

        # split into chunks of length (max_length-2)
        # 2 is for special tokens [CLS] and [SEP]
        tokens_list = []
        labels_list = []
        while len(tokens) > self.max_length - 2:
            tokens_list.append(tokens[:self.max_length - 2])
            tokens = tokens[self.max_length - 2:]
            labels_list.append(labels[:self.max_length - 2])
            labels = labels[self.max_length - 2:]
        if tokens:
            tokens_list.append(tokens)
            labels_list.append(labels)

        # add special tokens and get masks
        indexed_tokens_list = []
        mask_list = []
        text_mask_list = []
        for i, tokens in enumerate(tokens_list):
            # token -> ids
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

            # padding
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)
            indexed_tokens_list.append(indexed_tokens)

            # mask
            mask = np.zeros((self.max_length), dtype=np.int32)
            mask[:len(tokens)] = 1
            mask_list.append(mask)

            # text mask, also mask [CLS] and [SEP]
            text_mask = np.zeros((self.max_length), dtype=np.int32)
            text_mask[1:len(tokens) - 1] = 1
            text_mask_list.append(text_mask)

            assert len(labels_list[i]) == len(tokens) - 2, print(labels_list[i], tokens)
        return indexed_tokens_list, mask_list, text_mask_list, labels_list

    def __additem__(self, index, d, word, mask, text_mask, label):
        d['index'].append(index)
        d['word'] += word
        d['mask'] += mask
        d['label'] += label
        d['text_mask'] += text_mask

    def __populate__(self, idx_list, savelabeldic=False):
        '''
        populate samples into data dict
        set savelabeldic=True if you want to save label2tag dict
        'index': sample_index
        'word': tokenized word ids
        'mask': attention mask in BERT
        'label': NER labels
        'sentence_num': number of sentences in this set (a batch contains multiple sets)
        'text_mask': 0 for special tokens and paddings, 1 for real text
        '''
        dataset = {'index': [], 'word': [], 'mask': [], 'label': [], 'sentence_num': [], 'text_mask': []}
        for idx in idx_list:
            tokens, labels = self.__get_token_label_list__(self.samples[idx])
            word, mask, text_mask, label = self.__getraw__(tokens, labels)
            word = torch.tensor(word).long()
            mask = torch.tensor(np.array(mask)).long()
            text_mask = torch.tensor(np.array(text_mask)).long()
            self.__additem__(idx, dataset, word, mask, text_mask, label)
        dataset['sentence_num'] = [len(dataset['word'])]
        if savelabeldic:
            dataset['label2tag'] = [self.label2tag]
        return dataset

    def __getitem__(self, index):
        target_classes, support_idx, query_idx = self.sampler.__next__()
        # add 'O' and make sure 'O' is labeled 0
        distinct_tags = ['O'] + target_classes
        self.tag2label = {tag: idx for idx, tag in enumerate(distinct_tags)}
        self.label2tag = {idx: tag for idx, tag in enumerate(distinct_tags)}
        support_set = self.__populate__(support_idx)
        query_set = self.__populate__(query_idx, savelabeldic=True)
        return support_set, query_set

    def __len__(self):
        return 100000


class FewShotNERDataset(FewShotNERDatasetWithRandomSampling):
    def __init__(self, filepath, tokenizer, max_length, ignore_label_id=-1):
        if not os.path.exists(filepath):
            print("[ERROR] Data file does not exist!")
            assert (0)
        self.class2sampleid = {}
        self.tokenizer = tokenizer
        self.samples = self.__load_data_from_file__(filepath)
        self.max_length = max_length
        self.ignore_label_id = ignore_label_id

    def __load_data_from_file__(self, filepath):
        with open(filepath) as f:
            lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = json.loads(lines[i].strip())
        return lines

    def __additem__(self, d, word, mask, text_mask, label):
        d['word'] += word
        d['mask'] += mask
        d['label'] += label
        d['text_mask'] += text_mask

    def __get_token_label_list__(self, words, tags):
        tokens = []
        labels = []
        for word, tag in zip(words, tags):
            word_tokens = self.tokenizer.tokenize(word)
            if word_tokens:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                word_labels = [self.tag2label[tag]] + [self.ignore_label_id] * (len(word_tokens) - 1)
                labels.extend(word_labels)
        return tokens, labels

    def __populate__(self, data, savelabeldic=False):
        '''
        populate samples into data dict
        set savelabeldic=True if you want to save label2tag dict
        'word': tokenized word ids
        'mask': attention mask in BERT
        'label': NER labels
        'sentence_num': number of sentences in this set (a batch contains multiple sets)
        'text_mask': 0 for special tokens and paddings, 1 for real text
        '''
        dataset = {'word': [], 'mask': [], 'label': [], 'sentence_num': [], 'text_mask': []}
        for i in range(len(data['word'])):
            tokens, labels = self.__get_token_label_list__(data['word'][i], data['label'][i])
            word, mask, text_mask, label = self.__getraw__(tokens, labels)
            word = torch.tensor(word).long()
            mask = torch.tensor(mask).long()
            text_mask = torch.tensor(text_mask).long()
            self.__additem__(dataset, word, mask, text_mask, label)
        dataset['sentence_num'] = [len(dataset['word'])]
        if savelabeldic:
            dataset['label2tag'] = [self.label2tag]
        return dataset

    def __getitem__(self, index):
        sample = self.samples[index]
        target_classes = sample['types']
        support = sample['support']
        query = sample['query']
        # add 'O' and make sure 'O' is labeled 0
        distinct_tags = ['O'] + target_classes
        self.tag2label = {tag: idx for idx, tag in enumerate(distinct_tags)}
        self.label2tag = {idx: tag for idx, tag in enumerate(distinct_tags)}
        support_set = self.__populate__(support)
        query_set = self.__populate__(query, savelabeldic=True)
        return support_set, query_set

    def __len__(self):
        return len(self.samples)


def collate_fn(data):
    batch_support = {'word': [], 'mask': [], 'label':[], 'sentence_num':[], 'text_mask':[]}
    batch_query = {'word': [], 'mask': [], 'label':[], 'sentence_num':[], 'label2tag':[], 'text_mask':[]}
    support_sets, query_sets = zip(*data)
    for i in range(len(support_sets)):
        for k in batch_support:
            batch_support[k] += support_sets[i][k]
        for k in batch_query:
            batch_query[k] += query_sets[i][k]
    for k in batch_support:
        if k != 'label' and k != 'sentence_num':
            batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        if k !='label' and k != 'sentence_num' and k!= 'label2tag':
            batch_query[k] = torch.stack(batch_query[k], 0)
    batch_support['label'] = [torch.tensor(tag_list).long() for tag_list in batch_support['label']]
    batch_query['label'] = [torch.tensor(tag_list).long() for tag_list in batch_query['label']]
    return batch_support, batch_query


def get_loader(filepath, tokenizer, N, K, Q, batch_size, max_length,
                       num_workers=8, collate_fn=collate_fn, ignore_index=-1, use_sampled_data=True):
    if not use_sampled_data:
        dataset = FewShotNERDatasetWithRandomSampling(filepath, tokenizer, N, K, Q, max_length,
                                                      ignore_label_id=ignore_index)
    else:
        dataset = FewShotNERDataset(filepath, tokenizer, max_length, ignore_label_id=ignore_index)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             pin_memory=True,
                             num_workers=num_workers,
                             collate_fn=collate_fn)
    return data_loader
