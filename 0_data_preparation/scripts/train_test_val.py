import argparse
import json
import random
import logging

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Jsonl the structured documents")
    parser.add_argument("--train_portion", help="float of the training portion", default=0.7, type=float)
    parser.add_argument("--dev_portion", help="float of the dev portion", default=0.10, type=float)
    parser.add_argument("--test_portion", help="float of the test portion", default=0.20, type=float)
    parser.add_argument("--output_train", help="output train file", default="data.train")
    parser.add_argument("--output_dev", help="output dev file", default="data.dev")
    parser.add_argument("--output_test", help="output test", default="data.test")
    parser.add_argument("--random_seed", help="random seed", default=42, type=int)
    parser.add_argument("--split_level", help="level to split data at (sentence, paragraph, chapter)", default="sentence")
    parser.add_argument("--split_style", help="by percent or explicit token count. changes what to pass to *_portion. Must be percent or count", default="percent")
    args, rest = parser.parse_known_args()

    assert args.split_level in ["sentence", "paragraph", "chapter"], "split level must be one of sentence, paragraph, chapter"
    assert args.split_style in ["percent", "count"], "split style must be percent or count"

    # setup
    random.seed(args.random_seed)
    logging.basicConfig(level=logging.INFO)


    all_texts = []
    texts_train = []
    texts_dev = []
    texts_test = []
            

    with open(args.input, "rt") as fin:
        for line in fin:
            jline = json.loads(line)
            # each line is a book, with book["structure"] holding a list of chapters, with a list of paragraphs, with a list of sentences
            # the first line is a preface usually so to be safe we skip it
            year = jline["year"]
            book_texts = []
            for i, chapter in enumerate(jline["structure"]):
                if i == 0:
                    continue
                if args.split_level == "sentence":
                    for paragraph in chapter:
                        book_texts.extend(paragraph)
                elif args.split_level == "paragraph" or args.split_style == "percent":
                    for paragraph in chapter:
                        book_texts.append(" ".join(paragraph))
                elif args.split_level == "chapter":
                    book_texts.append(" ".join([" ".join(paragraph) for paragraph in chapter]))
            book_texts = [{"year": year, "text": text} for text in book_texts if len(text.split()) > 0]
            all_texts.extend(book_texts)

    logging.info(f"Total {args.split_level}: {len(all_texts)}")

    #random.shuffle(all_texts)

    
    if args.split_style == "percent":
        train_size = int(len(all_texts) * args.train_portion)
        dev_size = int(len(all_texts) * args.dev_portion)
        test_size = len(all_texts) - train_size - dev_size

        logging.info(f"Train size: {train_size}")
        logging.info(f"Dev size: {dev_size}")
        logging.info(f"Test size: {test_size}")

        # randomly select train, dev, test without shuffling
        train_indices = sorted(random.sample(range(len(all_texts)), train_size))
        remaining_indices = set(range(len(all_texts))) - set(train_indices)
        dev_indices = sorted(random.sample(list(remaining_indices), dev_size))
        remaining_indices -= set(dev_indices)
        test_indices = sorted(list(remaining_indices))

        assert len(train_indices) + len(dev_indices) + len(test_indices) == len(all_texts), "Train, dev, and test indices do not cover all texts"
        assert len(set(train_indices).intersection(set(dev_indices))) == 0, "Train and dev indices overlap"
        assert len(set(train_indices).intersection(set(test_indices))) == 0, "Train and test indices overlap"
        print("generating splits with indices:")
        texts_train = [all_texts[i] for i in train_indices]
        texts_dev = [all_texts[i] for i in dev_indices]
        texts_test = [all_texts[i] for i in test_indices]

    #TODO: count doesnt work for continuous training, because the splitting probably 
    # will not be randomised.
    elif args.split_style == "count":

        def get_n_tok_split(n, paras):
            t_c = 0
            split_paragraphs = []
            other_paragraphs = []
            for p in paras:
                p_tok_n = len(p.split())
                if t_c + p_tok_n <= n and p_tok_n > 0:
                    t_c += p_tok_n
                    split_paragraphs.append(p)
                else:
                    other_paragraphs.append(p)
            return t_c, split_paragraphs, other_paragraphs
                
        
        logging.info(f"Extracting {args.train_portion} training tokens, {args.dev_portion} dev tokens and {args.test_portion} test tokens by paragraph")

        t_c, train, leftover = get_n_tok_split(args.train_portion, all_texts)
        logging.info(f"Extracted {t_c} training tokens")
        texts_train = train
                  
        t_c, test, leftover = get_n_tok_split(args.test_portion, leftover)
        logging.info(f"Extracted {t_c} test tokens")
        texts_test = test

        t_c, dev, leftover = get_n_tok_split(args.dev_portion, leftover)
        logging.info(f"Extracted {t_c} dev tokens")
        texts_dev = dev
    
    json.dump(texts_train, open(args.output_train, "wt"), indent=2)
    json.dump(texts_dev, open(args.output_dev, "wt"), indent=2)
    json.dump(texts_test, open(args.output_test, "wt"), indent=2)
    
    logging.info("Done")
    

    
