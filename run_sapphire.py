import logging
import argparse
import fasttext

from sapphire import Sapphire


formatter = '%(levelname)s : %(asctime)s : %(message)s'
logging.basicConfig(level=logging.INFO, format=formatter)


def run_sapphire(args):
    logging.info('loading pre-trained model')
    model = fasttext.FastText.load_model(args.model_path)
    logging.info('loading completed')

    aligner = Sapphire(model=model)
    aligner.set_params(lambda_=args.lambda_,
                       delta=args.delta,
                       alpha=args.alpha,
                       hungarian=args.use_hungarian)

    try:
        while True:
            print('\n' + '=' * 80)
            sentence_src = input('Input tokenized sentence (A)\n>>> ')
            sentence_trg = input('Input tokenized sentence (B)\n>>> ')

            if sentence_src == '' or sentence_trg == '':
                logging.warning('please input two sentences!')
                continue

            tokens_src = sentence_src.split()
            tokens_trg = sentence_trg.split()
            _, alignment = aligner(tokens_src, tokens_trg)

            print('{:-^48}'.format(' Result '))
            print('{0:^24}{1:^24}'.format('Sentence A', 'Sentence B'))

            for src_s, src_e, trg_s, trg_e in alignment:
                src_txt = ' '.join(tokens_src[src_s - 1:src_e])
                trg_txt = ' '.join(tokens_trg[trg_s - 1:trg_e])
                print('{0:>20}{1:^8}{2:<20}'.format(src_txt, '<-->', trg_txt))

    except KeyboardInterrupt:
        print()
        logging.info('interrupted')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path',
                        default='model/wiki-news-300d-1M-subword.bin',
                        help='path to fastText model')
    parser.add_argument('--lambda_', type=float, default=0.6,
                        help='threshold of word alignment candidate score')
    parser.add_argument('--delta', type=float, default=0.6,
                        help='threshold of phrase alignment candidate score')
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='bias for length of phrase')
    parser.add_argument('--use_hungarian', action='store_true',
                        help='use Hungarian-method for word alignment')
    args = parser.parse_args()

    run_sapphire(args)


if __name__ == '__main__':
    main()
