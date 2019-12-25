from sapphire import Sapphire


def run_sapphire():
    aligner = Sapphire()

    while True:

        sentence_src = ''
        while sentence_src == '':
            print('Input sentence1')
            sentence_src = input('> ')
        if sentence_src == 'exit':
            break

        sentence_trg = ''
        while sentence_trg == '':
            print('Input sentence2')
            sentence_trg = input('> ')
        if sentence_trg == 'exit':
            break

        tokens_src = sentence_src.split()
        tokens_trg = sentence_trg.split()

        alignment = aligner.align(tokens_src, tokens_trg)
        print('\n{}\n'.format(alignment))

        if '-' in alignment:
            for al in alignment.split(' '):
                srcs, trgs = al.split('-')
                src_s = int(srcs.split(',')[0])
                src_e = int(srcs.split(',')[-1])
                trg_s = int(trgs.split(',')[0])
                trg_e = int(trgs.split(',')[-1])
                text_src = ' '.join(tokens_src[src_s - 1:src_e])
                text_trg = ' '.join(tokens_trg[trg_s - 1:trg_e])
                print('{} <--> {}'.format(text_src, text_trg))
        
        print('---')


if __name__ == '__main__':
    run_sapphire()
