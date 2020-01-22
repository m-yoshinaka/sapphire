import fasttext

from sapphire import Sapphire, setting


def run_sapphire():
    print(' * Loading pre-trained model ...', flush=True, end='')
    model = fasttext.FastText.load_model(setting.FASTTEXT_PATH)
    print(' *  - completed')
    
    aligner = Sapphire(model)

    while True:

        sentence_src = ''
        while sentence_src == '':
            print('Input tokenized sentence (A)')
            sentence_src = input('> ')
        if sentence_src == 'exit':
            break

        sentence_trg = ''
        while sentence_trg == '':
            print('Input tokenized sentence (B)')
            sentence_trg = input('> ')
        if sentence_trg == 'exit':
            break

        tokens_src = sentence_src.split()
        tokens_trg = sentence_trg.split()

        result = aligner.align(tokens_src, tokens_trg)
        alignment = result.top_alignment[0][0]

        if not alignment:
            continue

        print('-'.join(['-' for i in range(0, 20)]))
        print('\n * Result')
        print('{0:^20}{1:<6}{2:^20}'.format('A', '', 'B'))

        for al in alignment:
            text_src = ' '.join(tokens_src[al[0] - 1 : al[1]])
            text_trg = ' '.join(tokens_trg[al[2] - 1 : al[3]])
            print('{0:<20}{1:<6}{2:<20}'.format(text_src, '<-->', text_trg))
        
        print('-'.join(['-' for i in range(0, 20)]))


if __name__ == '__main__':
    run_sapphire()
