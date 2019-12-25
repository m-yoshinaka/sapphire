from flask import Flask, render_template, request, redirect, url_for, flash
import fasttext
from sapphire import Sapphire


app = Flask(__name__)

FASTTEXT_PATH = 'model/wiki-news-300d-1M-subword.bin'
ALIGNER_PATH = 'model/sapphire.pkl'
sapphire = None


def prepare():
    print(' * Loading pre-trained model ...', flush=True, end='')
    model = fasttext.FastText.load_model(FASTTEXT_PATH)
    print(' *  - completed')
    global sapphire
    sapphire = Sapphire(model)


def index2text(tokens_a, tokens_b, alignment):
    text_a = []
    text_b = []
    for al in alignment:
        src_s, src_e, trg_s, trg_e = al
        text_a += [' '.join([tokens_a[i - 1] for i in range(src_s, src_e + 1)])]
        text_b += [' '.join([tokens_b[i - 1] for i in range(trg_s, trg_e + 1)])]
    
    return text_a + text_b


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/align', methods=['GET', 'POST'])
def align():
    sent_a = request.form["sent_a"]
    sent_b = request.form["sent_b"]

    tokens_a = sent_a.split()
    tokens_b = sent_b.split()
    result = sapphire.align(tokens_a, tokens_b)
    word_alignment = [(s, s, t, t) for (s, t) in result.word_alignment]
    phrase_alignment = result.top_alignment[0][0]

    word_alignment = index2text(tokens_a, tokens_b, word_alignment)
    phrase_alignment = index2text(tokens_a, tokens_b, phrase_alignment)
    
    return render_template('index.html', sent_a=sent_a, sent_b=sent_b,
                           word_alignment=word_alignment, phrase_alignment=phrase_alignment)


if __name__ == '__main__':
    # app.debug = True
    prepare()
    app.run()
