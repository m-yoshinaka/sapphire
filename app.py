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


def alignment2html(tokens_a, tokens_b, word_alignment, phrase_alignment):
    word_a = {}
    word_b = {}
    for a, b in word_alignment:
        word_a[a] = tokens_b[b - 1]
        word_b[b] = tokens_a[a - 1]

    phrase_a = {}
    phrase_b = {}
    for i, (a_str, a_end, b_str, b_end) in enumerate(phrase_alignment, start=1):
        phrase_a.update({a: i for a in range(a_str, a_end + 1)})
        phrase_b.update({b: i for b in range(b_str, b_end + 1)})

    text_sent = '<div class="parent"><div class="child" id="caption">Sentence {} :&nbsp;</div>'
    text_phrs_word = ' cp_tooltip" id="phrase{}">&nbsp;<span data-tooltip="{}" class="cp_tooltiptext">{}&nbsp;</span></div>'
    text_phrs = '" id="phrase{}">&nbsp;{}&nbsp;</div>'
    text_word = ' cp_tooltip">&nbsp;<span data-tooltip="{}" class="cp_tooltiptext">{}&nbsp;</span></div>'
    text_othr = '">&nbsp;{}&nbsp;</div>'

    text_a = text_sent.format('A')
    for a in range(1, len(tokens_a) + 1):
        text_a += '<div class="child'
        if a in phrase_a.keys() and a in word_a.keys():
            text_a += text_phrs_word.format(phrase_a[a], word_a[a], tokens_a[a - 1])
        elif a in phrase_a.keys() and a not in word_a.keys():
            text_a += text_phrs.format(phrase_a[a], tokens_a[a - 1])
        elif a in word_a.keys():
            text_a += text_word.format(word_a[a], tokens_a[a - 1])
        else:
            text_a += text_othr.format(tokens_a[a - 1])
    text_a += '</div>'
    
    text_b = text_sent.format('B')
    for b in range(1, len(tokens_b) + 1):
        text_b += '<div class="child'
        if b in phrase_b.keys() and b in word_b.keys():
            text_b += text_phrs_word.format(phrase_b[b], word_b[b], tokens_b[b - 1])
        elif b in phrase_b.keys() and b not in word_b.keys():
            text_b += text_phrs.format(phrase_b[b], tokens_b[b - 1])
        elif b in word_b.keys():
            text_b += text_word.format(word_b[b], tokens_b[b - 1])
        else:
            text_b += text_othr.format(tokens_b[b - 1])
    text_b += '</div>'
    
    return text_a + text_b


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/align', methods=['GET', 'POST'])
def align():
    sent_a = request.form['sent_a']
    sent_b = request.form['sent_b']
    tokens_a = sent_a.split()
    tokens_b = sent_b.split()

    result = sapphire.align(tokens_a, tokens_b)
    word_alignment = result.word_alignment
    phrase_alignment = result.top_alignment[0][0]

    output_text = alignment2html(tokens_a, tokens_b, word_alignment, phrase_alignment)
    
    return render_template('result.html', output_text=output_text)

if __name__ == '__main__':
    app.debug = True
    prepare()
    app.run()
