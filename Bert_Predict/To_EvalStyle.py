import pandas as pd
import csv


def readPred(file):
    pred = []
    with open(file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for prediction in reader:
            if prediction[0] > prediction[1]:
                pred.append('False')
            elif prediction[0] < prediction[1]:
                pred.append('True')

    return pred


def totxt(df, length, predictions):
    id = []
    for i in range(length):
        num = df.loc[i, 'ArticleID']
        if num < 10:
            str_id = '000000' + str(num)
        elif num < 100:
            str_id = '00000' + str(num)
        elif num < 1000:
            str_id = '0000' + str(num)
        elif num < 10000:
            str_id = '000' + str(num)
        elif num < 100000:
            str_id = '00' + str(num)
        elif num < 1000000:
            str_id = '0' + str(num)
        else:
            str_id = str(num)
        id.append(str_id)

    col = {'article id': id,
           'prediction': predictions}

    tf = pd.DataFrame(col)
    tf.to_csv('/home/lstrauch/Bachelorarbeit/env/Predictions/Predictions_BERT_Article-Publisher.txt', encoding='utf-8',
              index=False, sep=' ', header=None)


def main():
    df = pd.read_csv('/home/lstrauch/Bachelorarbeit/env/Data/Preprocessed_ByPublisher.csv', encoding='utf-8')
    predictions = readPred('/home/lstrauch/Bachelorarbeit/env/Bert/Output_Article/test_results.tsv')
    totxt(df, 600000, predictions)


main()
