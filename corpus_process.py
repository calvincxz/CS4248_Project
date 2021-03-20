import json

sms_corpus_path = r'../Data/archive/smsCorpus_en_2015.03.09_all.json'

def get_sms_data():
    sms_corpus_text_list = []
    with open(sms_corpus_path, 'r') as sms_corpus_file:
        sms_corpus_data = json.loads(sms_corpus_file.read())
        sms_corpus_data = sms_corpus_data['smsCorpus']['message']
        sms_corpus_text_list = [str(msg['text']['$']) for msg in sms_corpus_data]
    return sms_corpus_text_list


if __name__ == '__main__':
    sms_corpus_text_list = get_sms_data()
    print(sms_corpus_text_list)
    print(len(sms_corpus_text_list))