import speechbrain as sb
import pandas as pd
class SpeakerEncoder():
    def __init__(self, csv_train_path, csv_valid_path, csv_test_path):

        self.encoder = sb.dataio.encoder.CategoricalEncoder()
        fileName = './spkr_encoded.txt'
        if not self.encoder.load_if_possible(fileName):
            df_train = pd.read_csv(csv_train_path)
            df_valid = pd.read_csv(csv_valid_path)
            df_test = pd.read_csv(csv_test_path)

            data = [df_train, df_valid, df_test]
            df_merged = pd.concat(data)
            #print('speaker count' + str(self.speaker_count))
            for index, row in df_merged.iterrows():
                self.encoder.ensure_label(row["spk_id"])
            self.encoder.save(fileName)
           
        self.speaker_count = 1251 #hardcoded for now vox1celeb1 count #df_merged['spk_id'].nunique()

    def get_speaker_labels_encoded(self, labels):
        return self.encoder.encode_label_torch(labels)

    def get_speaker_label_encoded(self, label):
        return self.encoder.encode_label(label)

    def get_total_speaker_count(self):
        return self.speaker_count