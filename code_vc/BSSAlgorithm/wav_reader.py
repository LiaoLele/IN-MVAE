import os
import fnmatch
import soundfile as sf

class WavReader(object):
    """WavReader
        find wav files recursively and read audio data from the wav files
    """
    def __init__(self, path):
        self.path = path
        self.files = self.find_files(self.path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.read(self.files[index])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def filename(self, index):
        return self.files[index]

    @staticmethod
    def find_files(path, ptn="*.wav"):
        file_list = []
        for root, _, files in os.walk(path):
            wav_files = fnmatch.filter(files, ptn)
            wav_files = [os.path.join(root, f) for f in wav_files]
            file_list.extend(wav_files)
        file_list.sort()
        return file_list

    @staticmethod
    def read(filename):
        return sf.read(filename, always_2d=True)


if __name__ == "__main__":
    reader =  WavReader('/home/nis/lele.liao/projects/FastIVE/mixture')
    for data, fs in reader:
        print(data)

    