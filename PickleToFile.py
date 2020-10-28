import pickle


class PickleToFile:
    def __init__(self, name, pickle_file):
        self.name = name
        self.pickle_file = pickle_file

    def create_pickle_file(self):
        checkThePickle = open(self.pickle_file, "rb")
        with open(self.name, 'w') as filehandler:
            for listItem in checkThePickle:
                filehandler.write('%s\n' % listItem)
