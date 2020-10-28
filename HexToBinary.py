class HexToBinary:

    def __init__(self, name):
        self.name = name

    def read_file(self):
        with open(self.name, 'r') as fp:
            hex_list = ["{:02x}".format(ord(c)) for c in fp.read()]
        return hex_list

