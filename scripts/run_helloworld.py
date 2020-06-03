###############################################################################
# python scripts/run_helloworld.pyで動くように設計 ############################
###############################################################################
import argparse

def set_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument('-it', '--input_text', default='Hello World!')

    args = parser.parse_args()
    return args.__dict__


class HelloWorld(object):
    """Documentation for HelloWorld
    Examples
    --------
    python scripts/run_helloworld.py
    Hello World!
    """
    def __init__(self, text):
        super(HelloWorld, self).__init__()
        self.text = text

    def pprint(self, text=None):
        """Hello Worldをprintするモジュール

        Parameters
        ----------
        text : string
            出力する文字列
        """
        if text is None:
            text = self.text
        print(text)


def main(input_text):
    helloworld = HelloWorld(input_text)
    helloworld.pprint()


if __name__ == '__main__':
    args = set_argument()
    main(**args)
