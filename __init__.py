import sys
if sys.hexversion < 0x30600f0:
    print("must use python 3.6+. current is {}".format(sys.version))
    sys.exit(-1)