import sys
def main(argv):
    nums=[int(x) for x in argv]
    print(sum(nums))
if __name__=='__main__': main(sys.argv[1:])
