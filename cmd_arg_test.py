import sys, getopt, math
from progress_bar import *



# def progress_bar(progress, total):
#     percent = 100 * (progress/ float(total))
#     bar = '+' * int(percent) + '-' * (100-int(percent))
#     print(f"\r|{bar}|{percent:.2f}%",end="\r")
#     if progress == total:
#         print(f"\r|{bar}|{percent:.2f}%")



def main(argv):
    
    try: opts, args = getopt.getopt(argv,'t:', ["total="])
    except:
        print("error..")
        sys.exit(2)

    for opt, arg, in opts:
        if opt in ("-t", "--total"):
            total = int(arg)

    # print("total is {}".format(total))


    progress_bar(0, total)
    for i in range(total):
        math.factorial(100000)
        progress_bar(i+1, total)








if __name__ == "__main__":
    main(sys.argv[1:])
