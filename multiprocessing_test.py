import time
import timeit
import math
import concurrent.futures


def task(i):
    print("starting task...")
    # time.sleep(0.01)
    math.factorial(i)
    print("task done")

def do_tasks_mp():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(task,i) for i in range(100000,100000+10)]

def do_tasks():
    for i in range(100000,100000+10):
        task(i)


if __name__ == "__main__":
    print(timeit.timeit('do_tasks_mp()' ,"from __main__ import do_tasks_mp", number=1))
    print(timeit.timeit('do_tasks()' ,"from __main__ import do_tasks", number=1))
