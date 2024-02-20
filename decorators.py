import time


def get_time(func):
    def wrapper():
        start = time.time()
        func()
        total_time = time.time() - start
        print(f'{total_time:.3f} seconds')

    return wrapper


@get_time
def main():
    print(' '.join([str(i) for i in range()]))


if __name__ == '__main__':
    main()
