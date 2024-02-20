def collatz(n: int):
    while True:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        yield n

        if n == 1:
            break


def values():
    yield 'hi'
    yield 'mum'
    yield '!'


def main():
    for value in values():
        print(value)

    seq = list(collatz(413342154))
    print(seq)


if __name__ == '__main__':
    main()