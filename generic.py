from typing import Generic, TypeVar

T = TypeVar('T')


class Overflow(Exception):
    def __init__(self, msg: str = 'Overflowed'):
        self.msg = msg

    def __str__(self):
        return f'{self.msg}'


class Queue(Generic[T]):
    def __init__(self, limit: int):
        self._elements: T = []
        self._limit: int = limit

    @property
    def elements(self) -> list:
        return self._elements

    def push(self, value: T):
        if len(self._elements) == self._limit:
            raise Overflow()

        self._elements.append(value)

    def get(self) -> T:
        if len(self._elements) <= 0:
            return None

        return self._elements.pop(0)

    def __len__(self) -> int:
        return len(self._elements)


if __name__ == '__main__':
    q = Queue[int](5)
    for i in range(5):
        q.push(i)
    print(len(q))
