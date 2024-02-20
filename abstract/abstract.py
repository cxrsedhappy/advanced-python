from __future__ import annotations


class Message:
    def __init__(self, text: str):
        self.text: str = text
        self.author: User

    def __repr__(self):
        return f'{self.text}'


class User:
    def __init__(self, name: str):
        self.name: str = name

    def send(self, text: Message):
        print(f'{self.name} send {text}')


class Guild:
    def __init__(self, name: str):
        self.name: str = name
        self._members: list[User] = []

    @property
    def members(self):
        return self._members
