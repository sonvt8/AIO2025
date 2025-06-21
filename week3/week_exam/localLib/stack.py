class MyStack:
    def __init__(self, capacity: int):
        self.__capacity = capacity
        self.__stack = []

    def getStack(self):
        return self.__stack

    def isEmpty(self):
        return True if len(self.__stack) == 0 else False

    def isFull(self):
        return True if len(self.__stack) == self.__capacity else False

    def pop(self):
        return self.__stack.pop(-1)

    def push(self, value):
        return self.__stack.append(value)

    def top(self):
        return self.__stack[-1]
