class MyQueue:
    def __init__(self, capacity: int):
        self.__capacity = capacity
        self.__queue = []

    def getQueue(self):
        return self.__queue

    def isEmpty(self):
        return True if len(self.__queue) == 0 else False

    def isFull(self):
        return True if len(self.__queue) == self.__capacity else False

    def dequeue(self):
        return self.__queue.pop(0)

    def enqueue(self, value):
        return self.__queue.append(value)

    def front(self):
        return self.__queue[0]
