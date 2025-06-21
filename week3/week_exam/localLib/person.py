from abc import ABC, abstractmethod


class Person(ABC):
    def __init__(self, name: str, yob: int):
        self._name = name
        self._yob = yob

    @abstractmethod
    def describe(self):
        pass


class Student(Person):
    def __init__(self, name: str, yob: int, grade: str):
        super().__init__(name, yob)
        self.__grade = grade   # Thuộc tính riêng của lớp con

    def describe(self) -> str:
        return (f"Student - Name: {self._name} - "
                f"YoB: {self._yob} - "
                f"Grade: {self.__grade}")


class Teacher(Person):
    def __init__(self, name: str, yob: int, subject: str):
        super().__init__(name, yob)
        self.__subject = subject   # Thuộc tính riêng của lớp con

    def getYob(self):
        return self._yob

    def describe(self) -> str:
        return (f"Teacher - Name: {self._name} - "
                f"YoB: {self._yob} - "
                f"Grade: {self.__subject}")


class Doctor(Person):
    def __init__(self, name: str, yob: int, specialist: str):
        super().__init__(name, yob)
        self.__specialist = specialist   # Thuộc tính riêng của lớp con

    def describe(self) -> str:
        return (f"Doctor - Name: {self._name} - "
                f"YoB: {self._yob} - "
                f"Grade: {self.__specialist}")
