from localLib.person import Person, Doctor, Teacher


class Ward:
    def __init__(self, name: str):
        self._name = name
        self.__citizens: list[Person] = []

    def getList(self):
        return self.__citizens

    def addPerson(self, person: Person):
        return self.__citizens.append(person)

    def countDoctor(self):
        return len([
            doctor
            for doctor in self.__citizens
            if isinstance(doctor, Doctor)
        ])

    def sortAge(self):
        self.__citizens.sort(key=lambda person: person._yob, reverse=True)

    def aveTeacherYearOfBirth(self):
        teachers = [
            teacher
            for teacher in self.__citizens
            if isinstance(teacher, Teacher)
        ]

        return sum(t.getYob() for t in teachers) / len(teachers)

    def describe(self):
        print(f"Ward Name: {self._name}")
        for idx, people in enumerate(self.__citizens, start=1):
            print(f"{idx}. {people.describe()}")
