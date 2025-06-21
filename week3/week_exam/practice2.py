from localLib.person import Student, Doctor, Teacher
from localLib.ward import Ward

student = Student(name="studentA", yob=2010, grade=7)
teacher1 = Teacher(name="teacherA", yob=1969, subject="Math")
teacher2 = Teacher(name="teacherB", yob=1995, subject="History")
doctor1 = Doctor(name="doctorA", yob=1945, specialist="Endocrinologists")
doctor2 = Doctor(name="doctorB", yob=1975, specialist="Cardiologists")

print(student.describe())
print(teacher1.describe())
print(doctor1.describe())

print("="*20)

listPeople = [student, teacher1, teacher2, doctor1, doctor2]
ward = Ward(name="Ward1")
ward.addPerson(student)
ward.addPerson(teacher1)
ward.addPerson(teacher2)
ward.addPerson(doctor1)
ward.addPerson(doctor2)
ward.describe()

print("="*20)
print(f"Number of doctors: {ward.countDoctor()}")

print("="*20)
print("After sorting Age of Ward1 people")
ward.sortAge()
ward.describe()

print("="*20)
print(f"Average year of birth (teachers): {ward.aveTeacherYearOfBirth()}")
