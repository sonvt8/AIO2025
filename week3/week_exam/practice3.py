from localLib.stack import MyStack
from localLib.queue import MyQueue

stack1 = MyStack(capacity=5)

stack1.push(1)
stack1.push(2)

print(f"Stack before:{stack1.getStack()}")
print(stack1.isFull())
print(stack1.top())
print(stack1.pop())
print(stack1.pop())
print(stack1.isEmpty())
print(f"Stack after:{stack1.getStack()}")

print("="*40)

queue1 = MyQueue(capacity=5)

queue1.enqueue(1)
queue1.enqueue(2)

print(f"Queue before:{queue1.getQueue()}")
print(queue1.isFull())
print(queue1.front())
print(queue1.dequeue())
print(queue1.front())
print(queue1.dequeue())
print(queue1.isEmpty())
print(f"Queue after:{queue1.getQueue()}")
