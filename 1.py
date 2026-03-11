# 栈
from stack import Stack

maze = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1, 0, 0, 1, 1, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0, 0, 1, 1, 0, 1],
        [1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]

start = (1, 1)
end = (8, 8)
s = Stack()
s.push(start)

def stack_print(stack):
    ls = []
    while not stack.is_empty():
        item = stack.pop()
        ls.append(item)
    print(ls[::-1])


while not s.is_empty():
    now = s.peek()
    if now == end:
        print("走出来了")
        stack_print(s)
        break
    row, col = now
    maze[row][col] = 2
    if maze[row - 1][col] == 0:
        s.push((row - 1, col))
        continue
    elif maze[row][col + 1] == 0:
        s.push((row, col + 1))
        continue
    elif maze[row + 1][col] == 0:
        s.push((row + 1, col))
        continue
    elif maze[row][col - 1] == 0:
        s.push((row, col - 1))
        continue
    else:
        s.pop()
else:
    print("迷宫走不通")
