def two_sum(source: list, target: int):
    result = set()
    for i, num in enumerate(source):
        remain = target - num
        indexes = [
            j for j, value in enumerate(source)
            if value == remain and j != i
        ]
        for j in indexes:
            pair = tuple(sorted((i, j)))  # đảm bảo (nhỏ, lớn)
            result.add(pair)
    return result


source = [6, 5, 7, 1, 4, 2, 3, 2, 4]
target = 8
result = two_sum(source, target)
print(result)
