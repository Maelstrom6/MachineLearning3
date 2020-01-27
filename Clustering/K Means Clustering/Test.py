point1 = [1, 2, 3]
point2 = [0, 0, 0]

print(sum([(x-y)**2 for x, y in zip(point1, point2)]))