import tensorflow


node1 = tensorflow.constant(-1)
node2 = tensorflow.constant(4)


def sum():
    return node1 + node2


result = sum()
print(result)
