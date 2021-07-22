def test(num):
    a = num +6
    b = num +7
    yield a

print(test(1))