
def f(n):
    for skip in range(1, 1000-1):
        if len(range(0, 1000-1, skip)) == n:
            break
    seq = range(0, 1000-1, skip)
    print(len(seq), list(seq))

def g(n):
    skip = float(1000/n)
    seq = [int(skip*i) for i in range(n)]
    print(len(seq), list(seq))


breakpoint()

