with open("hostfile-helin-all", "w") as f:
    for x in [1, 2]:
        f.write("r01c02s0{} slots=10\n".format(x))
    
    for x in range(3, 8):
        for y in range(1, 5):
            f.write("r01c0{}s0{} slots=10\n".format(x, y))

    for x in range(1, 10):
        for y in range(1, 5):
            f.write("r02c0{}s0{} slots=10\n".format(x, y))

    for x in range(10, 14):
        for y in range(1, 5):
            f.write("r02c{}s0{} slots=10\n".format(x, y))
