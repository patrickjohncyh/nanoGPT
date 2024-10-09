import os
import random

max_int = 100
numDec1 = 0  # number of decimals to round to for the first number
ops = ["+", "-", "*", "num"]
percentMode = True


# Generates a random equation
def expr(max_depth):
    if max_depth == 1:
        # return terminal
        return random.randint(0, max_int)
    else:
        op = random.choice(ops)
        if op == "num":
            return random.randint(0, max_int)

        l_op = expr(max_depth - 1)
        r_op = expr(max_depth - 1)
        return "({} {} {})".format(l_op, op, r_op)


def generate(num_samples=10000):
    cnt = 0
    out = []
    while cnt < num_samples:
        if cnt % 1001 == 1:
            print(cnt)
        sample = expr(3)
        try:
            ans = eval(sample)
            if abs(ans) > 100:
                assert False
            sample_str = "Q: {} ? A: {}".format(sample, ans)
            out.append(sample_str)
            cnt += 1
            # print(sample_str)
        except:
            pass
    return out


if __name__ == "__main__":
    samples = generate(10000)
    with open(os.path.join(os.path.dirname(__file__), "arithmetic.txt"), "w") as f:
        f.write("\n".join(samples))
    f.close()
