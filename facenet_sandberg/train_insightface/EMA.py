
class EMA:
    def __init__(self, s=0.01):
        self.s = s
        self.v = 0
        self.n = 0

    def __call__(self, v):
        if self.n == 0:
            self.v = v
        else:
            self.v = self.v * (1 - self.s) + v * self.s
        self.n = self.n + 1
        return self.v
