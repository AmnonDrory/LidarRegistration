from time import time

tic_time = None
def tic():
    global tic_time
    tic_time = time()
def toc(string=None, outfile=None, silent=False):
    global tic_time
    elapsed = time() - tic_time
    if not silent:
        if string is None:
            string = ''
        string +=  " elapsed:" + str(elapsed)
        print(string)
        if outfile is not None:
            print(string, file=outfile)
    return elapsed

class BBR_Timer():
    def __init__(self, name):
        self.acc = 0.0
        self.count = 0
        self.tic_time = None
        self.name = name
    
    def tic(self):
        self.tic_time = time()

    def toc(self, report=True):
        assert self.tic_time is not None, "error: called toc() before calling tic()"
        elapsed = time() - self.tic_time
        self.count += 1
        self.acc += elapsed
        if report:
            print("%s: %f" % ( self.name, self.acc/self.count ) )