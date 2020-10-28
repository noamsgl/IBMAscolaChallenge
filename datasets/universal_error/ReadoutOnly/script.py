import pandas

for n in range(7, 11):
    filename = "U3_{}.csv".format(n)
    with open(filename, 'r'):
        df = pandas.read_csv(filename)
        df['population'] = 0
        df.to_csv(filename)
