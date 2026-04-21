from nn_engine import MLP
from exp_visualzer import ExpViz

X = [
    [1,2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

m = MLP(3, [3])
print(len(m.parameters()))

O = []

for x in X:
    O.append(m(x))

s = 0
c = len(O) * len(O[0])

for i,j in enumerate(O):
    c = 0
    for k,val in enumerate(j):
        O[i][k] = O[i][k].exp()
        c += O[i][k]
        s += O[i][k]

mean = s/c

print(mean)

ExpViz.viz_expr(mean)




# ExpViz.viz_expr(O[0][0])